import io
import tempfile

import click
import experiments.utils as utils
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pyparsing as pp
import xcsf.xcsf as xcsf
from sklearn import metrics  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tasks.book.generated_function import generate

from . import log_xcs_params


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
def run_experiment(n_iter, seed, show, sample_size):

    mlflow.set_experiment("xcsf.generated_function")
    with mlflow.start_run() as run:

        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        X, y = generate(sample_size)
        X_test, y_test_true = generate(1000, random_state=12345)

        # generate equidistant, denoised data as well (only for visual reference)
        X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
        _, y_denoised = generate(1000, noise=False, X=X_denoised)

        utils.log_array(X, "X")
        utils.log_array(y, "y")
        utils.log_array(X_test, "X_test")
        utils.log_array(y_test_true, "y_test_true")
        utils.log_array(X_denoised, "X_denoised")
        utils.log_array(y_denoised, "y_denoised")

        scaler_X = StandardScaler()
        # Preen scales outputs to [0, 1].
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        X = scaler_X.fit_transform(X)
        X_test = scaler_X.transform(X_test)
        y = scaler_y.fit_transform(y)
        y_test_true = scaler_y.transform(y_test_true)

        xcs = xcsf.XCS(X.shape[1], y.shape[1], 1)  # only 1 (dummy) action

        xcs.action("integer")  # (dummy) integer actions

        # NOTE Preen: “Hyperrectangles and hyperellipsoids currently use the
        # centre-spread representation (and axis-rotation is not yet implemented.)”
        args = {
            "min": -1,  # minimum value of a center
            "max": 1,  # maximum value of a center
            "spread-min": 0.1,  # minimum initial spread
            "eta":
            0,  # disable gradient descent of centers towards matched input mean
        }
        xcs.condition("hyperrectangle", args)
        mlflow.log_param("xcs.condition", "hyperrectangle")
        mlflow.log_param("xcs.condition.args.min", args["min"])
        mlflow.log_param("xcs.condition.args.max", args["max"])
        mlflow.log_param("xcs.condition.args.spread-min", args["spread-min"])
        mlflow.log_param("xcs.condition.args.eta", args["eta"])

        args = {
            "x0": 1,  # bias attribute
            "rls-scale-factor":
            1000,  # initial diagonal values of the gain-matrix
            "rls-lambda": 1,  # forget rate (small values may be unstable)
        }
        xcs.prediction("rls-linear", args)
        mlflow.log_param("xcs.prediction", "rls-linear")
        mlflow.log_param("xcs.prediction.args.x0", args["x0"])
        mlflow.log_param("xcs.prediction.args.rls-scale-factor",
                         args["rls-scale-factor"])
        mlflow.log_param("xcs.prediction.args.rls-lambda", args["rls-lambda"])

        xcs.OMP_NUM_THREADS = 8  # number of CPU cores to use
        xcs.POP_SIZE = 1000  # maximum population size
        xcs.MAX_TRIALS = n_iter  # number of trials per fit()
        xcs.LOSS_FUNC = "mse"  # mean squared error
        xcs.SET_SUBSUMPTION = True
        xcs.EA_SUBSUMPTION = True
        xcs.E0 = 0.005  # target error
        log_xcs_params(xcs)

        xcs.fit(X, y, True)

        y_test = xcs.predict(X_test)

        X = scaler_X.inverse_transform(X)
        X_test = scaler_X.inverse_transform(X_test)
        y = scaler_y.inverse_transform(y)
        y_test = scaler_y.inverse_transform(y_test)

        mae = metrics.mean_absolute_error(y_test_true, y_test)
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true,
                              y_test)  # TODO Has to be done in-sample?
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2-score", r2)

        f = tempfile.NamedTemporaryFile(prefix=f"model-", suffix=f".model")
        xcs.save(f.name)
        mlflow.log_artifact(f.name)
        f.close()

        out = io.BytesIO()
        with utils.stdout_redirector(out):
            xcs.print_pset(True, True, True)

        pop = parse_pop(out.getvalue().decode("utf-8"))
        utils.log_json(pop, "population")

        print(f"number of rules: {xcs.pset_size()}")
        mlflow.log_metric("xcs.size", xcs.pset_size())

        # next start plotting posthoc from mlflow, i guess (and not from here)

        # TODO Plot single classifier predictions?
        # y_cls = scaler_y.inverse_transform(y_cls)

        fig, ax = utils.plot_prediction(X=X,
                                        y=y,
                                        X_test=X_test,
                                        y_test=y_test,
                                        X_denoised=X_denoised,
                                        y_denoised=y_denoised)
        utils.save_plot(fig, seed)

        if show:
            plt.show()


if __name__ == "__main__":
    run_experiment()
