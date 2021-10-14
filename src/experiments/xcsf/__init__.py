import io
import tempfile

import experiments.utils as utils
import matplotlib.pyplot as plt
import mlflow
import pyparsing as pp
import xcsf.xcsf as xcsf
from experiments.utils import log_array, plot_prediction, save_plot
from sklearn import metrics  # type: ignore
from sklearn.preprocessing import StandardScaler


def log_xcs_params(xcs):
    mlflow.log_param("xcs.OMP_NUM_THREADS", xcs.OMP_NUM_THREADS)
    mlflow.log_param("xcs.POP_INIT", xcs.POP_INIT)
    mlflow.log_param("xcs.POP_SIZE", xcs.POP_SIZE)
    mlflow.log_param("xcs.MAX_TRIALS", xcs.MAX_TRIALS)
    mlflow.log_param("xcs.PERF_TRIALS", xcs.PERF_TRIALS)
    mlflow.log_param("xcs.LOSS_FUNC", xcs.LOSS_FUNC)
    mlflow.log_param("xcs.HUBER_DELTA", xcs.HUBER_DELTA)
    mlflow.log_param("xcs.E0", xcs.E0)
    mlflow.log_param("xcs.ALPHA", xcs.ALPHA)
    mlflow.log_param("xcs.NU", xcs.NU)
    mlflow.log_param("xcs.BETA", xcs.BETA)
    mlflow.log_param("xcs.DELTA", xcs.DELTA)
    mlflow.log_param("xcs.THETA_DEL", xcs.THETA_DEL)
    mlflow.log_param("xcs.INIT_FITNESS", xcs.INIT_FITNESS)
    mlflow.log_param("xcs.INIT_ERROR", xcs.INIT_ERROR)
    mlflow.log_param("xcs.M_PROBATION", xcs.M_PROBATION)
    mlflow.log_param("xcs.STATEFUL", xcs.STATEFUL)
    mlflow.log_param("xcs.SET_SUBSUMPTION", xcs.SET_SUBSUMPTION)
    mlflow.log_param("xcs.THETA_SUB", xcs.THETA_SUB)
    mlflow.log_param("xcs.COMPACTION", xcs.COMPACTION)
    mlflow.log_param("xcs.TELETRANSPORTATION", xcs.TELETRANSPORTATION)
    mlflow.log_param("xcs.GAMMA", xcs.GAMMA)
    mlflow.log_param("xcs.P_EXPLORE", xcs.P_EXPLORE)
    mlflow.log_param("xcs.EA_SELECT_SIZE", xcs.EA_SELECT_SIZE)
    mlflow.log_param("xcs.THETA_EA", xcs.THETA_EA)
    mlflow.log_param("xcs.LAMBDA", xcs.LAMBDA)
    mlflow.log_param("xcs.P_CROSSOVER", xcs.P_CROSSOVER)
    mlflow.log_param("xcs.ERR_REDUC", xcs.ERR_REDUC)
    mlflow.log_param("xcs.FIT_REDUC", xcs.FIT_REDUC)
    mlflow.log_param("xcs.EA_SUBSUMPTION", xcs.EA_SUBSUMPTION)
    mlflow.log_param("xcs.EA_PRED_RESET", xcs.EA_PRED_RESET)


def parse_pop(s):
    """
    Parses the output of XCSF's ```print_pset()```.

    For now, only rectangular conditions are supported.
    """
    num = pp.pyparsing_common.number
    sup = pp.Suppress

    header = sup("***********************************************\n")

    condition = pp.Group(
        pp.MatchFirst([
            sup("CONDITION\nrectangle: (l=") + num.set_results_name("lower")
            + sup(", u=") + num.set_results_name("upper") + sup(")\n"),
            sup("CONDITION\nrectangle: (c=") + num.set_results_name("center")
            + sup(", s=") + num.set_results_name("spread") + sup(")\n"),
        ])).set_results_name("condition")

    predictor = pp.Group(
        sup("PREDICTOR\nRLS weights: ") + num + sup(", ") + num
        + sup(",")).set_results_name("prediction")

    action = (sup("ACTION\n") + num.set_results_name("action"))

    parameters = pp.Group(
        sup("err=") + num.set_results_name("error") + sup("fit=")
        + num.set_results_name("fitness") + sup("num=")
        + num.set_results_name("numerosity") + sup("exp=")
        + num.set_results_name("experience") + sup("size=") + num
        + sup("time=") + num + sup("age=") + num + sup("mfrac=")
        + num).set_results_name("parameters")

    rule = pp.Group(header + condition + predictor + action + parameters)

    rules = pp.OneOrMore(rule)

    res = [x.as_dict() for x in rules.parse_string(s)]

    return res


def experiment(name,
               X,
               y,
               X_test,
               y_test_true,
               X_denoised,
               y_denoised,
               n_iter,
               pop_size,
               seed,
               show,
               sample_size,
               standardize=False):
    mlflow.set_experiment(name)
    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        log_array(X, "X")
        log_array(y, "y")
        log_array(X_test, "X_test")
        log_array(y_test_true, "y_test_true")
        log_array(X_denoised, "X_denoised")
        log_array(y_denoised, "y_denoised")

        if standardize:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
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

        xcs.seed(seed)

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
        xcs.POP_SIZE = pop_size  # maximum population size
        xcs.MAX_TRIALS = n_iter  # number of trials per fit()
        xcs.LOSS_FUNC = "mse"  # mean squared error
        xcs.SET_SUBSUMPTION = True
        xcs.EA_SUBSUMPTION = True
        xcs.E0 = 0.005  # target error
        log_xcs_params(xcs)

        xcs.fit(X, y, True)

        # make predictions for test data
        y_test = xcs.predict(X_test)

        # TODO get unmixed classifier predictions

        if standardize:
            X = scaler_X.inverse_transform(X)
            X_test = scaler_X.inverse_transform(X_test)
            y = scaler_y.inverse_transform(y)
            y_test = scaler_y.inverse_transform(y_test)

        log_array(y_test, "y_test")

        # get metrics
        mae = metrics.mean_absolute_error(y_test_true, y_test)
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        mlflow.log_metric("mae", mae, n_iter)
        mlflow.log_metric("mse", mse, n_iter)
        mlflow.log_metric("r2-score", r2, n_iter)

        mlflow.log_metric("size", xcs.pset_size())

        # store the model, you never know when you need it
        f = tempfile.NamedTemporaryFile(prefix=f"model-", suffix=f".model")
        xcs.save(f.name)
        mlflow.log_artifact(f.name)
        f.close()

        out = io.BytesIO()
        with utils.stdout_redirector(out):
            xcs.print_pset(True, True, True)

        pop = parse_pop(out.getvalue().decode("utf-8"))
        utils.log_json(pop, "population")

        fig, ax = plot_prediction(X=X,
                                  y=y,
                                  X_test=X_test,
                                  y_test=y_test,
                                  X_denoised=X_denoised,
                                  y_denoised=y_denoised)
        save_plot(fig, seed)

        if show:
            plt.show()
