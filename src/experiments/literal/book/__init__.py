import joblib as jl
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore
from berbl import BERBL
from berbl.literal.hyperparams import HParams
from berbl.search.operators.drugowitsch import DefaultToolbox
from experiments.utils import log_array, plot_prediction, save_plot
from sklearn import metrics  # type: ignore
from sklearn.compose import TransformedTargetRegressor  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


def experiment(name,
               matchcls,
               gaparams,
               X,
               y,
               X_test,
               y_test_true,
               X_denoised,
               y_denoised,
               n_iter,
               seed,
               show,
               sample_size,
               literal=False,
               standardize=False):
    mlflow.set_experiment(name)
    with mlflow.start_run() as run:
        mlflow.log_params(HParams().__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)
        mlflow.log_params("literal", literal)
        mlflow.log_params("standardize", standardize)

        log_array(X, "X")
        log_array(y, "y")
        log_array(X_test, "X_test")
        log_array(y_test_true, "y_test_true")
        log_array(X_denoised, "X_denoised")
        log_array(y_denoised, "y_denoised")

        random_state = check_random_state(seed)

        toolbox = DefaultToolbox(
            matchcls=matchcls,
            # TODO Get rid of unneeded gaparams dict
            n=gaparams["n"],
            p=gaparams["p"],
            tournsize=gaparams["tournsize"],
            literal=literal,
            fit_mixing="laplace",
            random_state=random_state)

        estimator = BERBL(toolbox, search="drugowitsch", n_iter=n_iter)

        if standardize:
            estimator = make_pipeline(
                StandardScaler(),
                TransformedTargetRegressor(regressor=estimator,
                                           transformer=StandardScaler()))

        estimator = estimator.fit(X, y)

        # make predictions for test data
        y_test, var = estimator.predict_mean_var(X_test)

        # get unmixed classifier predictions
        y_cls = estimator.predicts(X)

        log_array(y_test, "y_test")
        log_array(var, "var")
        log_array(np.hstack(y_cls), "y_cls")

        # two additional statistics to maybe better gauge solution performance
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        mlflow.log_metric("elitist.size", estimator.search_.size_[0], n_iter)
        mlflow.log_metric("elitist.p_M_D", estimator.search_.p_M_D_[0], n_iter)
        mlflow.log_metric("elitist.mse", mse, n_iter)
        mlflow.log_metric("elitist.r2-score", r2, n_iter)

        # TODO Reimplement copying properly
        # model_file = f"models/Model {seed}.joblib"
        # jl.dump(estimator, model_file)
        # mlflow.log_artifact(model_file)

        fig, ax = plot_prediction(X=X,
                                  y=y,
                                  X_test=X_test,
                                  y_test=y_test,
                                  var=var,
                                  X_denoised=X_denoised,
                                  y_denoised=y_denoised)

        plot_cls(X=X, y=y_cls, ax=ax)
        add_title(ax, estimator.search_.size_[0], estimator.search_.p_M_D_[0],
                  mse, r2)
        save_plot(fig, seed)

        if show:
            plt.show()


def plot_cls(X, y, ax=None):
    """
    Parameters
    ----------
    X : array of shape (N, 1)
        Points for which the classifiers made predictions.
    y : array of shape (K, N, Dy)
        Predictions of the ``K`` classifiers.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    for k in range(len(y)):
        ax.plot(X.ravel(),
                y[k],
                c="grey",
                linestyle="-",
                linewidth=0.5,
                alpha=0.7,
                zorder=10)

    return fig, ax


def add_title(ax, K, p_M_D, mse, r2):
    # add metadata to plot for ease of use
    ax.set(title=(f"K = {K}, "
                  f"p(M|D) = {(p_M_D):.2}, "
                  f"mse = {mse:.2}, "
                  f"r2 = {r2:.2}"))
