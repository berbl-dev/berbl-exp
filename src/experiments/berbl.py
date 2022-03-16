from copy import deepcopy
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore
from berbl import BERBL
from berbl.literal.hyperparams import HParams
from berbl.match.radial1d_drugowitsch import RadialMatch1D
from berbl.match.softinterval1d_drugowitsch import SoftInterval1D
from berbl.search.operators.drugowitsch import DefaultToolbox
from sklearn import metrics  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from . import Experiment, maybe_override
from .utils import log_array, plot_prediction, save_plot


class BERBLExperiment(Experiment):
    algorithm = "berbl"

    def init_estimator(self):
        self.n_iter = self.params["n_iter"]

        if self.params["literal"]:
            mlflow.log_params(HParams().__dict__)

        random_state = check_random_state(self.seed)

        if self.params["match"] == "radial":
            matchcls = RadialMatch1D
        elif self.params["match"] == "softint":
            matchcls = SoftInterval1D
        else:
            print(f"Unsupported match function family: {self.params['match']}")

        try:
            match_args = self.params["match_args"]
        except:
            match_args = {}

        # Overrides for low level parameters (i.e. for ``Mixing``, ``Rule``,
        # ``Mixture`` etc.). We want to keep ``self.params`` as it is but cannot
        # have the same keyward argument to ``DefaultToolbox`` repeated.
        low_params = deepcopy(self.params)
        del low_params["n"]
        del low_params["p"]
        del low_params["tournsize"]
        del low_params["literal"]
        del low_params["fit_mixing"]
        try:
            del low_params["match_args"]
        except:
            pass
        del low_params["match"]

        toolbox = DefaultToolbox(matchcls=matchcls,
                                 n=self.params["n"],
                                 p=self.params["p"],
                                 tournsize=self.params["tournsize"],
                                 literal=self.params["literal"],
                                 fit_mixing=self.params["fit_mixing"],
                                 random_state=random_state,
                                 match_args=match_args,
                                 # Provide all parameters as keyword arguments.
                                 # This way, we can override Rule, Mixing and
                                 # Mixture hyperparameters from
                                 # Experiment.run().
                                 **low_params)

        self.estimator = BERBL(toolbox,
                               search="drugowitsch",
                               n_iter=self.n_iter)

    def evaluate(self, X, y, X_test, y_test_true, X_denoised, y_denoised):
        # make predictions for test data
        y_test, var = self.estimator.predict_mean_var(X_test)

        # get unmixed classifier predictions
        y_cls = self.estimator.predicts(X)

        log_array(y_test, "y_test")
        log_array(var, "var")
        log_array(np.hstack(y_cls), "y_cls")

        # additional statistics to maybe better gauge solution performance
        mae = metrics.mean_absolute_error(y_test_true, y_test)
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        mlflow.log_metric("elitist.size", self.learner_.search_.size_[0],
                          self.n_iter)
        mlflow.log_metric("elitist.p_M_D", self.learner_.search_.p_M_D_[0],
                          self.n_iter)
        mlflow.log_metric("elitist.mae", mae, self.n_iter)
        mlflow.log_metric("elitist.mse", mse, self.n_iter)
        mlflow.log_metric("elitist.r2-score", r2, self.n_iter)

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
        add_title(ax, self.learner_.search_.size_[0],
                  self.learner_.search_.p_M_D_[0], mse, r2)
        save_plot("pred_incl_cls", fig)

        if self.show:
            plt.show()

        if not self.learner_.toolbox.literal:
            try:
                X_points = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5,
                                     1])[:, np.newaxis]
                log_array(X_points, "X_points")

                y_points_mean, y_points_var = self.estimator.predict_mean_var(
                    X_points)
                y_points_std = np.sqrt(y_points_var)
                log_array(y_points_mean, "y_points_mean")
                log_array(y_points_std, "y_points_std")

                pdf = self.estimator.predict_distribution(X_points)

                for i in range(len(X_points)):
                    # TODO This arange sometimes gets a nan and then fails.
                    y_points = np.arange(
                        y_points_mean[i] - 2 * y_points_std[i],
                        y_points_mean[i] + 2 * y_points_std[i], 0.01)
                    prob_y_points = pdf(y_points)[:, i]
                    log_array(y_points, f"y_points_{i}")
                    log_array(prob_y_points, f"prob_y_points_{i}")
            except Exception as e:
                print("Evaluation of predictive distribution at designated "
                      "points failed (gracefully).")
                print(e)


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
