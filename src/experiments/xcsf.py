import io
import tempfile

import experiments.utils as utils  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import xcsf.xcsf as xcsf  # type: ignore
from berbl.utils import randseed
from sklearn import metrics  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import check_is_fitted, check_X_y

from .utils import log_array, plot_prediction, save_plot
from . import Experiment


class XCSFExperiment(Experiment):
    algorithm = "xcsf"

    def init_estimator(self):
        self.estimator = XCSF(self.params, random_state=self.seed)

    def evaluate(self, X, y, X_test, y_test_true, X_denoised, y_denoised):
        # make predictions for test data
        y_test = self.estimator.predict(X_test)
        log_array(y_test, "y_test")

        # TODO get unmixed classifier predictions

        # get metrics
        mae = metrics.mean_absolute_error(y_test_true, y_test)
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        mlflow.log_metric("mae", mae, self.params["MAX_TRIALS"])
        mlflow.log_metric("mse", mse, self.params["MAX_TRIALS"])
        mlflow.log_metric("r2-score", r2, self.params["MAX_TRIALS"])

        mlflow.log_metric("size", self.learner_.xcs_.pset_size())

        pop = self.learner_.population()
        utils.log_json(pop, "population")

        fig, ax = plot_prediction(X=X,
                                  y=y,
                                  X_test=X_test,
                                  y_test=y_test,
                                  X_denoised=X_denoised,
                                  y_denoised=y_denoised)
        save_plot("pred", fig)

        if self.show:
            plt.show()


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
    mlflow.log_param("xcs.EA_SELECT_TYPE", xcs.EA_SELECT_TYPE)
    mlflow.log_param("xcs.EA_SELECT_SIZE", xcs.EA_SELECT_SIZE)
    mlflow.log_param("xcs.THETA_EA", xcs.THETA_EA)
    mlflow.log_param("xcs.LAMBDA", xcs.LAMBDA)
    mlflow.log_param("xcs.P_CROSSOVER", xcs.P_CROSSOVER)
    mlflow.log_param("xcs.ERR_REDUC", xcs.ERR_REDUC)
    mlflow.log_param("xcs.FIT_REDUC", xcs.FIT_REDUC)
    mlflow.log_param("xcs.EA_SUBSUMPTION", xcs.EA_SUBSUMPTION)
    mlflow.log_param("xcs.EA_PRED_RESET", xcs.EA_PRED_RESET)


def set_xcs_params(xcs, params):
    xcs.OMP_NUM_THREADS = params["OMP_NUM_THREADS"]
    xcs.POP_INIT = params["POP_INIT"]
    xcs.POP_SIZE = params["POP_SIZE"]
    xcs.MAX_TRIALS = params["MAX_TRIALS"]
    xcs.PERF_TRIALS = params["PERF_TRIALS"]
    xcs.LOSS_FUNC = params["LOSS_FUNC"]
    xcs.HUBER_DELTA = params["HUBER_DELTA"]
    xcs.E0 = params["E0"]
    xcs.ALPHA = params["ALPHA"]
    xcs.NU = params["NU"]
    xcs.BETA = params["BETA"]
    xcs.DELTA = params["DELTA"]
    xcs.THETA_DEL = params["THETA_DEL"]
    xcs.INIT_FITNESS = params["INIT_FITNESS"]
    xcs.INIT_ERROR = params["INIT_ERROR"]
    xcs.M_PROBATION = params["M_PROBATION"]
    xcs.STATEFUL = params["STATEFUL"]
    xcs.SET_SUBSUMPTION = params["SET_SUBSUMPTION"]
    xcs.THETA_SUB = params["THETA_SUB"]
    xcs.COMPACTION = params["COMPACTION"]
    xcs.TELETRANSPORTATION = params["TELETRANSPORTATION"]
    xcs.GAMMA = params["GAMMA"]
    xcs.P_EXPLORE = params["P_EXPLORE"]
    xcs.EA_SELECT_TYPE = params["EA_SELECT_TYPE"]
    xcs.EA_SELECT_SIZE = params["EA_SELECT_SIZE"]
    xcs.THETA_EA = params["THETA_EA"]
    xcs.LAMBDA = params["LAMBDA"]
    xcs.P_CROSSOVER = params["P_CROSSOVER"]
    xcs.ERR_REDUC = params["ERR_REDUC"]
    xcs.FIT_REDUC = params["FIT_REDUC"]
    xcs.EA_SUBSUMPTION = params["EA_SUBSUMPTION"]
    xcs.EA_PRED_RESET = params["EA_PRED_RESET"]


def get_xcs_params(xcs):
    return {
        "OMP_NUM_THREADS": xcs.OMP_NUM_THREADS,
        "POP_INIT": xcs.POP_INIT,
        "POP_SIZE": xcs.POP_SIZE,
        "MAX_TRIALS": xcs.MAX_TRIALS,
        "PERF_TRIALS": xcs.PERF_TRIALS,
        "LOSS_FUNC": xcs.LOSS_FUNC,
        "HUBER_DELTA": xcs.HUBER_DELTA,
        "E0": xcs.E0,
        "ALPHA": xcs.ALPHA,
        "NU": xcs.NU,
        "BETA": xcs.BETA,
        "DELTA": xcs.DELTA,
        "THETA_DEL": xcs.THETA_DEL,
        "INIT_FITNESS": xcs.INIT_FITNESS,
        "INIT_ERROR": xcs.INIT_ERROR,
        "M_PROBATION": xcs.M_PROBATION,
        "STATEFUL": xcs.STATEFUL,
        "SET_SUBSUMPTION": xcs.SET_SUBSUMPTION,
        "THETA_SUB": xcs.THETA_SUB,
        "COMPACTION": xcs.COMPACTION,
        "TELETRANSPORTATION": xcs.TELETRANSPORTATION,
        "GAMMA": xcs.GAMMA,
        "P_EXPLORE": xcs.P_EXPLORE,
        "EA_SELECT_TYPE": xcs.EA_SELECT_TYPE,
        "EA_SELECT_SIZE": xcs.EA_SELECT_SIZE,
        "THETA_EA": xcs.THETA_EA,
        "LAMBDA": xcs.LAMBDA,
        "P_CROSSOVER": xcs.P_CROSSOVER,
        "ERR_REDUC": xcs.ERR_REDUC,
        "FIT_REDUC": xcs.FIT_REDUC,
        "EA_SUBSUMPTION": xcs.EA_SUBSUMPTION,
        "EA_PRED_RESET": xcs.EA_PRED_RESET,
    }


def default_xcs_params():
    xcs = xcsf.XCS(1, 1, 1)
    return get_xcs_params(xcs)


class XCSF(BaseEstimator, RegressorMixin):
    """
    Almost a correct sklearn wrapper for ``xcsf.XCS``. For example, it can't yet
    be pickled.
    """
    def __init__(self, params, random_state):
        self.params = params
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # This is required so that XCS does not (silently!?) segfault (see
        # https://github.com/rpreen/xcsf/issues/17 ).
        y = y.reshape((len(X), -1))

        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(X.shape[1], 1, 1)  # only 1 (dummy) action
        xcs.seed(randseed(random_state))

        if self.params is None:
            set_xcs_params(xcs, def_params)
        else:
            set_xcs_params(xcs, self.params)
        log_xcs_params(xcs)

        xcs.action("integer")  # (dummy) integer actions

        args = {
            "min": -1,  # minimum value of a lower bound
            "max": 1,  # maximum value of an upper bound
            "spread_min": 0.1,  # minimum initial spread
            "eta":
            0,  # disable gradient descent of centers towards matched input mean
        }
        condition_string = "ub_hyperrectangle"
        xcs.condition(condition_string, args)
        mlflow.log_param("xcs.condition", condition_string)
        mlflow.log_param("xcs.condition.args.min", args["min"])
        mlflow.log_param("xcs.condition.args.max", args["max"])
        mlflow.log_param("xcs.condition.args.spread_min", args["spread_min"])
        mlflow.log_param("xcs.condition.args.eta", args["eta"])

        args = {
            "x0": 1,  # bias attribute
            "rls_scale_factor":
            1000,  # initial diagonal values of the gain-matrix
            "rls_lambda": 1,  # forget rate (small values may be unstable)
        }
        prediction_string = "rls_linear"
        xcs.prediction(prediction_string, args)
        mlflow.log_param("xcs.prediction", prediction_string)
        mlflow.log_param("xcs.prediction.args.x0", args["x0"])
        mlflow.log_param("xcs.prediction.args.rls_scale_factor",
                         args["rls_scale_factor"])
        mlflow.log_param("xcs.prediction.args.rls_lambda", args["rls_lambda"])

        xcs.fit(X, y, True)

        self.xcs_ = xcs

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return self.xcs_.predict(X)

    def population(self):
        check_is_fitted(self)

        out = io.BytesIO()
        with utils.stdout_redirector(out):
            self.xcs_.print_pset(True, True, True)

        pop = out.getvalue().decode("utf-8")
        return pop
