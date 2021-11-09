import io
import tempfile

import experiments.utils as utils  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import pyparsing as pp  # type: ignore
import xcsf.xcsf as xcsf  # type: ignore
from berbl.utils import randseed
from experiments.utils import log_array, plot_prediction, save_plot
from sklearn import metrics  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import check_is_fitted, check_X_y

from .. import Experiment


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

        mlflow.log_metric("size", self.estimator.xcs_.pset_size())

        # store the model, you never know when you need it
        f = tempfile.NamedTemporaryFile(prefix=f"model-", suffix=f".model")
        self.estimator.xcs_.save(f.name)
        mlflow.log_artifact(f.name)
        f.close()

        pop = get_pop(self.estimator.xcs_)
        utils.log_json(pop, "population")

        fig, ax = plot_prediction(X=X,
                                  y=y,
                                  X_test=X_test,
                                  y_test=y_test,
                                  X_denoised=X_denoised,
                                  y_denoised=y_denoised)
        save_plot(fig, self.seed)

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


def get_pop(xcs):
    out = io.BytesIO()
    with utils.stdout_redirector(out):
        xcs.print_pset(True, True, True)

    pop = parse_pop(out.getvalue().decode("utf-8"))
    return pop


class XCSF(BaseEstimator, RegressorMixin):
    """
    Almost a correct sklearn wrapper for ``xcsf.XCS``. For example, it can't yet
    be pickled.
    """
    def __init__(self, params, random_state=None):
        self.params = params
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(X.shape[1], 1, 1)  # only 1 (dummy) action
        xcs.seed(randseed(random_state))

        if self.params is None:
            set_xcs_params(xcs, def_params)
        else:
            set_xcs_params(xcs, self.params)
        log_xcs_params(xcs)

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
        xcs.condition("ub_hyperrectangle", args)
        mlflow.log_param("xcs.condition", "ub_hyperrectangle")
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

        xcs.fit(X, y, True)

        self.xcs_ = xcs

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return self.xcs_.predict(X)
