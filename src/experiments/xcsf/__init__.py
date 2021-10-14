import mlflow
import pyparsing as pp


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
