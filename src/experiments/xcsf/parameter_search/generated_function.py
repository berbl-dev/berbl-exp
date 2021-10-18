import click  # type: ignore
import mlflow  # type: ignore
from experiments.utils import log_array, log_json
from sklearn.model_selection import ParameterGrid  # type: ignore
from sklearn.model_selection import cross_validate  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from tasks.book.generated_function import generate

from .. import XCSF, get_pop, log_xcs_params

# Mainly based on (Urbanowicz and Browne, 2017; Stalph, Rubinsztajn et al.,
# 2012), unless otherwise noted.
param_grid = ParameterGrid({
    "OMP_NUM_THREADS": [8],  # not relevant for learning performance
    "POP_INIT": [True],  # randomly initialize population
    "POP_SIZE": [30, 50, 100, 200],  # “10 times the expected number of rules”
    "MAX_TRIALS": [int(2e5)],
    "PERF_TRIALS": [1000],  # irrelevant, we evaluate manually
    "LOSS_FUNC": ["mae"],  # irrelevant, we evaluate manually
    "HUBER_DELTA": [1],  # irrelevant since LOSS_FUNC != "huber"
    "E0": [1e-3, 1e-2, 5e-2, 1e-1],  # “if noise, use lower beta and higher e0”
    "ALPHA": [1],  # typical value in literature (stein2019, stalph2012c)
    "NU": [5], # typical value in literature
    "BETA": [0.001, 0.005, 0.01],  # lower value required if high noise
    "DELTA":
    [0.1
     ],  # not sensitive, typical value in literature (stein2019, stalph2012c)
    "THETA_DEL":
    [20
     ],  # not sensitive, typical value in literature (stein2019, stalph2012c)
    "INIT_FITNESS": [0.01],  # e.g. stein2019
    "INIT_ERROR": [0],  # e.g. stein2019
    "M_PROBATION": [int(1e8)],  # quasi disabled
    "STATEFUL": [True],
    "SET_SUBSUMPTION": [
        True
    ],  # without subsumption, the number of rules is fixed to the maximum population size
    "THETA_SUB":
    [20
     ],  # not sensitive, typical value in literature (stein2019, stalph2012c)
    "COMPACTION": [False],  # TODO Maybe enable this in the end?
    "TELETRANSPORTATION": [50],  # irrelevant for supervised learning
    "GAMMA": [0.95],  # irrelevant for supervised learning
    "P_EXPLORE": [0.9],  # irrelevant for supervised learning
    "EA_SELECT_TYPE": ["tournament"],  # tournament is the de-facto standard
    "EA_SELECT_SIZE": [0.4],  # e.g. stein2019
    "THETA_EA":
    [50
     ],  # not sensitive, typical value in literature (stein2019, stalph2012c)
    "LAMBDA": [2],  # de-facto standard
    "P_CROSSOVER": [0.8],  # e.g. stalph2012c
    "ERR_REDUC": [1],  # e.g. stein2019
    "FIT_REDUC": [0.1],  # e.g. stein2019
    "EA_SUBSUMPTION": [False],  # seldomly used, set subsumption should suffice
    "EA_PRED_RESET": [False],
})


@click.command()
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--data-seed", type=click.IntRange(min=0), default=1)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
@click.option("--standardize/--no-standardize", type=bool, default=True)
def run_experiment(seed, data_seed, show, sample_size, standardize):

    for params in param_grid:
        mlflow.set_experiment("xcsf.ps.generated_function")
        with mlflow.start_run() as run:
            mlflow.log_param("xcs.seed", seed)
            mlflow.log_param("data.seed", data_seed)
            mlflow.log_param("data.size", sample_size)

            X, y = generate(sample_size, random_state=data_seed)

            log_array(X, "data.X")
            log_array(y, "data.y")

            if standardize:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

            estimator = XCSF(params=params, random_state=seed)

            k = 5
            scoring = [
                "neg_mean_absolute_error", "r2", "neg_mean_squared_error"
            ]
            scores = cross_validate(estimator,
                                    X,
                                    y.ravel(),
                                    scoring=scoring,
                                    cv=k,
                                    return_estimator=True)

            # Go over each CV result.
            for i in range(k):
                # Log each computed score.
                for score in scoring:
                    score = f"test_{score}"
                    mlflow.log_metric(score, scores[score][i], i)
                # Log final population.
                log_json(get_pop(scores["estimator"][i].xcs_), "population")
                mlflow.log_metric("size",
                                  scores["estimator"][i].xcs_.pset_size())


if __name__ == "__main__":
    run_experiment()
