import os
from itertools import combinations
import pickle

import baycomp
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
from experiments.xcsf.parameter_search.generated_function import param_dict

from . import *

# These are the only parameters that we change in the parameter search.
changed_params = [f"xcs.{k}" for k in param_dict if len(param_dict[k]) > 1]


def metrics(run):
    metrics = run.data.metrics.keys()
    return pd.DataFrame({
        metric: [
            entry.value
            for entry in cl.get_metric_history(run.info.run_id, metric)
        ]
        for metric in metrics
    })


def top_mean(n, metric):
    return list(
        sorted(runs, key=lambda r: metrics(r)[metric].mean(),
               reverse=True))[:n]


def diff_params(run1, run2):
    params1 = run1.data.params
    params2 = run2.data.params
    diff_keys = [item[0] for item in params1.items() ^ params2.items()]
    return ({key: params1[key]
             for key in diff_keys}, {key: params2[key]
                                     for key in diff_keys})


def run_summary(run):
    return (
        run.data.params["data.seed"],
        metrics(run)["test_neg_mean_absolute_error"].mean(),  #
        metrics(run)["test_neg_mean_absolute_error"].var(),
        run.data.params["xcs.MAX_TRIALS"],
        run.info.status,
        duration(run))


def df_row(run):
    metadata = pd.Series({
        "id": run.info.run_id,
        "xcs.seed": run.data.params["xcs.seed"],
        "data.seed": run.data.params["data.seed"],
    } | {k: run.data.params[k]
         for k in changed_params})
    # Store mean, var but also each individual cv value.
    ms = metrics(run)
    mean = ms.mean().rename({k: f"{k}.mean" for k in ms.keys()})
    var = ms.var().rename({k: f"{k}.var" for k in ms.keys()})
    ms = ms.T.aggregate(lambda x: np.array(list(x)), axis=1)
    return pd.concat([metadata, pd.concat([mean, var]), ms])


def df(runs):
    return pd.concat([df_row(run) for run in runs], axis=1).T


k = 5
dir = f"eval/xcsf-ps-gen-func-{k}-cv"

# Loading runs from mlflow takes too long to do it every time we reload this
# script.
fil = f"{dir}/runs.pickle"
if os.path.exists(fil):
    print("Getting runs from pickle")
    with open(fil, "rb") as f:
        runs = pickle.load(f)
else:
    print("Fetching runs from mlflow")
    runs = git_runs(
        "xcsf.ps.generated_function",
        # "c5edfee6f1a549d0f806c36ce915b7fbe721bc49", # 10-fold cv
        "e6e954e5a58a2295c0b57c70b8e44457c52fd6a6",  # 5-fold cv
        unfinished=False)
    print("Caching runs in pickle")
    with open(fil, "wb") as f:
        pickle.dump(runs, f)

fildf = f"{dir}/runsdf.pickle"
if os.path.exists(fildf):
    print("Getting runs DataFrame from pickle")
    runsdf = pd.read_pickle(fildf)
else:
    print("Building runs DataFrame from scratch")
    runsdf = df(runs)
    print("Caching runs DataFrame in pickle")
    runsdf.to_pickle(fildf)

groups = runsdf.groupby(changed_params)
for g in groups.groups:
    # We have 5 * 5 data points per parametrization.
    assert len(groups.groups[g]) == 25, groups.groups[g]

metric_names = metrics(runs[0]).keys()


def checklen(x):
    assert len(x) == k, f"not length {k}: {x}"


for metric_name in metric_names:
    runsdf[metric_name].apply(checklen)

probabilities = []
# For every combination of parameters
for pi, pj in combinations(groups.groups.keys(), 2):
    # parametrization
    # pi = list(groups.groups.keys())[0]
    # pj = list(groups.groups.keys())[1]
    # pj = ("50", "0.001", "0.001")
    # all data for the parametrization
    datai = groups.get_group(pi)
    dataj = groups.get_group(pj)
    # data sets
    di = datai.groupby("data.seed")
    dj = dataj.groupby("data.seed")

    # we look at mean absolute error
    metric = "test_neg_mean_absolute_error"

    scoresi = datai.groupby("data.seed")[metric].aggregate(
        lambda x: np.concatenate(list(x)))
    scoresj = dataj.groupby("data.seed")[metric].aggregate(
        lambda x: np.concatenate(list(x)))
    scoresi = np.stack(scoresi.to_numpy())
    scoresj = np.stack(scoresj.to_numpy())
    probs, fig = baycomp.two_on_multiple(
        # probs = baycomp.two_on_multiple(
        scoresi,
        scoresj,
        # We have 5 runs of k-fold cv.
        runs=5,
        rope=0.01,
        plot=True,
        names=[str(pi), str(pj)])

    print(pi, pj, "=>", probs)
    probabilities.append((pi, pj, probs))
    print(probabilities)
    fig.savefig(
        f"{dir}/{pi[0]}-{pi[1]}-{pi[2]}-vs-{pj[0]}-{pj[1]}-{pj[2]}.pdf")
    # plt.show()

pd.DataFrame(probabilities).to_pickle(f"{dir}/probabilitiesdf.pickle")
