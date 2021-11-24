import os
import re
from datetime import datetime, timedelta, timezone

import baycomp
import matplotlib.pyplot as plt
import mlflow.tracking
import numpy as np
import pandas as pd


def to_date(number):
    """
    Converts mlflow times to ``datetime``s.
    """
    return datetime.fromtimestamp(number // 1000,
                                  tz=timezone(offset=timedelta(hours=2)))


def date_to_string(date):
    """
    Uniform string representation for dates.
    """
    return date.strftime("%Y-%m-%d %H:%M:%S")


def duration(run):
    if not run.info.end_time:
        return datetime.now(tz=timezone(offset=timedelta(hours=2))) - to_date(
            run.info.start_time)
    else:
        return to_date(run.info.end_time) - to_date(run.info.start_time)


def exp_id(exp_name):
    try:
        e_id = [
            exp.experiment_id for exp in mlflow.list_experiments()
            if exp.name == exp_name
        ][0]
    except Exception as e:
        print(
            f"Experiment with name {exp_name} does not exist in store {mlflow.get_tracking_uri}."
        )
        raise e
    return e_id


def runs(exp_name):
    expid = exp_id(exp_name)
    return mlflow.search_runs(expid)


def recent_runs(exp_name, n=1):
    """
    Recent but FINISHED runs.

    Parameters
    ----------
    n : int
        Number of runs to return.
    """
    rs = [r for r in runs(exp_name) if r.info.status == "FINISHED"]
    return list(sorted(rs, key=time, reverse=True))[:n]


def git_runs(exp_name, commit, unfinished=False):
    """
    Runs at the given commit.
    """
    expid = exp_id(exp_name)
    return [
        r for r in mlflow.search_runs(
            expid,
            f"tags.mlflow.source.git.commit = '{commit}'",
            max_results=10000)
        # , output_format="pandas" if pandas else "list") # Only in mlflow 1.20.2
        if r.info.status == "FINISHED" or unfinished
    ]


def get_data(artifact_uri):
    files = os.listdir(artifact_uri)
    contents = {re.sub(".csv", "", f): f for f in files if f.endswith(".csv")}
    data = {
        key: pd.read_csv(f"{artifact_uri}/{contents[key]}", index_col=0)
        for key in contents
    }
    return data


def check_input_data(run_datas):
    for data in run_datas:
        for key in [
                "X", "y", "X_denoised", "y_denoised", "X_test", "y_test_true"
        ]:
            assert all(data[key] == run_datas[0][key])


def plot_training_data(ax, artifact_uri):
    """
    Plot training data (and denoised data for visual reference).
    """
    data = get_data(artifact_uri)
    ax.plot(data["X"], data["y"], "k+")
    ax.plot(data["X_denoised"], data["y_denoised"], "k--")


def plot_prediction(ax, artifact_uri):
    data = get_data(artifact_uri)

    # sort and get permutation of prediction data points
    X_test = data["X_test"].to_numpy().ravel()
    perm = np.argsort(X_test)
    X_test = X_test[perm]
    y_test = data["y_test"].to_numpy().ravel()[perm]

    # plot prediction means
    ax.plot(X_test, y_test, "C0")

    # plot prediction stds, if var exists in data
    try:
        var = data["var"].to_numpy().ravel()[perm]
        std = np.sqrt(var)
        ax.fill_between(X_test,
                        y_test - std,
                        y_test + std,
                        color="C0",
                        alpha=0.3)
        ax.plot(X_test, y_test - std, c="C0", linestyle="dotted")
        ax.plot(X_test, y_test + std, c="C0", linestyle="dotted")
    except KeyError:
        pass

def save_plot(exp_name, plot_name, fig):
    fig_folder = f"eval/plots/{exp_name}"
    os.makedirs(fig_folder, exist_ok=True)
    fig_file = f"{fig_folder}/{plot_name}.pdf"
    print(f"Storing plot in {fig_file}")
    fig.savefig(fig_file)

def metrics_histories(run):
    client = mlflow.tracking.MlflowClient(
        tracking_uri=mlflow.get_tracking_uri())
    metrics = run.data.metrics.keys()
    return pd.DataFrame({
        metric: [
            entry.value
            for entry in client.get_metric_history(run.info.run_id, metric)
        ]
        for metric in metrics
    })


def top_mean(n, metric):
    return list(
        sorted(runs,
               key=lambda r: metrics_histories(r)[metric].mean(),
               reverse=True))[:n]


def flatten(l):
    return [val for sublist in l for val in sublist]


# Let's build DataFrames for easier manipulation.
def df_row(run):
    metrics = pd.Series(run.data.metrics)
    params = pd.Series(run.data.params)
    metadata = pd.Series({
        "git": run.data.tags["mlflow.source.git.commit"],
        "id": run.info.run_id,
        "name": run.data.tags["mlflow.runName"]
    })
    return pd.concat([metrics, params, metadata])


def df(runs):
    return pd.concat([df_row(run) for run in runs], axis=1).T


def apply(f, dct):
    return {key: f(dct[key]) for key in dct}


def task_name(exp_name):
    return re.sub(".*\..*\.", "", exp_name)


def strs_to_nums(tup):
    return tuple([float(x) if float(x) < 1 else int(x) for x in tup])


def stat_test(runs1, runs2, rope, **kwargs):
    """
    Parameters
    ----------
    runs1 : list of float
        For each of the data sets, the mean of the considered metric
        calculated on the runs of the first algorithm.
    runs2 : list of float
        For each of the data sets, the mean of the considered metric
        calculated on the runs of the second algorithm.
    """
    return baycomp.two_on_multiple(x=runs1, y=runs2, rope=rope, **kwargs)


# run = berbl_experiments[exp][0]
# for standardize in
#     rs = [
#         r for r in berbl_experiments[exp]
#         if r.data.params["standardize"] == standardize
#     ]

# Performance analysis: Compare standardization with Drugowitsch's results (p(M | D)).

# Performance analysis: MAE of the mean.

# Comparison to XCSF: MAE.

# Plot distribution exemplarily for a certain y, compare to XCSF where only the
# matching classifiers' bookkeeping parameters are available.

# runs = git_runs("xcsf.generated_function",
#                 "ca721318e4e25ccd95f3d62af69dbf2ff022cc3c")

# run_datas = [get_data(run) for run in runs]

# plot training data (and denoised data for visual reference)
# plt.plot(run_datas[0]["X"], run_datas[0]["y"], "+")
# plt.plot(run_datas[0]["X_denoised"], run_datas[0]["y_denoised"], "k--")

# check_input_data(run_datas)
# plot_training_data(runs[0])
# for run in runs:
#     plot_prediction(run)

# plt.show()

# may be useful
# get_metric_history(run_id, key)
