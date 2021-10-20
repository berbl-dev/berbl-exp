import os
import re
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import mlflow.tracking
import numpy as np
import pandas as pd

node = "oc-compute03"

cl = mlflow.tracking.MlflowClient(
    tracking_uri=f"mlruns-remote-{node}/localized/mlruns",
    registry_uri=f"mlruns-remote-{node}/localized/mlruns")


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
    return [
        exp.experiment_id for exp in cl.list_experiments()
        if exp.name == exp_name
    ][0]


def runs(exp_name):
    expid = exp_id(exp_name)
    return cl.search_runs(expid)


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
        r
        for r in cl.search_runs(expid,
                                f"tags.mlflow.source.git.commit = '{commit}'",
                                max_results=10000)
        # , output_format="pandas" if pandas else "list") # Only in mlflow 1.20.2
        if r.info.status == "FINISHED" or unfinished
    ]


def get_data(run):
    folder = run.info.artifact_uri.replace("file://", "")
    files = os.listdir(folder)
    contents = {re.sub(".csv", "", f): f for f in files if f.endswith(".csv")}
    data = {
        key: pd.read_csv(f"{folder}/{contents[key]}", index_col=0)
        for key in contents
    }
    return data


def check_input_data(run_datas):
    for data in run_datas:
        for key in [
                "X", "y", "X_denoised", "y_denoised", "X_test", "y_test_true"
        ]:
            assert all(data[key] == run_datas[0][key])


def plot_training_data(run):
    """
    Plot training data (and denoised data for visual reference).
    """
    data = get_data(run)
    plt.plot(data["X"], data["y"], "+")
    plt.plot(data["X_denoised"], data["y_denoised"], "k--")


def plot_prediction(run):
    data = get_data(run)

    # sort and get permutation of prediction data points
    X_test_ = data["X_test"].to_numpy().ravel()
    perm = np.argsort(X_test_)
    X_test_ = X_test_[perm]

    # plot predictions
    for data in run_datas:
        plt.plot(X_test_, data["y_test"].to_numpy().ravel()[perm])


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
