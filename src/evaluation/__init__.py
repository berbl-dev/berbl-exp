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


def task_name(exp_name):
    return re.sub(".*\..*\.", "", exp_name)


def read_mlflow(exp_names, commit=None):
    print("Loading runs from mlflow and checking data, may take a few "
          "seconds …")
    runs = pd.DataFrame()
    for exp_name in exp_names:
        expid = exp_id(exp_name)
        if commit is None:
            rs = mlflow.search_runs(expid,
                                    max_results=10000,
                                    output_format="pandas")
        else:
            rs = mlflow.search_runs(
                expid, (f"tags.mlflow.source.git.commit = '{commit}'"),
                max_results=10000,
                output_format="pandas")

        # Check whether all runs being considered are finished
        assert (rs.status != "FINISHED"
                ).sum() == 0, f"Some runs for {exp_name} are not FINISHED yet."

        rs["algorithm"], rs["variant"], rs["task"] = exp_name.split(".")
        runs = runs.append(rs)

    runs.index = pd.MultiIndex.from_arrays(
        [runs["algorithm"], runs["variant"], runs["task"], runs.index])
    del runs["algorithm"]
    del runs["variant"]
    del runs["task"]
    return runs

def keep_unstandardized(runs):
    print("Filtering out experiments with standardized data …")
    runs = runs[runs["params.standardize"] == "False"]
    return runs

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


def print_stat_results(name1, name2, metric, probs, rope, higher_better):
    """
    Parameters
    ----------
    name1, name2, metric: str
        Names of the two options being compared as well as of the metric they
        are being compared on.
    probs : triple of floats
        The three probabilities (see baycomp docs or their paper).
    higher_better : bool
        Whether for the considered metric, higher values are better (e.g. for
        errors, generally lower values are better whereas for accuracy a higher
        value is preferable).
    """
    p_worse = probs[2] if higher_better else probs[0]
    p_equal = probs[1]
    p_better = probs[0] if higher_better else probs[2]
    print(f"Regarding metric {metric} for rope={rope}:")
    print(f"p({name1} << {name2}) = {p_worse}")
    print(f"p({name1} ≡ {name2}) = {p_equal}")
    print(f"p({name1} >> {name2}) = {p_better}")
