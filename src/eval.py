import os
from itertools import combinations

import baycomp
import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import run as runpy
from evaluation import *
from experiments.xcsf.parameter_search import param_dict, param_grid

pd.options.display.max_rows = 2000


@click.command()
@click.argument("PATH")
@click.option("--graphs/--no-graphs",
              default=False,
              help="Whether to show graphs for the better MAE lattices",
              show_default=True)
def main(path, graphs):
    """
    Analyse the parameter search results found at PATH.
    """
    mlflow.set_tracking_uri(path)

    berbl_experiment_names = [
        f"berbl.{exp_name}" for exp_name in runpy.berbl_experiments
    ]
    xcsf_experiment_names = [
        f"xcsf.{exp_name}" for exp_name in runpy.xcsf_experiments
    ]

    # Only consider runs that ran from this commit.
    commit = "769097c49033e89433ac8fbabe7e15472acc844f"

    print()
    print("# Comparison with Drugowitsch's results (p(M | D) and size)")
    print()

    # Get all run data gathered for berbl and perform a few checks.
    print("Loading runs from mlflow and checking data, may take a few seconds …")
    berbl_experiments = {}
    for exp_name in berbl_experiment_names:
        expid = exp_id(exp_name)
        rs = mlflow.search_runs(
            expid, (f"tags.mlflow.source.git.commit = '{commit}'"),
            max_results=10000,
            output_format="pandas")

        # Factor 2 due to all experiments having been made once with
        # standardized data and once without.
        n_runs = 5 * 5 * 2
        assert len(rs) == n_runs, (
            f"There should be {n_runs} runs for {exp_name} (standardize = "
            f" False) but there are {len(rs)}.")
        # Check whether all runs being considered are finished
        assert (rs.status != "FINISHED"
                ).sum() == 0, f"Some runs for {exp_name} are not FINISHED yet."

        berbl_experiments |= {exp_name: rs}

    # Performance analysis: Compare with Drugowitsch's results (p(M | D)).

    # Do not concern ourselves yet with the standardized experiments but only with
    # the 1:1 reproduction of Drugowitsch's results.
    repr_berbl_experiments = {
        exp_name: berbl_experiments[exp_name][
            berbl_experiments[exp_name]["params.standardize"] == "False"]
        for exp_name in berbl_experiments
    }
    # Introduce a shorthand for the REPL.
    rpe = repr_berbl_experiments

    # These are Drugowitsch's results on these tasks (taken from his book).
    drugowitsch_ga = pd.DataFrame({
        "generated_function": {
            "metrics.elitist.p_M_D": 118.81,
            "metrics.elitist.size": 2
        },
        "sparse_noisy_data": {
            "metrics.elitist.p_M_D": -159.07,
            "metrics.elitist.size": 2
        },
        "variable_noise": {
            "metrics.elitist.p_M_D": -63.12,
            "metrics.elitist.size": 2
        },
        "sine": {
            "metrics.elitist.p_M_D": -155.68,
            "metrics.elitist.size": 7
        },
    })
    drugowitsch_mcmc = pd.DataFrame({
        "generated_function": {
            "metrics.elitist.p_M_D": 174.50,
            "metrics.elitist.size": 3
        },
        "sparse_noisy_data": {
            "metrics.elitist.p_M_D": -158.55,
            "metrics.elitist.size": 2
        },
        "variable_noise": {
            "metrics.elitist.p_M_D": -58.59,
            "metrics.elitist.size": 2
        },
        "sine": {
            "metrics.elitist.p_M_D": -29.39,
            "metrics.elitist.size": 5
        },
    })

    # for each *task* add one table that puts together book vs modular vs drugowitsch vs xcsf
    metrics = [f"metrics.elitist.{m}" for m in ["size", "mae", "p_M_D"]]
    # for exp_name, df in sorted(repr_berbl_experiments.items(),
    #                            key=lambda it: task_name(it[0])):
    for exp_name, rs in repr_berbl_experiments.items():

        tname = task_name(exp_name)
        print()
        print(f"## {exp_name}")
        print()

        d = pd.DataFrame({
            "mean": rs[metrics].mean(),
            "std": rs[metrics].std(),
            "max": rs[metrics].max(),
            "min": rs[metrics].min()
        })
        d = pd.concat([d, drugowitsch_ga[tname]], axis=1)
        d = d.rename(columns={tname: "Drugowitsch's GA"})
        d = pd.concat([d, drugowitsch_mcmc[tname]], axis=1)
        d = d.rename(columns={tname: "Drugowitsch's MCMC"})
        d = d.T

        d2 = rs.groupby("params.data.seed")[metrics].mean()
        d2.index = [f"data seed {i}: mean" for i in d2.index]
        d = pd.concat([d, d2])

        d2 = rs.groupby("params.data.seed")[metrics].std()
        d2.index = [f"data seed {i}: std" for i in d2.index]
        d = pd.concat([d, d2])

        print(d.to_markdown())

    # fig, ax = plot_prediction(X=X,
    #                             y=y,
    #                             X_test=X_test,
    #                             y_test=y_test,
    #                             var=var,
    #                             X_denoised=X_denoised,
    #                             y_denoised=y_denoised)

    #         for i in range(len(X_points)):
    #             fig, ax = plt.subplots()
    #             y = np.arange(mean[i] - 2 * std[i], mean[i] + 2 * std[i], 0.01)
    #             std1 = mean[i] - std[i]
    #             std2 = mean[i] + std[i]

    #             plt.plot(y, pdf(y)[:, i], "b--")
    #             plt.axvline(mean[i], color="b")
    #             ax.fill_between(y,
    #                             pdf(y)[:, i],
    #                             alpha=0.3,
    #                             where=np.logical_and(std1 < y, y < std2))
    #             plt.vlines([std1, std2],
    #                        ymin=0,
    #                        ymax=[pdf(std1)[:, i], pdf(std2)[:, i]],
    #                        linestyle="dotted",
    #                        color="b")


if __name__ == "__main__":
    main()
