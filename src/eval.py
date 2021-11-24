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
              help="Whether to show all plots and graphs",
              show_default=True)
@click.option("--commit",
              default=None,
              help="Only consider runs that ran from this commit",
              show_default=True)
def main(path, graphs, commit):
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

    print()
    print("# Comparison with Drugowitsch's results (p(M | D) and size)")
    print()

    # Get all run data gathered for berbl and perform a few checks.
    print(
        "Loading runs from mlflow and checking data, may take a few seconds …")
    berbl_experiments = {}
    for exp_name in berbl_experiment_names:
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

        # Factor 2 due to all experiments having been made once with
        # standardized data and once without.
        n_runs = 10 * 5 * 2
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
    berbl_experiments_unstandardized = {
        exp_name: berbl_experiments[exp_name][
            berbl_experiments[exp_name]["params.standardize"] == "False"]
        for exp_name in berbl_experiments
    }
    # Introduce a shorthand for the REPL.
    rpe = berbl_experiments_unstandardized

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

    metrics = [f"metrics.elitist.{m}" for m in ["size", "p_M_D"]]
    task_names = [
        "generated_function", "sparse_noisy_data", "variable_noise", "sine"
    ]

    print()
    print(
        f"## Compare metric results to Drugowitsch's results on {task_names}")
    print()
    df = pd.DataFrame()
    i = 0
    for tname in task_names:
        i += 1
        exp_name = f"berbl.book.{tname}"
        rs = berbl_experiments_unstandardized[exp_name]
        exp_name_mod = f"berbl.non_literal.{tname}"
        rs_mod = berbl_experiments_unstandardized[exp_name_mod]

        d = pd.DataFrame({
            "task": i,
            "mean": rs[metrics].mean(),
            "median": rs[metrics].median(),
            # "std": rs[metrics].std(),
            "max": rs[metrics].max(),
            "mean ": rs_mod[metrics].mean(),
            "median ": rs_mod[metrics].median(),
            # "std ": rs_mod[metrics].std(),
            "max ": rs_mod[metrics].max(),
        })
        d = pd.concat([d, drugowitsch_ga[tname]], axis=1)
        d = d.rename(columns={tname: "GA"})
        d = pd.concat([d, drugowitsch_mcmc[tname]], axis=1)
        d = d.rename(columns={tname: "MCMC"})
        d = d.round(2)

        df = df.append(d)

        # d2 = rs.groupby("params.data.seed")[metrics].mean()
        # d2.index = [f"data seed {i}: mean" for i in d2.index]
        # d = pd.concat([d, d2])

        # d2 = rs.groupby("params.data.seed")[metrics].std()
        # d2.index = [f"data seed {i}: std" for i in d2.index]
        # d = pd.concat([d, d2])

    df = df.reset_index().rename(columns={"index": "metric"})
    df["metric"] = df["metric"].apply(
        lambda s: "$K$"
        if s == "metrics.elitist.size" else "$p(\MM \mid \DD)$")
    df.index = pd.MultiIndex.from_arrays([df["task"], df["metric"]])
    del df["task"]
    del df["metric"]
    print(df.to_latex(escape=False))

    print("## Plotting median p(M | D) run for each task to compare plots")
    print()

    # Now plot one for each, show that they look the same as Drugowitsch's
    # graphs.
    prediction_plots = {}
    for exp_name, rs in berbl_experiments_unstandardized.items():
        print(f"Plotting median p(M | D) run for {exp_name} …")
        metric = "metrics.elitist.p_M_D"
        r = rs[rs[metric] == rs[metric].quantile(
            interpolation="higher")].iloc[0]
        prediction_plots |= {exp_name: r}
        # TODO Ugly that we hardcode "mlruns/" here
        fixed_art_uri = f"{path}/{r['artifact_uri'].removeprefix('mlruns/')}"
        rdata = get_data(fixed_art_uri)

        fig, ax = plt.subplots()
        plot_training_data(ax, fixed_art_uri)
        plot_prediction(ax, fixed_art_uri)
        ax.set_xlabel("Input x")
        ax.set_ylabel("Output y")
        save_plot(exp_name, "pred", fig)

        if graphs:
            plt.show()

    # For each task, get the mean p_M_D for literal and non_literal
    for metric in ["metrics.elitist.p_M_D", "metrics.elitist.mae"]:
        mshort = metric.removeprefix("metrics.elitist.")
        means_literal = np.array([
            berbl_experiments_unstandardized[f"berbl.book.{tname}"]
            [metric].mean() for tname in task_names
        ])
        means_non_literal = np.array([
            berbl_experiments_unstandardized[f"berbl.non_literal.{tname}"]
            [metric].mean() for tname in task_names
        ])
        probs, fig = stat_test(means_literal,
                               means_non_literal,
                               rope=0.01,
                               plot=True)
        save_plot("stat", f"literal-vs-modular-{mshort}", fig)

        # TODO Not yet sure how meaningful this is
        if False:
            runs_literal = np.array([
                berbl_experiments_unstandardized[f"berbl.book.{tname}"][metric]
                for tname in task_names
            ])
            runs_non_literal = np.array([
                berbl_experiments_unstandardized[f"berbl.non_literal.{tname}"]
                [metric] for tname in task_names
            ])
            import baycomp
            probs, fig = baycomp.two_on_multiple(x=runs_literal,
                                                 y=runs_non_literal,
                                                 rope=0.01,
                                                 plot=True)
            save_plot("stat", f"literal-vs-modular-hierarchical-{mshort}", fig)

        # TODO Not yet sure how meaningful this is
        if False:
            runs_literal = np.array(
                flatten([
                    berbl_experiments_unstandardized[f"berbl.book.{tname}"].
                    groupby("params.data.seed").agg(list)[metric]
                    for tname in task_names
                ]))
            runs_non_literal = np.array(
                flatten([
                    berbl_experiments_unstandardized[
                        f"berbl.non_literal.{tname}"].groupby(
                            "params.data.seed").agg(list)[metric]
                    for tname in task_names
                ]))
            import baycomp
            probs, fig = baycomp.two_on_multiple(x=runs_literal,
                                                 y=runs_non_literal,
                                                 rope=0.01,
                                                 plot=True)
            save_plot("stat",
                      f"literal-vs-modular-hierarchical-flattened-{mshort}",
                      fig)

            if graphs:
                plt.show()

    print()
    print("# Comparison with XCSF (MAE)")
    print()

    # Get all run data gathered for XCSF and perform a few checks.
    print(
        "Loading XCSF runs from mlflow and checking data, may take a few seconds …"
    )
    xcsf_experiments = {}
    for exp_name in xcsf_experiment_names:
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

        n_runs = 10 * 5
        assert len(rs) == n_runs, (
            f"There should be {n_runs} runs for {exp_name} (standardize = "
            f" False) but there are {len(rs)}.")
        # Check whether all runs being considered are finished
        assert (rs.status != "FINISHED"
                ).sum() == 0, f"Some runs for {exp_name} are not FINISHED yet."

        xcsf_experiments |= {exp_name: rs}

    # For each task, get the mean mae for xcsf, literal and non_literal
    means_xcsf = np.array([
        xcsf_experiments[f"xcsf.book.{tname}"]["metrics.mae"].mean()
        for tname in task_names
    ])
    # Use the interval-based experiments for a fair comparison.
    literal_exps = {
        "generated_function":
        berbl_experiments_unstandardized[
            "berbl.additional_literal.generated_function"],
        "sparse_noisy_data":
        berbl_experiments_unstandardized[
            "berbl.additional_literal.sparse_noisy_data"],
        "variable_noise":
        berbl_experiments_unstandardized["berbl.book.variable_noise"],
        "sine":
        berbl_experiments_unstandardized["berbl.book.sine"]
    }
    means_literal = np.array([
        literal_exps[tname]["metrics.elitist.mae"].mean()
        for tname in task_names
    ])
    non_literal_exps = {
        "generated_function":
        berbl_experiments_unstandardized[
            "berbl.additional_non_literal.generated_function"],
        "sparse_noisy_data":
        berbl_experiments_unstandardized[
            "berbl.additional_non_literal.sparse_noisy_data"],
        "variable_noise":
        berbl_experiments_unstandardized["berbl.non_literal.variable_noise"],
        "sine":
        berbl_experiments_unstandardized["berbl.non_literal.sine"]
    }
    means_non_literal = np.array([
        non_literal_exps[tname]["metrics.elitist.mae"].mean()
        for tname in task_names
    ])

    df = pd.DataFrame()
    i = 0
    for tname in task_names:
        i += 1

        # Use the interval-based experiments for a fair comparison.
        if tname in ["generated_function", "sparse_noisy_data"]:
            literal_exp_name = f"berbl.additional_literal.{tname}"
            modular_exp_name = f"berbl.additional_non_literal.{tname}"
        else:
            literal_exp_name = f"berbl.book.{tname}"
            modular_exp_name = f"berbl.non_literal.{tname}"

        rs_xcsf = xcsf_experiments[f"xcsf.book.{tname}"]
        rs_literal = berbl_experiments_unstandardized[literal_exp_name]
        rs_modular = berbl_experiments_unstandardized[modular_exp_name]

        berbl_metric_name = "metrics.elitist.mae"
        xcsf_metric_name = "metrics.mae"
        s = pd.Series(
            {
                "mean": rs_literal[berbl_metric_name].mean(),
                "std": rs_literal[berbl_metric_name].std(),
                "mean ": rs_modular[berbl_metric_name].mean(),
                "std ": rs_modular[berbl_metric_name].std(),
                "mean  ": rs_xcsf[xcsf_metric_name].mean(),
                "std  ": rs_xcsf[xcsf_metric_name].std(),
            },
            name=i)
        df = df.append(s)

    df["mean"] = df["mean"].round(4)
    df["mean "] = df["mean "].round(4)
    df["mean  "] = df["mean  "].round(4)
    df["std"] = df["std"].round(2)
    df["std "] = df["std "].round(2)
    df["std  "] = df["std  "].round(2)
    print(df.to_latex())

    mshort = "mae"
    probs, fig = stat_test(means_xcsf, means_literal, rope=0.01, plot=True)
    print(f"Stat. test result for XCSF vs. literal: {probs}")
    save_plot("stat", f"xcsf-vs-literal-{mshort}", fig)
    probs, fig = stat_test(means_xcsf, means_non_literal, rope=0.01, plot=True)
    print(f"Stat. test result for XCSF vs. modular: {probs}")
    save_plot("stat", f"xcsf-vs-modular-{mshort}", fig)

    # Now plot XCSF graphs (take the median as well).
    for exp_name, rs in xcsf_experiments.items():
        print(f"Plotting median MAE run for {exp_name} …")
        metric = "metrics.mae"
        r = rs[rs[metric] == rs[metric].quantile(
            interpolation="higher")].iloc[0]
        rid = r["run_id"]
        # TODO Ugly that we hardcode "mlruns/" here
        fixed_art_uri = f"{path}/{r['artifact_uri'].removeprefix('mlruns/')}"
        rdata = get_data(fixed_art_uri)

        fig, ax = plt.subplots()
        plot_training_data(ax, fixed_art_uri)
        plot_prediction(ax, fixed_art_uri)
        ax.set_xlabel("Input x")
        ax.set_ylabel("Output y")
        save_plot(exp_name, "pred", fig)

        if graphs:
            plt.show()

    plt.close("all")

    # Now plot BERBL point distributions (median run).
    exp_name = "berbl.non_literal.generated_function"

    r = prediction_plots[exp_name]
    print(
        f"Plotting point distribution of median p(M | D) run for {exp_name} …")
    rid = r["run_id"]
    # TODO Ugly that we hardcode "mlruns/" here
    fixed_art_uri = f"{path}/{r['artifact_uri'].removeprefix('mlruns/')}"
    rdata = get_data(fixed_art_uri)
    # prob_y_points_4 is 0.25, see experiments.berbl.BERBLExperiment.evaluate.
    mean = rdata["y_points_mean"].loc[4][0]
    std = rdata["y_points_std"].loc[4][0]
    y = rdata["y_points_4"]
    pdf = rdata["prob_y_points_4"]

    fig, ax = plt.subplots()
    std1 = mean - std
    std2 = mean + std

    ax.plot(y, pdf, "b--")
    ax.axvline(mean, color="b")
    ax.fill_between(x=y.to_numpy().ravel(),
                    y1=pdf.to_numpy().ravel(),
                    alpha=0.3,
                    where=np.logical_and(std1 < y,
                                         y < std2).to_numpy().ravel())

    # Roughly approximate std indexes (suffices for drawing the figure).
    idx = np.where(np.logical_and(std1 < y, y < std2))[0]
    stdidx1 = idx[0]
    stdidx2 = idx[-1]
    ax.vlines([y.loc[stdidx1], y.loc[stdidx2]],
              ymin=0,
              ymax=[pdf.loc[stdidx1], pdf.loc[stdidx2]],
              linestyle="dotted",
              color="b")

    ax.set_xlabel("Output y")
    ax.set_ylabel("p(y | x = 0.25)")
    save_plot(exp_name, f"dist-{i}", fig)

    if graphs:
        plt.show()


if __name__ == "__main__":
    main()
