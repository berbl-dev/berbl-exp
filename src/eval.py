import warnings

import click
import matplotlib.pyplot as plt
import numpy as np
import run as runpy
from evaluation import *
from evaluation.plot import *

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.options.display.max_rows = 2000

# Metric and whether higher is better.
metrics = {"p_M_D": True, "mae": False, "size": False}
ropes = {"p_M_D": 10, "mae": 0.01, "size": 0.5}


def table_compare_drugowitsch(runs):
    print()
    print("# Comparison with Drugowitsch's results (p(M | D) and size)")
    print()

    # These are Drugowitsch's results on these tasks (taken from his book).
    drugowitsch_ga = pd.DataFrame({
        "generated_function": {
            "$p(\MM \mid \DD)$": 118.81,
            "$K$": 2
        },
        "sparse_noisy_data": {
            "$p(\MM \mid \DD)$": -159.07,
            "$K$": 2
        },
        "variable_noise": {
            "$p(\MM \mid \DD)$": -63.12,
            "$K$": 2
        },
        "sine": {
            "$p(\MM \mid \DD)$": -155.68,
            "$K$": 7
        },
    })
    drugowitsch_mcmc = pd.DataFrame({
        "generated_function": {
            "$p(\MM \mid \DD)$": 174.50,
            "$K$": 3
        },
        "sparse_noisy_data": {
            "$p(\MM \mid \DD)$": -158.55,
            "$K$": 2
        },
        "variable_noise": {
            "$p(\MM \mid \DD)$": -58.59,
            "$K$": 2
        },
        "sine": {
            "$p(\MM \mid \DD)$": -29.39,
            "$K$": 5
        },
    })

    runs = keep_unstandardized(runs)

    print("Keeping only reproduction BERBL experiments …")
    # First level of index is algorithm.
    rs = runs.loc["berbl"]
    # Second level of index is variant.
    rs = rs.loc[["book", "non_literal"]]

    assert len(rs) == 10 * 5 * (4 + 4)

    rs = rs.rename(lambda s: s.removeprefix("metrics.elitist."), axis=1)

    metrics = ["p_M_D", "size"]
    groups = rs.groupby(level=["variant", "task"])[metrics]

    means = groups.mean()
    means = pd.DataFrame(means.stack()).rename(columns={0: "mean"})
    medians = groups.median()
    medians = pd.DataFrame(medians.stack()).rename(columns={0: "median"})
    maxs = groups.max()
    maxs = pd.DataFrame(maxs.stack()).rename(columns={0: "max"})

    table = means.join([medians, maxs])
    table = table.unstack(0)
    table.columns = table.columns.swaplevel(0, 1)
    table = table[table.columns[[0, 2, 4, 1, 3, 5]]]

    table.index = table.index.rename(["task", "metric"])
    table = table.reset_index()
    table["metric"] = table["metric"].apply(
        lambda s: "$K$" if s == "size" else "$p(\MM \mid \DD)$")
    table.index = pd.MultiIndex.from_arrays([table["task"], table["metric"]])
    del table["task"]
    del table["metric"]
    table = table.sort_values("metric")
    table = table.reindex(
        ["generated_function", "sparse_noisy_data", "variable_noise", "sine"],
        level=0)

    table = table.round(2)

    dga = drugowitsch_ga.stack()
    dga.index = dga.index.swaplevel(0, 1)
    dga = dga.rename("Drugowitsch")
    dga.index = dga.index.rename(["task", "metric"])

    table = table.join(dga)

    print(table)
    print()

    print(table.to_latex(escape=False))
    print()


def median_run(runs, metric, algorithm, variant, task):
    rs = runs.loc[(algorithm, variant, task)]

    assert len(rs) == 10 * 5, len(rs)
    r = rs[rs[metric] == rs[metric].quantile(interpolation="higher")].iloc[0]
    return r


def plot_median_predictions(runs, path, graphs):
    print()
    print(
        "## Plotting median p(M | D) (or MAE) run for all algorithms and each "
        "task to compare plots")
    print()

    exp_names = runs.unstack().index

    prediction_plots = {}
    for exp_name in exp_names:
        algorithm, variant, task = exp_name
        if algorithm == "xcsf":
            metric = "metrics.mae"
            rs = runs
        else:
            metric = "metrics.elitist.p_M_D"
            rs = keep_unstandardized(runs)

        r = median_run(rs, metric, algorithm, variant, task)

        print(f"Plotting median run for {exp_name} …")
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
        else:
            plt.close("all")


def stat_tests_lit_mod(runs):
    print()
    print("# Performing statistical tests for literal vs. modular")
    print()

    runs = keep_unstandardized(runs)

    print("Keeping only reproduction BERBL experiments …")
    # First level of index is algorithm.
    rs = runs.loc["berbl"]
    # Second level of index is variant.
    rs = rs.loc[["book", "non_literal"]]

    assert len(rs) == 10 * 5 * (4 + 4)

    rs = rs.rename(lambda s: s.removeprefix("metrics.elitist."), axis=1)

    means = rs.groupby(["variant", "task"]).mean()[metrics.keys()]
    for metric, higher_better in metrics.items():
        probs, fig = stat_test(means.loc["book"][metric],
                               means.loc["non_literal"][metric],
                               rope=ropes[metric],
                               plot=True)
        save_plot("stat", f"literal-vs-modular-{metric}", fig)
        print_stat_results("lit", "mod", metric, probs, ropes[metric],
                           higher_better)
        print()


def table_stat_tests_berbl_xcsf(runs):
    print()
    print("# Comparison of BERBL with XCSF (MAE)")
    print()

    metric = "mae"

    print("Selecting interval-based runs …")
    rs_interval = runs[runs["params.match"] == "softint"]
    rs_interval = keep_unstandardized(rs_interval)

    rs_book = runs.loc[("berbl", "book")]
    assert len(rs_interval) == len(rs_book), (
        "There are params.matching==softint runs lacking for some experiments")

    rs_lit = rs_interval[rs_interval["params.literal"] == "True"]
    rs_mod = rs_interval[rs_interval["params.literal"] == "False"]
    assert len(rs_lit) == len(rs_mod), ("Different number of runs for "
                                        "interval-based literal and "
                                        "modular")

    groups_lit = rs_lit.sort_values("task").groupby(
        "task")[f"metrics.elitist.{metric}"]
    groups_mod = rs_mod.sort_values("task").groupby(
        "task")[f"metrics.elitist.{metric}"]
    groups_xcsf = runs.loc[("xcsf", "book")].groupby(["task"
                                                      ])[f"metrics.{metric}"]

    table = pd.DataFrame()
    for grp, f in [
        (groups, f) for f in [np.mean, np.std]
            for groups in [("literal",
                            groups_lit), ("modular",
                                          groups_mod), ("xcsf", groups_xcsf)]
    ]:
        name, groups = grp
        table = table.append(groups.agg(f).rename(f"{name}###{f.__name__}"))

    table.index = table.index.map(lambda x: tuple(x.split("###")))
    table.index = table.index.rename(["algorithm", "statistic"])
    table = table.T
    # TODO Store list of experiments/numbers/order at the toplevel
    table = table.reindex(
        ["generated_function", "sparse_noisy_data", "variable_noise", "sine"])
    table = table.sort_values(by="algorithm", axis=1)

    print(table)
    print()

    table_rounded = table.copy()
    for variant, statistic in table.keys():
        if statistic == "std":
            table_rounded[(variant, statistic)] = table_rounded[(variant, statistic)].round(2)
        else:
            table_rounded[(variant, statistic)] = table_rounded[(variant, statistic)].round(4)

    print(table_rounded.to_latex())
    print()

    print()
    print("## Statistical comparison of BERBL with XCSF (MAE)")
    print()

    for variant in ["literal", "modular"]:
        probs, fig = stat_test(table[("xcsf", "mean")],
                               table[(variant, "mean")],
                               rope=ropes[metric],
                               plot=True)
        save_plot("stat", f"xcsf-vs-{variant}-{metric}", fig)
        print_stat_results(
            "XCSF",
            variant,
            metric,
            probs,
            ropes[metric],
            # Lower MAE are better.
            higher_better=False)
        print()


def plot_extra_xcsf_prediction(runs, path, graphs):
    print()
    print("## Plotting XCSF prediction (median run re MAE) on data seed of "
          "BERBL median run")
    print()

    data_seed = median_run(keep_unstandardized(runs), "metrics.elitist.p_M_D",
                           "berbl", "non_literal",
                           "generated_function")["params.data.seed"]

    exp_name = "xcsf.book.generated_function"

    rs_xcsf = runs.loc[("xcsf", "book", "generated_function")]
    rs_xcsf = rs_xcsf[rs_xcsf["params.data.seed"] == data_seed]
    metric = "metrics.mae"
    # TODO Ugly that we hardcode "mlruns/" here
    r = rs_xcsf[rs_xcsf[metric] == rs_xcsf[metric].quantile(
        interpolation="higher")].iloc[0]
    rid = r["run_id"]
    fixed_art_uri = f"{path}/{r['artifact_uri'].removeprefix('mlruns/')}"
    rdata = get_data(fixed_art_uri)

    fig, ax = plt.subplots()
    plot_training_data(ax, fixed_art_uri)
    plot_prediction(ax, fixed_art_uri)
    ax.set_xlabel("Input x")
    ax.set_ylabel("Output y")
    save_plot(exp_name, "pred-same-data-berbl", fig)

    if graphs:
        plt.show()
    else:
        plt.close("all")


def plot_berbl_pred_dist(runs, path, graphs):
    print()
    print("## Plotting BERBL predictive distribution")
    print()

    exp_name = "berbl.non_literal.generated_function"
    r = median_run(keep_unstandardized(runs), "metrics.elitist.p_M_D", "berbl",
                   "non_literal", "generated_function")

    # index 4 corresponds to p(y | x = 0.25), see
    # experiments.berbl.BERBLExperiment.evaluate.
    place = 0.25
    index = 4
    print(
        f"Plotting point distribution at y={place} of median p(M | D) run for "
        "{exp_name}")
    rid = r["run_id"]
    # TODO Ugly that we hardcode "mlruns/" here
    fixed_art_uri = f"{path}/{r['artifact_uri'].removeprefix('mlruns/')}"
    rdata = get_data(fixed_art_uri)
    mean = rdata["y_points_mean"].loc[index][0]
    std = rdata["y_points_std"].loc[index][0]
    y = rdata[f"y_points_{index}"]
    pdf = rdata[f"prob_y_points_{index}"]

    fig, ax = plt.subplots()
    std1 = mean - std
    std2 = mean + std

    ax.plot(y, pdf, "C0--")
    ax.axvline(mean, color="C0")
    ax.fill_between(x=y.to_numpy().ravel(),
                    y1=pdf.to_numpy().ravel(),
                    color="C0",
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
              color="C0")

    ax.set_xlabel("Output y")
    ax.set_ylabel(f"p(y | x = {place})")
    save_plot(exp_name, f"dist-{place}", fig)

    if graphs:
        plt.show()
    else:
        plt.close("all")


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
    exp_names = berbl_experiment_names + xcsf_experiment_names

    runs = read_mlflow(exp_names, commit=commit)

    n_runs = (
        # BERBL experiments (4 from book, 4 from book with modular backend, 2
        # additional each with interval-based matching).
        (((4 + 4 + 2 + 2)
          # BERBL experiments were performed twice, w/ and w/o standardization.
          * 2)
         # XCSF experiments.
         + 4)
        # 5 data seeds per experiment.
        * 5
        # 10 runs.
        * 10)
    assert len(
        runs) == n_runs, f"Expected {n_runs} runs but there were {len(runs)}"

    table_compare_drugowitsch(runs)

    plot_median_predictions(runs, path, graphs)

    stat_tests_lit_mod(runs)

    table_stat_tests_berbl_xcsf(runs)

    plot_extra_xcsf_prediction(runs, path, graphs)

    plot_berbl_pred_dist(runs, path, graphs)

    # TODO Perform statistical tests on standardized data, too


if __name__ == "__main__":
    main()
