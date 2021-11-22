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

    experiment_names = [f"xcsf.{exp}" for exp in runpy.xcsf_experiments]

    # Only consider runs that ran from these commits.
    shas = [
        "29963ee41bfd5d55e855788f801f4e28205ac558"
    ]

    for exp_name in experiment_names:
        print()
        print(f"# Analysis of the results of experiment {exp_name}")
        print()
        commit = shas[0]
        expid = exp_id(exp_name)
        print("Loading runs from mlflow, may take a few seconds …")
        rs = mlflow.search_runs(expid,
                                f"tags.mlflow.source.git.commit = '{commit}'",
                                max_results=10000,
                                output_format="pandas")

        print("Performing checks on data …")
        n_runs = len(param_grid) * 5 * 5
        # Check whether all runs exist (for each of the parameterizations we
        # performed 5 runs for each of 5 data sets).
        assert len(rs) == n_runs, (
            f"There should be {n_runs} runs for {exp_name} "
            f"but there are {len(rs)}. "
            "(Common problem: Did you specify the correct commit in "
            "eval-ps.py?)"
        )
        # Check whether all runs being considered are finished.
        assert (rs.status != "FINISHED"
                ).sum() == 0, f"Some runs for {exp_name} are not FINISHED yet."

        changed_params = [
            f"params.xcs.{k}" for k in param_dict if len(param_dict[k]) > 1
        ]

        # mean = rs.groupby(changed_params + ["params.data.seed"]).agg(np.mean)
        # std = rs.groupby(changed_params).agg(np.std)

        # For each parametrization, get the mean of the considered metric of the
        # runs performed on each data set.
        options = rs.groupby(changed_params
                             + ["params.data.seed"])["metrics.mae"].aggregate(
                                 np.mean)
        # Join the means of the data sets of each parametrization into a single list
        # each. They should have the correct order at this point.
        options = options.groupby(changed_params).aggregate(list)

        n_options = len(options)
        n_combs = np.math.factorial(n_options) / 2 / np.math.factorial(
            n_options - 2)

        rope = 0.01
        print(f"Performing Bayesian hypothesis testing with rope={rope} …")

        print(f"Checking/creating cache directory …")
        cache_dir = f"eval/param-search/{exp_name}"
        os.makedirs(cache_dir, exist_ok=True)
        probabilities_csv = f"{cache_dir}/probabilities.csv"

        try:
            probabilities = pd.read_csv(probabilities_csv,
                                        index_col=np.arange(0, 6))
            print(f"Existing {probabilities_csv} found …")
            assert len(probabilities) == n_combs
        except FileNotFoundError:
            print(f"No existing {probabilities_csv} found, computing pairwise "
                  f"statistical tests from scratch, may take some time …")
            # While I would prefer the more declarative form, we use a loop so that
            # the user doesn't have to stare at a blank screen for several minutes.
            # probabilities = {(key1, key2): stat_test(np.array(options[key1]),
            # np.array(options[key2])) for key1, key2 in combinations(dict(options),
            # 2)}
            probabilities_dict = {}
            i = 0
            for key1, key2 in combinations(dict(options), 2):
                i += 1
                print(f"Comparing ({i}/{n_combs}): {key1} vs. {key2}")
                result = stat_test(np.array(options[key1]),
                                   np.array(options[key2]),
                                   rope=rope)
                print(result)
                probabilities_dict[(key1, key2)] = result

            # Flatten the tuples in the keys so pandas index behaves better.
            probabilities_dict_ = {
                tuple(flatten(key)): probabilities_dict[key]
                for key in probabilities_dict
            }

            probabilities = pd.DataFrame(probabilities_dict_).T

            probabilities = probabilities.rename(columns={
                0: "left",
                1: "rope",
                2: "right"
            })

            # Convert strings in index to numbers.
            probabilities.index = probabilities.index.map(strs_to_nums)

            # Add names to index.
            index_names = [
                "POP_SIZE1", "E01", "BETA1", "POP_SIZE2", "E02", "BETA2"
            ]
            probabilities.index = probabilities.index.rename(index_names)

            probabilities.to_csv(probabilities_csv)

        # Note that we compare differences of errors. This implies that p(right) is
        # the probability that the second option's error is higher than the one of
        # the first option (unlike the accuracy metrics used for classification,
        # where it is the other way around). Analogously, p(left) corresponds to the
        # probability of the first option being worse than the second option.

        # For this analysis, we're fine with 80% certainty.
        decision_point = 0.8
        print(f"Set decision point to {decision_point} …")

        candidates = [strs_to_nums(k) for k in options.keys()]
        print(f"We have {len(candidates)} configurations to choose from …")

        print(f"Computing domination relation and building graph …")
        # For debugging purposes, we draw a graph which shows the overall structure
        # of the domination relation.
        G = nx.DiGraph()
        G.add_nodes_from(candidates)

        better_than = {c: [] for c in candidates}
        practically_equal = {c: [] for c in candidates}
        worse_than = {c: [] for c in candidates}
        for (key, val) in probabilities.iterrows():
            cand1 = key[0:3]
            cand2 = key[3:]
            if val["left"] >= decision_point:
                worse_than[cand1].append(cand2)
                better_than[cand2].append(cand1)
                # “better than” edge
                G.add_edge(cand2, cand1)
            if val["rope"] >= decision_point:
                practically_equal[cand1].append(cand2)
                practically_equal[cand2].append(cand1)
            if val["right"] >= decision_point:
                worse_than[cand2].append(cand1)
                better_than[cand1].append(cand2)
                # “better than” edge
                G.add_edge(cand1, cand2)

        if graphs:
            layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
            nx.draw_networkx(G, pos=layout)
            plt.show()

        len_worse_than = pd.Series({k: len(worse_than[k]) for k in worse_than})
        len_better_than = pd.Series(
            {k: len(better_than[k])
             for k in better_than})
        len_practically_equal = pd.Series(
            {k: len(practically_equal[k])
             for k in practically_equal})

        print(
            "Computing dominating parametrizations that dominate the most other "
            f"parametrizations …")
        # We now want to find the solutions that dominate the most solutions (may be
        # more than one) and are not dominated by any other solution.
        # Remove the candidates that are dominated by others.
        max_dom_candidates = len_better_than[len_worse_than == 0]
        # Find the maximum domination count.
        max_dom_count = max_dom_candidates.max()
        # Find all candidates with that domination count.
        max_dom_candidates = max_dom_candidates[max_dom_candidates ==
                                                max_dom_count].index.values

        print()

        rope_decision_point = 0.99
        print("Checking maximally dominating parametrizations pairwise for "
              f"p(rope) < {rope_decision_point}.")
        for cand1 in max_dom_candidates:
            for cand2 in [
                    cand for cand in max_dom_candidates if cand != cand1
            ]:
                try:
                    comparison = probabilities.loc[tuple(
                        flatten((cand1, cand2)))]
                except KeyError:
                    comparison = probabilities.loc[tuple(
                        flatten((cand2, cand1)))]
                    # In this case swap the candidates for correct ordering of
                    # the probabilities.
                    tmp = cand1
                    cand1 = cand2
                    cand2 = tmp
                if comparison["rope"] < 0.99:
                    print(f"Comparison not rope between: {cand1} vs. {cand2}")
                    print(f"Probabilities: {tuple(comparison.values)}")
                    print()
        print(">>> Choose one of the following practically equivalent "
              f"maximally dominating parametrizations for {exp_name} <<<")
        max_dom_candidates_df = pd.DataFrame(
            [list(c) for c in max_dom_candidates], columns=changed_params)
        print(max_dom_candidates_df)


if __name__ == "__main__":
    main()
