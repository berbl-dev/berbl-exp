import os
import importlib
import pathlib
import shutil
from subprocess import Popen, PIPE, STDOUT
import tempfile

import click


@click.group()
def main():
    pass


# NEXT Add repeat and
# THEN Fix Slurm stuff
# THEN Add all().


def get_data(module, data_seed):
    task_path = f"tasks.{module}"
    task_mod = importlib.import_module(task_path)
    return task_mod.data(data_seed)


def experiment_name(algorithm, module):
    return f"{algorithm}.{module}"


def maybe_override(params, param, value):
    if value is not None and params[param] != value:
        print(f"Warning: Overriding {param}={params[param]} with "
              f"{param}={value}")
        params[param] = value


@click.command()
@click.argument("ALGORITHM")
@click.argument("MODULE")
@click.option("-n", "--n-iter", type=click.IntRange(min=1))
@click.option("-s",
              "--seed",
              type=click.IntRange(min=0),
              default=0,
              show_default=True)
@click.option("--data-seed",
              type=click.IntRange(min=0),
              default=1,
              show_default=True)
@click.option("--show/--no-show", type=bool, default=False, show_default=True)
@click.option("--standardize/--no-standardize",
              type=bool,
              default=False,
              show_default=True)
# only applicable to berbl
@click.option("--fit-mix", type=str, default=None)
@click.option("--literal/--no-literal", type=bool, default=None)
@click.option("--match", type=str, default=None)
# only applicable to XCSF
@click.option("-p", "--pop_size", type=click.IntRange(min=1), default=100)
def single(algorithm, module, n_iter, seed, data_seed, show, standardize,
           fit_mix, literal, match, pop_size):
    """
    Use ALGORITHM ("berbl" or "xcsf") in an experiment defined by MODULE
    (module path appended to "experiments.ALGORITHM.").
    """
    algorithms = ["berbl", "xcsf"]
    if not algorithm in algorithms:
        print(f"ALGORITHM has to be one of {algorithms} but is {algorithm}")
        exit(1)

    exp = experiment_name(algorithm, module)
    data = get_data(module, data_seed)

    param_path = f"experiments.{exp}"
    param_mod = importlib.import_module(param_path)
    params = param_mod.params

    if algorithm == "berbl":
        maybe_override(params, "n_iter", n_iter)
        maybe_override(params, "match", match)
        maybe_override(params, "literal", literal)
        maybe_override(params, "fit_mixing", fit_mix)
        from experiments.berbl import run_experiment
        run_experiment(name=exp,
                       data=data,
                       standardize=standardize,
                       seed=seed,
                       params=params,
                       show=show)
    elif algorithm == "xcsf":
        maybe_override(params, "MAX_TRIALS", n_iter)
        from experiments.xcsf import run_experiment
        run_experiment(name=exp,
                       data=data,
                       standardize=standardize,
                       seed=seed,
                       params=params,
                       show=show)
        # TODO Optimize parameters for each experiment
    else:
        print(f"Algorithm {algorithm} not one of [berbl, xcsf].")


@click.command()
def all():
    """
    Runs all the experiments in sequence.
    """
    n_reps = 5
    n_data_sets = 5

    seeds = range(n_reps)
    data_seeds = range(n_reps, n_reps + n_data_sets)
    # Name of task and whether soft interval matching is used.
    tasks = [
        "book.generated_function",
        "book.sparse_noisy_data",
        "book.noisy_sinus",
        "book.variable_noise",
        # Not in the book but required for fairer comparison with XCSF.
        "additional_literal.generated_function",
        "additional_literal.sparse_noisy_data",
        # Expected to behave the same as the literal implementation.
        "non_literal.generated_function",
        "non_literal.sparse_noisy_data",
        "non_literal.noisy_sinus",
        "non_literal.variable_noise",
    ]

    from experiments.berbl import run_experiment
    for seed in seeds:
        for data_seed in data_seeds:
            for task in tasks:
                data = get_data(task, data_seed)
                exp = experiment_name("berbl", task)
                param_path = f"experiments.{exp}"
                param_mod = importlib.import_module(param_path)
                params = param_mod.params
                run_experiment(name=exp,
                               data=data,
                               standardize=False,
                               seed=seed,
                               params=params,
                               show=False)
                run_experiment(name=exp,
                               data=data,
                               standardize=True,
                               seed=seed,
                               params=params,
                               show=False)

    from experiments.xcsf import run_experiment
    for seed in seeds:
        for data_seed in data_seeds:
            for task in tasks:
                data = get_data(task, data_seed)
                exp = experiment_name("xcsf", task)
                param_path = f"experiments.{exp}"
                param_mod = importlib.import_module(param_path)
                params = param_mod.params
                run_experiment(name=exp,
                               data=data,
                               standardize=True,
                               seed=seed,
                               params=params,
                               show=False)
                # TODO Optimize parameters for each experiment

    # TODO Store run IDs somewhere and then use them in eval

    pass


main.add_command(single)
main.add_command(all)

if __name__ == "__main__":
    main()

# Local Variables:
# mode: python
# End:
