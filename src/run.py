import os
import importlib
import pathlib
import shutil
from subprocess import Popen, PIPE, STDOUT
import tempfile

import click


@click.group()
@click.pass_context
@click.argument("ALGORITHM")
def main(ctx, algorithm):
    ctx.ensure_object(dict)
    algorithms = ["berbl", "xcsf"]
    if not algorithm in algorithms:
        print(f"ALGORITHM has to be one of {algorithms} but is {algorithm}")
        exit(1)
    ctx.obj["algorithm"] = algorithm




@click.command()
@click.pass_context
@click.argument("MODULE")
@click.option("-n", "--n-iter", type=click.IntRange(min=1))
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--data-seed", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--show/--no-show", type=bool, default=False, show_default=True)
@click.option("--standardize/--no-standardize", type=bool, default=False, show_default=True)
# only applicable to berbl
@click.option("--fit-mix", type=str, default="laplace")
@click.option("--literal/--no-literal", type=bool, default=False)
@click.option("--softint/--no-softint", type=bool, default=False)
# only applicable to XCSF
@click.option("-p", "--pop_size", type=click.IntRange(min=1), default=100)
def single(ctx, module, n_iter, seed, data_seed, show,
           standardize, fit_mix, literal, softint, pop_size):
    """
    Use ALGORITHM ("berbl" or "xcsf") in an experiment defined by MODULE
    (module path appended to "experiments.ALGORITHM.").
    """
    algorithm = ctx.obj["algorithm"]

    task_path = f"tasks.{module}"
    exp = f"{algorithm}.{module}"
    param_path = f"experiments.{exp}"

    task_mod = importlib.import_module(task_path)
    data = task_mod.data(data_seed)

    if algorithm == "berbl":
        param_mod = importlib.import_module(param_path)
        gaparams = param_mod.gaparams
        if n_iter is not None:
            print(f"Warning: Overriding n_iter={gaparams['n_iter']} with "
                  f"n_iter={n_iter}")
            gaparams["n_iter"] = n_iter
        from experiments.berbl import run_experiment
        run_experiment(name=exp,
                       softint=softint,
                       gaparams=gaparams,
                       data=data,
                       seed=seed,
                       show=show,
                       literal=literal,
                       standardize=standardize,
                       fit_mixing=fit_mix)
    elif algorithm == "xcsf":
        from experiments.xcsf import run_experiment
        run_experiment(name=exp,
                       data=data,
                       pop_size=pop_size,
                       n_iter=n_iter,
                       seed=seed,
                       show=show,
                       standardize=standardize,
                       # TODO Optimize parameters for each experiment
                       params=None)
    else:
        print(f"Algorithm {algorithm} not one of [berbl, xcsf].")


@click.command()
@click.option("-s",
              "--seed",
              type=click.IntRange(min=0),
              default=0,
              help="Seed of the first repetition to run.")
@click.option("--data-seed",
              type=click.IntRange(min=0),
              default=1,
              help="Seed for generating the data.")
@click.option("-t",
              "--time",
              type=click.IntRange(min=10),
              default=30,
              help="Slurm's --time in minutes, (default: 30).")
@click.option("--mem",
              type=click.IntRange(min=1),
              default=100,
              help="Slurm's --mem in megabytes, (default: 100).")
@click.option("-r",
              "--reps",
              type=click.IntRange(min=1),
              default=10,
              help="Number of repetitions to run.")
@click.argument("experiment")
def repeat(seed, data_seed, time, reps, mem, experiment):
    pass


@click.command()
@click.argument("experiment")
def all(seed, data_seed, time, reps, mem, experiment):
    """
    Runs all the experiments in sequence.
    """
    reps = 30
    seeds = range(reps)
    data_seeds = range(reps, reps + reps)

    # experiments/berbl/book/generated_function.py (int)
    # experiments/berbl/book/noisy_sinus.py
    # experiments/berbl/book/variable_noise.py
    # experiments/berbl/book/sparse_noisy_data.py (int)
    # experiments/berbl/book/sparse_noisy_data.py

    # experiments/xcsf/generated_function.py
    # experiments/xcsf/parameter_search
    # experiments/xcsf/parameter_search/generated_function.py
    pass


main.add_command(single)
# main.add_command(repeat)
# main.add_command(all)

if __name__ == "__main__":
    main()

# Local Variables:
# mode: python
# End:
