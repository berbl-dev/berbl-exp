import importlib

import click
from experiments.berbl import BERBLExperiment
from experiments.xcsf import XCSFExperiment


@click.group()
def main():
    pass


# NEXT Add repeat and
# THEN Fix Slurm stuff
# THEN Add all().


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

    if algorithm == "berbl":
        exp = BERBLExperiment(module, seed, data_seed, standardize,
                              show)
        exp.run(n_iter=n_iter,
                match=match,
                literal=literal,
                fit_mixing=fit_mix)
    elif algorithm == "xcsf":
        exp = XCSFExperiment(module, seed, data_seed, standardize,
                             show)
        exp.run(MAX_TRIALS=n_iter)
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

    for seed in seeds:
        for data_seed in data_seeds:
            for module in tasks:
                exp = BERBLExperiment(module,
                                      seed,
                                      data_seed,
                                      standardize=False,
                                      show=False)
                exp.run()
                exp = BERBLExperiment(module,
                                      seed,
                                      data_seed,
                                      standardize=True,
                                      show=False)
                exp.run()

    from experiments.xcsf import run_experiment
    for seed in seeds:
        for data_seed in data_seeds:
            for module in tasks:
                exp = XCSFExperiment(module,
                                     seed,
                                     data_seed,
                                     standardize=True,
                                     show=False)
                exp.run()
                # TODO Optimize parameters for each experiment

    # TODO Store run IDs somewhere and then use them in eval


@click.command()
def slurm():
    ...
    # TODO Implement Slurm batch submitter


main.add_command(single)
main.add_command(all)

if __name__ == "__main__":
    main()

# Local Variables:
# mode: python
# End:
