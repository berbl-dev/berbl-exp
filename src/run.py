import os
import pathlib
import shutil
import tempfile
from subprocess import PIPE, Popen

import click
from experiments.berbl import BERBLExperiment
from experiments.xcsf import XCSFExperiment
from experiments.xcsf.parameter_search import param_grid


@click.group()
def main():
    pass


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
@click.option("-p", "--pop-size", type=click.IntRange(min=1), default=100)
@click.option("--epsilon-zero", type=float, default=0.01)
@click.option("--beta", type=float, default=0.01)
@click.option("--run-name", type=str, default=None)
# TODO Extract tracking-uri option to context
@click.option("--tracking-uri", type=str, default="mlruns")
def single(algorithm, module, n_iter, seed, data_seed, show, standardize,
           fit_mix, literal, match, pop_size, epsilon_zero, beta, run_name,
           tracking_uri):
    """
    Use ALGORITHM ("berbl" or "xcsf") in an experiment defined by MODULE
    (module path appended to "experiments.ALGORITHM.").
    """
    algorithms = ["berbl", "xcsf"]
    if not algorithm in algorithms:
        print(f"ALGORITHM has to be one of {algorithms} but is {algorithm}")
        exit(1)

    if algorithm == "berbl":
        exp = BERBLExperiment(module, seed, data_seed, standardize, show,
                              run_name, tracking_uri)
        exp.run(n_iter=n_iter,
                match=match,
                literal=literal,
                fit_mixing=fit_mix)
    elif algorithm == "xcsf":
        exp = XCSFExperiment(module, seed, data_seed, standardize, show,
                             run_name, tracking_uri)
        exp.run(MAX_TRIALS=n_iter,
                POP_SIZE=pop_size,
                E0=epsilon_zero,
                BETA=beta)
        # TODO Optimize parameters for each experiment
    else:
        print(f"Algorithm {algorithm} not one of [berbl, xcsf].")


n_reps = 5
n_data_sets = 5
seeds = range(0, n_reps)
data_seeds = range(n_reps, n_reps + n_data_sets)
seed0 = 0
data_seed0 = 0
# Name of task and whether soft interval matching is used.
berbl_experiments = [
    "book.generated_function",
    "book.sparse_noisy_data",
    "book.sine",
    "book.variable_noise",
    # Not in the book but required for fairer comparison with XCSF.
    "additional_literal.generated_function",
    "additional_literal.sparse_noisy_data",
    # Expected to behave the same as the literal implementation.
    "non_literal.generated_function",
    "non_literal.sparse_noisy_data",
    "non_literal.sine",
    "non_literal.variable_noise",
]
xcsf_experiments = [
    "book.generated_function",
    "book.sparse_noisy_data",
    "book.sine",
    "book.variable_noise",
]


@click.command()
@click.option("--tracking-uri", type=str, default="mlruns")
def all(tracking_uri):
    """
    Runs all the experiments in sequence.
    """
    for data_seed in data_seeds:
        for seed in seeds:
            for module in berbl_experiments:
                exp = BERBLExperiment(module,
                                      seed,
                                      data_seed,
                                      standardize=False,
                                      show=False,
                                      tracking_uri=tracking_uri)
                exp.run()
                exp = BERBLExperiment(module,
                                      seed,
                                      data_seed,
                                      standardize=True,
                                      show=False,
                                      tracking_uri=tracking_uri)
                exp.run()

    for data_seed in data_seeds:
        for seed in seeds:
            for module in xcsf_experiments:
                exp = XCSFExperiment(module,
                                     seed,
                                     data_seed,
                                     standardize=True,
                                     show=False,
                                     tracking_uri=tracking_uri)
                exp.run()
                # TODO Optimize parameters for each experiment

    # TODO Store run IDs somewhere and then use them in eval


def submit(node,
           time,
           mem,
           algorithm,
           module,
           standardize,
           tracking_uri,
           params=""):
    """
    Submit one ``single(…)`` job to the cluster for each repetition.
    """
    job_dir = os.getcwd()
    os.makedirs("output", exist_ok=True)
    os.makedirs("jobs", exist_ok=True)

    njobs = n_reps * n_data_sets

    standardize_option = '--standardize' if standardize else '--no-standardize'
    sbatch = "\n".join([
        f'#!/usr/bin/env bash',  #
        f'#SBATCH --nodelist={node}',
        f'#SBATCH --time={time}',
        f'#SBATCH --mem={mem}',
        f'#SBATCH --partition=cpu',
        f'#SBATCH --output={job_dir}/output/output-%A-%a.txt',
        f'#SBATCH --array=0-{njobs - 1}',
        (
            f'nix-shell "{job_dir}/shell.nix" --command '
            f'"PYTHONPATH=\'{job_dir}/src:$PYTHONPATH\' '
            f'python -m run single {algorithm} {module} '
            f'{standardize_option} '
            # NOTE / is integer division in bash.
            f'--seed=$(({seed0} + $SLURM_ARRAY_TASK_ID / {n_data_sets})) '
            f'--data-seed=$(({data_seed0} + $SLURM_ARRAY_TASK_ID % {n_data_sets})) '
            '--run-name=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} '
            f'--tracking-uri={tracking_uri} '
            f'{params}"')
    ])
    print(sbatch)
    print()

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w+") as f:
        f.write(sbatch)
    print(f"Wrote sbatch to {tmp.name}.")
    print()

    p = Popen(["sbatch", f"{tmp.name}"], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    output = p.communicate()
    stdout = output[0].decode("utf-8")
    stderr = output[1].decode("utf-8")
    print(f"stdout:\n{stdout}\n")
    print(f"stderr:\n{stderr}\n")
    jobid = int(stdout.replace("Submitted batch job ", ""))
    print(f"Job ID: {jobid}")
    print()

    sbatch_dir = f"{job_dir}/jobs"
    os.makedirs(sbatch_dir, exist_ok=True)
    tmppath = pathlib.Path(tmp.name)
    fname = pathlib.Path(sbatch_dir, f"{jobid}.sbatch")
    shutil.copy(tmppath, fname)
    print(f"Renamed {tmp.name} to {fname}")


@click.command()
@click.argument("NODE")
@click.option("-t",
              "--time",
              type=click.IntRange(min=10),
              default=60,
              help="Slurm's --time in minutes.",
              show_default=True)
@click.option("--mem",
              type=click.IntRange(min=1),
              default=1000,
              help="Slurm's --mem in megabytes.",
              show_default=True)
@click.option("--tracking-uri", type=str, default="mlruns")
def slurm(node, time, mem, tracking_uri):
    """
    Submits all experiments to NODE.
    """
    for module in berbl_experiments:
        submit(node,
               time,
               mem,
               "berbl",
               module,
               standardize=False,
               tracking_uri=tracking_uri)
        submit(node,
               time,
               mem,
               "berbl",
               module,
               standardize=True,
               tracking_uri=tracking_uri)
    for module in xcsf_experiments:
        submit(node,
               time,
               mem,
               "xcsf",
               module,
               standardize=True,
               tracking_uri=tracking_uri)


@click.command()
@click.argument("NODE")
@click.argument("ALGORITHM")
@click.argument("MODULE")
@click.option("-t",
              "--time",
              type=click.IntRange(min=10),
              default=60,
              help="Slurm's --time in minutes.",
              show_default=True)
@click.option("--mem",
              type=click.IntRange(min=1),
              default=1000,
              help="Slurm's --mem in megabytes.",
              show_default=True)
@click.option("--standardize/--no-standardize",
              type=bool,
              default=False,
              show_default=True)
@click.option("--tracking-uri", type=str, default="mlruns")
def slurm1(node, algorithm, module, time, mem, standardize, tracking_uri):
    """
    Submits a single experiment (running ALGORITHM on task MODULE) to NODE.
    """
    submit(node,
           time,
           mem,
           algorithm,
           module,
           standardize=standardize,
           tracking_uri=tracking_uri)


def make_param_string(params):
    return (f"--pop-size={params['POP_SIZE']} "
            f"--epsilon_zero={params['E0']} "
            f"--beta={params['BETA']}")


@click.command()
@click.argument("NODE")
@click.option("-t",
              "--time",
              type=click.IntRange(min=10),
              default=60,
              help="Slurm's --time in minutes.",
              show_default=True)
@click.option("--mem",
              type=click.IntRange(min=1),
              default=1000,
              help="Slurm's --mem in megabytes.",
              show_default=True)
@click.option("--tracking-uri", type=str, default="mlruns")
def paramsearch(node, time, mem, tracking_uri):
    for module in xcsf_experiments:
        for params in param_grid:
            submit(node,
                   time,
                   mem,
                   "xcsf",
                   module,
                   standardize=True,
                   tracking_uri=tracking_uri,
                   params=make_param_string(params))


main.add_command(paramsearch)
main.add_command(single)
main.add_command(all)
main.add_command(slurm)
main.add_command(slurm1)

if __name__ == "__main__":
    main()

# Local Variables:
# mode: python
# End:
