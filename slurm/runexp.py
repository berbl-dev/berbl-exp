#! /usr/bin/env nix-shell
#! nix-shell -i python -p "python38.withPackages(ps: with ps; [click])"

import os
import pathlib
import shutil
from subprocess import Popen, PIPE, STDOUT
import tempfile

import click


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
def run_experiment(seed, data_seed, time, reps, mem, experiment):
    """
    Run EXPERIMENT on the cluster.
    """
    experiment = f"experiments/{experiment}"

    job_dir = "/data/oc-compute02/hoffmada/prolcs-reproducibility-experiments"

    path = pathlib.Path(job_dir, "src", f"{experiment}.py")
    if not path.is_file():
        print(f"Experiment {path.name} does not exist. Check path ({path}).")
        exit(1)
    # Use module instead of path (otherwise we get errors when using
    # relative imports).
    experiment = experiment.replace("/", ".")

    sbatch = "\n".join([
        f'#!/usr/bin/env bash',  #
        f'#SBATCH --nodelist=oc-compute03',
        f'#SBATCH --time={time}',
        f'#SBATCH --mem={mem}',
        f'#SBATCH --partition=cpu',
        f'#SBATCH --output={job_dir}/output/output-%A-%a.txt',
        f'#SBATCH --array=0-{reps - 1}',
        (f'nix-shell "{job_dir}/shell.nix" --command '
         f'"PYTHONPATH=\'{job_dir}/src:$PYTHONPATH\' python -m {experiment} '
         f'--seed=$(({seed} + $SLURM_ARRAY_TASK_ID)) '
         f'--data-seed=$(({data_seed} + $SLURM_ARRAY_TASK_ID))"')
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

if __name__ == "__main__":
    run_experiment()

# Local Variables:
# mode: python
# End:
