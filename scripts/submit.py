import os
import pathlib
import shutil
import tempfile
from datetime import datetime
from subprocess import PIPE, Popen

import click


def get_dirs():
    job_dir = os.getcwd()
    datetime_ = datetime.now().isoformat()
    results_dir = f"{job_dir}/results/{datetime_}"
    os.makedirs(f"{results_dir}/output", exist_ok=True)
    os.makedirs(f"{results_dir}/jobs", exist_ok=True)
    return job_dir, results_dir


@click.group()
def cli():
    pass


@cli.command()
@click.option("-o",
              "--slurm-options",
              type=str,
              default=None,
              show_default=True,
              help=("Override Slurm options "
                    "(for now, see file source for defaults)"))
@click.option("-r",
              "--n-reps",
              type=int,
              default=30,
              show_default=True,
              help="Number of times to repeat experiment")
# TODO Maybe @click.option("--seed-offset") to set seed_offset
@click.option("--experiment-name", type=str, required=True)
@click.option("-F",
              "--config-file",
              default=None,
              type=str,
              show_default=True,
              help="Algorithm configuration to use")
@click.argument("NPZFILE")
def submit(slurm_options, n_reps, experiment_name, config_file, npzfile):

    if slurm_options is not None:
        raise NotImplementedError("Has to be implemented")

    job_dir, results_dir = get_dirs()

    seed_offset = 0

    sbatch = "\n".join([
        f'#!/usr/bin/env bash',  #
        # Default Slurm settings.
        f'#SBATCH --nodelist=oc-compute03',
        f'#SBATCH --time=1-00:00:00',
        f'#SBATCH --mem=2G',
        f'#SBATCH --partition=cpu-prio',
        f'#SBATCH --output="{results_dir}/output/output-%A-%a.txt"',
        f'#SBATCH --array=0-{n_reps - 1}',
        (
            # Always use srun within sbatch.
            # https://stackoverflow.com/a/53640511/6936216
            f'srun '
            f'nix develop "{job_dir}" --command '
            f'python scripts/run.py run "{npzfile}" '
            f'{"" if config_file is None else "--config-file={config_file}"} '
            f'--experiment-name={experiment_name} '
            f'--seed=$(({seed_offset} + $SLURM_ARRAY_TASK_ID)) '
            '--run-name=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}\n')
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

    sbatch_dir = f"{results_dir}/jobs"
    os.makedirs(sbatch_dir, exist_ok=True)
    tmppath = pathlib.Path(tmp.name)
    fname = pathlib.Path(sbatch_dir, f"{jobid}.sbatch")
    shutil.copy(tmppath, fname)
    print(f"Renamed {tmp.name} to {fname}")


if __name__ == "__main__":
    cli()
