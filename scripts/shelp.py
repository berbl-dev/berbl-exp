import click
import subprocess


@click.group()
def cli():
    """
    Helpers when running stuff on a Slurm cluster.
    """
    pass


@cli.command()
def watch():
    squeue_format = ("'%.18i %.9P %.8u %.2t %.10M %.5D %.4C %.10m %.12n %.12N "
                     "%.10Q %.19L %.19S %.19e %R'")
    subprocess.run(
        ["watch", "-n", "60", "squeue", "-o", squeue_format, "-u", "hoffmada"])


# TODO Add most_recent_stdout
# TODO Add other earlier scripts (fails etc.)

if __name__ == "__main__":
    cli()
