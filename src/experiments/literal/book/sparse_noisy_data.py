import click
import numpy as np  # type: ignore
from tasks.book.sparse_noisy_data import f, generate
from berbl.match.radial1d_drugowitsch import RadialMatch1D

from . import experiment


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--data-seed", type=click.IntRange(min=0), default=1)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=200)
@click.option("--standardize/--no-standardize", type=bool, default=False)
def run_experiment(n_iter, seed, data_seed, show, sample_size, standardize):

    X, y = generate(sample_size)
    X_test, y_test_true = generate(1000, random_state=data_seed)

    # generate equidistant, denoised data as well (only for visual reference)
    X_denoised = np.linspace(0, 4, 100)[:, np.newaxis]
    y_denoised = f(X_denoised, noise_var=0)

    gaparams = {
        "n": 4,
        "p": 0.5,
        "tournsize": 5,
    }
    experiment("lit.book.sparse_noisy_data",
               RadialMatch1D,
               gaparams,
               X,
               y,
               X_test,
               y_test_true,
               X_denoised,
               y_denoised,
               n_iter,
               seed,
               show,
               sample_size,
               standardize=standardize)


if __name__ == "__main__":
    run_experiment()
