# TODO Why do I only get a fitness of ~52 instead of Drugowitschs >100?

import click
import numpy as np  # type: ignore
from tasks.book.generated_function import generate
from berbl.match.softinterval1d_drugowitsch import SoftInterval1D

from ... import experiment


@click.command()
@click.option("-n", "--n-iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--data-seed", type=click.IntRange(min=0), default=1)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
@click.option("--standardize/--no-standardize", type=bool, default=False)
def run_experiment(n_iter, seed, data_seed, show, sample_size, standardize):

    X, y = generate(sample_size)
    X_test, y_test_true = generate(1000, random_state=data_seed)

    # generate equidistant, denoised data as well (only for visual reference)
    X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
    _, y_denoised = generate(1000, noise=False, X=X_denoised)

    gaparams = {
        "n": 8,
        "p": 0.5,
        "tournsize": 5,
    }
    experiment("lit.book.int.generated_function",
               SoftInterval1D,
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
               literal=False,
               standardize=standardize)


if __name__ == "__main__":
    run_experiment()
