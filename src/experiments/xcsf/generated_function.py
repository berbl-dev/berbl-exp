import click
import numpy as np
from tasks.book.generated_function import generate

from . import experiment


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-p", "--pop_size", type=click.IntRange(min=1), default=100)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
@click.option("--standardize/--no-standardize", type=bool, default=True)
def run_experiment(n_iter, pop_size, seed, show, sample_size, standardize):

    X, y = generate(sample_size)
    X_test, y_test_true = generate(1000, random_state=12345)

    X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
    _, y_denoised = generate(1000, noise=False, X=X_denoised)

    experiment("xcsf.generated_function",
               X,
               y,
               X_test,
               y_test_true,
               X_denoised,
               y_denoised,
               pop_size,
               n_iter,
               seed,
               show,
               sample_size,
               standardize=standardize)


if __name__ == "__main__":
    run_experiment()
