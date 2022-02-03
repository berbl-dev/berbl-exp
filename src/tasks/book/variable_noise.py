import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


# NOTE Drugowitsch writes about noise variances in his text but uses the
# specified values for std in the code accompanying his book.
def f(X, noise_stds=(0.6, 0.1), random_state: np.random.RandomState = 0):
    """
    Function underlying the variable noise samples.

    Parameters
    ----------
    X : array-like
        The input point(s) at which to retrieve the function values.
    noise_stds : pair of floats
        The two different noise standard deviations for input values below ``0``
        and above ``0``.
    """
    random_state = check_random_state(random_state)
    return np.where(
        X < 0, -1 - 2 * X
        + random_state.normal(0, noise_stds[0], size=X.shape), -1
        + 2 * X + random_state.normal(0, noise_stds[1], size=X.shape))


def generate(n: int = 200, random_state: np.random.RandomState = 0):
    """
    [PDF p. 262]

    :param n: the number of samples to generate

    :returns: input and output matrices X (N × 1) and y (N × 1)
    """
    random_state = check_random_state(random_state)

    X = random_state.uniform(low=-1, high=1, size=(n, 1))
    y = f(X, random_state=random_state)

    return X, y


def data(data_seed):
    X, y = generate(random_state=data_seed)
    X_test, y_test_true = generate(1000, random_state=data_seed + 1)

    # generate equidistant, denoised data as well (only for visual reference)
    X_denoised = np.linspace(-1, 1, 100)[:, np.newaxis]
    y_denoised = f(X_denoised, noise_stds=(0, 0))

    return {
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test_true": y_test_true,
        "X_denoised": X_denoised,
        "y_denoised": y_denoised
    }
