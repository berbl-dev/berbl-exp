import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


# NOTE Drugowitsch writes noise variance being 0.15 in his text but uses
# std=0.15 in the code accompanying his book.
def f(x, noise_std=0.15, random_state: np.random.RandomState = 0):
    random_state = check_random_state(random_state)
    return np.sin(2 * np.pi * x) + random_state.normal(
        0, noise_std, size=x.shape)


def generate(n: int = 300, random_state: np.random.RandomState = 0):
    """
    Creates a sample from the fourth benchmark function in (Drugowitsch, 2007;
    [PDF p. 265]).

    Parameters
    ----------
    n : int
        The number of samples to generate. Supplying ``X`` overrides this.
    noise : bool
        Whether to generate noisy data (the default) or not. The latter may be
        useful for visualization purposes.
    X : Sample the function at these exact input points (instead of generating
        ``n`` input points randomly).

    Returns
    -------
    array of shape (N, 1)
        input matrix X
    array of shape (N, 1)
        output matrices y
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
    y_denoised = f(X_denoised, noise_std=0)

    return {
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test_true": y_test_true,
        "X_denoised": X_denoised,
        "y_denoised": y_denoised
    }
