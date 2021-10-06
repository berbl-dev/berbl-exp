import numpy as np  # type: ignore
from prolcs.common import check_phi, matching_matrix
from prolcs.literal import mixing
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from prolcs.utils import add_bias
from sklearn.utils import check_random_state  # type: ignore

# The individual used in function generation.
ms = [
    RadialMatch1D(mu=0.2, sigma_2=0.05, has_bias=False),
    RadialMatch1D(mu=0.5, sigma_2=0.01, has_bias=False),
    RadialMatch1D(mu=0.8, sigma_2=0.05, has_bias=False),
]

def generate(n: int = 300,
             noise=True,
             X=None,
             random_state: np.random.RandomState = 0):
    """
    Creates a sample from the first benchmark function in (Drugowitsch, 2007;
    [PDF p. 260]) which is generated by three rbf matching–based classifiers.

    Note that the input range is [0, 1] (i.e. standardization may be necessary).

    Parameters
    ----------
    n : int
        The size of the sample to generate. Supplying ``X`` overrides this.
    noise : bool
        Whether to generate noisy data (the default) or not. The latter may be
        useful for visualization purposes.
    X : Sample the function at these exact input points (instead of generating
        ``n`` input points uniformly at random).

    Returns
    -------
    array of shape (N, 1)
        input matrix X
    array of shape (N, 1)
        output matrices y
    """
    random_state = check_random_state(random_state)

    if X is None:
        X = random_state.random((n, 1))

    M = matching_matrix(ms, X)
    Phi = check_phi(None, X)

    W = [
        np.array([0.05, 0.5]),
        np.array([2, -4]),
        np.array([-1.5, 2.5]),
    ]
    Lambda_1 = [
        np.array([0.1]),
        np.array([0.1]),
        np.array([0.1]),
    ]
    V = np.array([0.5, 1.0, 0.4]).reshape(1, 3)

    G = mixing(M, Phi, V)

    # After matching, augment samples by prepending 1 to enable non-zero
    # intercepts.
    X_ = add_bias(X)
    y = np.zeros(X.shape)
    for n in range(len(X)):
        y[n] = 0
        for k in range(len(ms)):
            # sample the three classifiers
            if noise:
                y[n] += random_state.normal(loc=G[n][k] * (W[k] @ X_[n]),
                                            scale=Lambda_1[k])
            else:
                y[n] += G[n][k] * (W[k] @ X_[n])

    # Return the non-augmented data points.
    return X, y
