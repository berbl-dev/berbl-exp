import numpy as np  # type: ignore
from berbl.literal import mixing
from berbl.match.radial1d_drugowitsch import RadialMatch1D
from berbl.utils import add_bias, check_phi, matching_matrix
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
                y[n] += G[n][k] * random_state.normal(loc=(W[k] @ X_[n]),
                                                      scale=Lambda_1[k])
            else:
                y[n] += G[n][k] * (W[k] @ X_[n])

    # LCSBookCode does it like this.
    # def own_f(x):
    #     """Returns f(x) for given x.
    #     """
    #     from numpy import (arange, array, dot, double, empty, exp, hstack, inf,
    #                        linspace, ones, pi, power, sin, sort, sqrt, sum)
    #     # functions are
    #     # f1(x) = 0.05 + 0.5 x
    #     # f2(x) = 2 - 4 x
    #     # f3(x) = -1.5 + 2.5 x
    #     fns = array([[0.05, 0.5], [2.0, -4.0], [-1.5, 2.5]], double)
    #     # gaussian basis functions are given by (mu, var, weight):
    #     # (0.2, 0.05), (0.5, 0.01), (0.8, 0.05)
    #     gbfs = array([[0.2, 0.05, 0.5], [0.5, 0.01, 1.0], [0.8, 0.05, 0.4]],
    #                  double)
    #     # plain function values
    #     fx = fns[:, 0] + x * fns[:, 1]
    #     #print "%f\t%f\t%f\t%f" % (x, fx[0], fx[1], fx[2])
    #     # mixing weights
    #     mx = gbfs[:, 2] * exp(-0.5 / gbfs[:, 1] * power(x - gbfs[:, 0], 2.0))
    #     mx /= sum(mx)
    #     #print "%f\t%f\t%f\t%f" % (x, mx[0], mx[1], mx[2])
    #     # return mixed function
    #     return dot(fx, mx)
    # y_ = np.array([own_f(x) for x in X])
    # y_ += random_state.normal(size=y_.shape) * 0.1

    # Return the non-augmented data points.
    return X, y


def data(data_seed):
    X, y = generate(random_state=data_seed)
    X_test, y_test_true = generate(1000, random_state=data_seed + 1)

    # generate equidistant, denoised data as well (only for visual reference)
    X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
    _, y_denoised = generate(1000, noise=False, X=X_denoised)

    return {
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test_true": y_test_true,
        "X_denoised": X_denoised,
        "y_denoised": y_denoised
    }
