import numpy as np


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gaussian similarity between the rows of X (shape (n, d)) and the
    rows of Y (shape (m, d)). Returns an array of shape (n, m) where
    entry [i, j] is exp(-gamma * ||X[i] - Y[j]||^2).

    Clip squared distances at zero before exponentiating, so rounding
    cannot push identical points above 1.
    """
    sq_x = np.sum(X *X , axis = 1 , keepdims = True)
    sq_y = np.sum(Y*Y , axis = 1, keepdims = True)

    sq_dist = np.maximum(sq_x + sq_y.T - 2.0 * (X @ Y.T) , 0.0)

    return np.exp(-gamma * sq_dist)
