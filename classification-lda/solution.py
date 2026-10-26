import numpy as np


def lda_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Returns (w, b) for the linear discriminant boundary, computed from
    per-class means, one shared (pooled) covariance matrix, and class priors.
    """
    # TODO: Split X by class, compute each class's mean and the pooled
    # (shared) covariance, then derive w and b from Theory's formulas.
    X0 , X1 = X[y == 0] , X[y == 1]
    mu0 , mu1 = np.mean(X0 , axis=0) , np.mean(X1 , axis = 0)
    
    n0 , n1 = len(X0) , len(X1)
    n = n0 + n1
    cov0 , cov1 = (X0 - mu0).T @ (X0 - mu0) , (X1 - mu1).T @ (X1 - mu1)

    pooled_cov = (cov0 + cov1) / (n - 2)

    pooled_cov_inv = np.linalg.inv(pooled_cov)

    w = pooled_cov_inv @ (mu1 - mu0)

    p0 , p1 = n0/n , n1/n

    b = (-0.5 * (mu1 @ pooled_cov_inv @ mu1)
         + 0.5 * (mu0 @ pooled_cov_inv @ mu0)
         + np.log(p1 / p0))

    return w , b
