import numpy as np


def ridge_regression_closed_form(
    input: np.ndarray, target: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    `Stretch: L2 Regularization (Ridge)` derived ridge's GRADIENT for
    use with gradient descent. Ridge regression also has a closed
    form, a small modification of `Linear Regression: closed form
    (Normal Equation)`'s own formula:

        theta = (X^T @ X + alpha * I) @ X^T @ y

    where `I` is the identity matrix EXCEPT with a `0` in the last
    diagonal entry, corresponding to the bias term, which is
    conventionally never regularized (see Theory for why).
    """

    n ,d = input.shape
    input_aug = np.hstack([input , np.ones((n , 1))])
    target_col = target.reshape(-1,1)

    l2_penalty = alpha * np.eye(d + 1)

    l2_penalty[-1,-1] = 0.0
    
    theta = np.linalg.inv(input_aug.T @ input_aug + l2_penalty) @ input_aug.T @target_col

    weight = theta[:-1].T
    bias = theta[-1]
    return weight, bias
