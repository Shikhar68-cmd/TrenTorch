import numpy as np


def bce_gradient(
    input: np.ndarray,
    p: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    p:      shape (batch_size, 1), sigmoid(linear(input, weight, bias))
    target: shape (batch_size, 1)

    Returns:
        grad_weight: shape (1, in_features)
        grad_bias: shape (1,)
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    error = p - target

    n_samples = input.shape[0]

    return (error.T @ input) / n_samples , np.sum(error , axis = 0)/n_samples
