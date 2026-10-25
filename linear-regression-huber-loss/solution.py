import numpy as np


def huber_loss(
    input: np.ndarray, target: np.ndarray, delta: float = 1.0, reduction: str = "mean"
) -> float | np.ndarray:
    """
    Mirrors torch.nn.functional.huber_loss(input, target, delta=delta,
    reduction=reduction): quadratic (MSE-like) for small errors,
    linear (L1-like) for large ones, splitting at |error| == delta.

        elementwise = 0.5 * error^2                    if |error| <= delta
                    = delta * (|error| - 0.5 * delta)   otherwise

    See Theory for why this specific split point makes the two
    branches meet smoothly (equal value AND equal slope at the seam).
    """
    error = input - target
    abs_error = np.abs(error)
    quadratic_branch = 0.5 * error**2
    linear_branch = delta * (abs_error - 0.5 * delta)
    elementwise = np.where(abs_error <= delta, quadratic_branch, linear_branch)

    if reduction == "mean":
        return float(np.mean(elementwise))
    elif reduction == "sum":
        return float(np.sum(elementwise))
    elif reduction == "none":
        return elementwise
    else:
        raise ValueError(f"Invalid reduction: {reduction!r}")
