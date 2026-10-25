import numpy as np


def l1_loss(input: np.ndarray, target: np.ndarray, reduction: str = "mean") -> float | np.ndarray:
    """
    input, target: same shape, any shape.
    reduction: "mean" (default), "sum", or "none".

    Mirrors torch.nn.functional.l1_loss: mean absolute error, not
    squared error. See Theory for why this changes MSE Loss's
    behavior around outliers.
    """
    absolute_error = np.abs(input - target)
    if reduction == "mean":
        return float(np.mean(absolute_error))
    elif reduction == "sum":
        return float(np.sum(absolute_error))
    elif reduction == "none":
        return absolute_error
    else:
        raise ValueError(f"Invalid reduction: {reduction!r}")
