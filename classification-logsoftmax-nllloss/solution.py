import numpy as np


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """
    log(softmax(Z)), computed directly rather than as np.log(softmax(Z)):
    same math, but avoids ever computing log(a number very close to 0),
    which the naive two-step version does whenever one class's
    probability collapses toward zero.

        log_softmax(Z) = (Z - max(Z)) - log(sum(exp(Z - max(Z))))

    (the same max-shift 06-softmax-cce's own softmax uses, applied one
    level deeper into the log-space computation)
    """
    shifted = Z - np.max(Z , axis = 1 , keepdims = True)
    return shifted - np.log(np.sum(np.exp(shifted) , axis = 1 , keepdims = True))


def nll_loss(log_probs: np.ndarray, y_indices: np.ndarray) -> float:
    """
    Negative log-likelihood: given already-log-space class
    probabilities and integer class indices, index out each row's
    true-class log-probability and average their negatives.

        nll_loss(log_probs, y) = -mean(log_probs[i, y[i]] for each row i)
    """
    n = log_probs.shape[0]
    return float(-np.mean(log_probs[np.arange(n), y_indices]))
