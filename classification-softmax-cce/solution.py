import numpy as np


def softmax(Z: np.ndarray) -> np.ndarray:
    """Z: shape (n_samples, n_classes). Rows sum to 1."""
    # TODO: Subtract each row's max before exponentiating for numerical
    # stability, then normalize each row so it sums to 1.
    
    shifted = Z - np.max(Z ,axis = 1 , keepdims = True)
    exp_logits = np.exp(shifted)

    return exp_logits / np.sum(exp_logits , axis = 1 , keepdims = True)


def cce_loss(P: np.ndarray, y_indices: np.ndarray) -> float:
    """P: shape (n_samples, n_classes) from softmax. y_indices: integer class index per sample."""
    # TODO: Pick out each sample's predicted probability for its true
    # class, clip it away from 0, and return the mean negative log as
    # a plain Python float.
    n = P.shape[0]
    p_correct = np.clip(P[np.arange(n), y_indices], 1e-12, 1.0)
    return float(-np.mean(np.log(p_correct)))
