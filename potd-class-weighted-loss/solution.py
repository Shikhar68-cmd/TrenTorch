import numpy as np


def class_weighted_loss(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    """
    logits: n x C logits.
    labels: length-n 0-based true class indices.
    weights: length-C per-class weights.

    p_ic = softmax(logits_i)_c
    L = -(1/n) * sum_i weights[labels[i]] * log(p_i, labels[i])

    Divide by n, NOT by sum(weights[labels]). Return L.
    """
    # TODO: stable log-softmax per row, weight by true-class weight,
    # sum, divide by n.
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    # Extract log probability of each true class
    true_log_probs = log_probs[np.arange(len(labels)), labels]

    # Get corresponding class weights
    sample_weights = weights[labels]

    # Weighted cross entropy
    loss = -np.sum(sample_weights * true_log_probs) / len(labels)

    return float(loss)
