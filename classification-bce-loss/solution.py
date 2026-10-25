import numpy as np


def bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    """
    p: predicted probabilities, shape (n_samples,)
    y: true labels (0 or 1), shape (n_samples,)
    Returns a single scalar.
    """
    # TODO: Implement Binary Cross-Entropy, as derived in Theory.
    # Clip p away from exactly 0 or 1 before taking a log, and return
    # a plain Python float, not a NumPy scalar.
    p = np.clip(p, 1e-15, 1 - 1e-15)
    loss = 0.0
    m = len(p)
    for real , pred in zip( y , p):
        loss = loss + ((real * np.log(pred)) + ((1 - real)* (np.log( 1 - pred)))) 

    loss = -(1* loss) / m

    return float(loss)
