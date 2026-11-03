import numpy as np


def temperature_calibrator(z: np.ndarray, tau: float) -> np.ndarray:
    """
    z: logits.
    tau: temperature, tau > 0.

    p = softmax(z / tau), using an overflow-safe softmax (subtract the
    max before exponentiating). Divide by tau, never multiply.

    Return p.
    """
    
    if tau <= 0:
        raise ValueError()

    scaled = z / tau

    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
