import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    z: any shape of real numbers.
    Returns elementwise sigmoid, same shape as z, values strictly in (0, 1).
    """
    # TODO: Implement the sigmoid function from Theory.
    # Clip z before exponentiating so large |z| never overflows or
    # returns nan.
    
    z = np.clip(z, -500, 500)

    sig_Z = np.zeros_like(z, dtype=float)

    for i in range(len(z)):
        sig_Z[i] = 1 / (1 + np.exp(-z[i]))

    return sig_Z
