import numpy as np


def sum_(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Mirrors torch.sum(a, dim=axis, keepdim=keepdims)."""
    return np.sum(a , axis = axis , keepdims = keepdims)


def mean_(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Mirrors torch.mean(a, dim=axis, keepdim=keepdims)."""
    return np.mean(a , axis = axis , keepdims = keepdims)


def max_(a: np.ndarray, axis: int | None = None, keepdims: bool = False):
    """
    Mirrors torch.max(a, dim=axis, keepdim=keepdims):
      - axis=None: returns a single scalar, the max over the whole array.
      - axis given: returns (values, indices), the max value AND its
        index along that axis, both with the same shape (respecting
        keepdims). This differs from np.max, which only ever returns
        the values, never the indices.
    """
    # TODO: axis is None -> just np.max(a). Otherwise, np.max(...) for
    # values AND np.argmax(...) for indices, both with the same
    # axis/keepdims, returned as a (values, indices) tuple.
    if axis is None:
        return np.max(a)
    values = np.max(a, axis=axis, keepdims=keepdims)
    indices = np.argmax(a, axis=axis, keepdims=keepdims)
    return values, indices
