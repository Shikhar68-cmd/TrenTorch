import numpy as np


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return sum(x * y for x, y in zip(a, b))


def l1_norm(x: np.ndarray) -> float:
    l1 = 0.0
    for num in x:
        l1 += abs(num)          
    return l1


def l2_norm(x: np.ndarray) -> float:
    l2 = 0.0
    for num in x:
        l2 += num ** 2          
    return np.sqrt(l2)          


def linf_norm(x: np.ndarray) -> float:
    return max(abs(num) for num in x)
