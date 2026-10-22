import numpy as np


def skewness(x: np.ndarray) -> float:
    
    skew = 0.0
    for num in x:
        skew += ((num - np.mean(x)) / np.std(x)) ** 3

    return skew / len(x)


def summarize_distribution(x: np.ndarray) -> dict:
    
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "skew": float(skewness(x)),
    }
