import numpy as np

def solve(scores, V):
    """Implement attention-weighted-sum from the mathematical contract in README.md."""
    s = np.asarray(scores , float)

    v = np.asarray(V , float)

    #Softmax

    p = np.exp(s - s.max())

    p = p/p.sum()

    return p @ v
