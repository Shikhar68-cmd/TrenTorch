import numpy as np

def solve(logits):
    """Implement stable-softmax-logits from the mathematical contract in README.md."""
    x=np.asarray(logits,dtype=float)
    z=x-np.max(x)
    e=np.exp(z)
    return e/e.sum()
