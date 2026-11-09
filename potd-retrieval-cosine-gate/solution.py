

import numpy as np

def retrieval_cosine_gate(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query)
    d_norms = np.linalg.norm(documents, axis=1)
    denom = q_norm * d_norms

    dots = documents @ query
    out = np.zeros(documents.shape[0], dtype=np.result_type(dots, np.float64))
    np.divide(dots, denom, out=out, where=denom > 0)
    return out
