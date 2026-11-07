"""Validate embeddings before cosine retrieval."""
import numpy as np


def checked_normalize(values, name):
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name}: expected non-empty 2D embeddings")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite embeddings")
    # float64 avoids overflow when squaring large finite float32 values.
    norm = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm == 0).any():
        raise ValueError(f"{name}: zero or non-finite embedding norm")
    return values / norm
