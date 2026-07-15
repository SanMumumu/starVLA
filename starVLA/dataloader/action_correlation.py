"""Utilities for estimating correlated flow-matching action noise."""

import numpy as np


def validate_action_correlation_cholesky(
    cholesky: np.ndarray,
    *,
    expected_size: int | None = None,
    triangular_atol: float = 1.0e-6,
) -> np.ndarray:
    """Validate and return a float32 action-noise Cholesky factor.

    Correlated-noise checkpoints depend on this matrix at both training and
    inference time.  Silently accepting a malformed/stale array (or falling
    back to iid noise) changes the model's sampling distribution, so keep the
    artifact contract deliberately strict.
    """

    matrix = np.asarray(cholesky)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square Cholesky matrix, got shape={matrix.shape}.")
    if expected_size is not None and matrix.shape != (int(expected_size), int(expected_size)):
        raise ValueError(
            f"Cholesky shape={matrix.shape}, expected ({int(expected_size)}, {int(expected_size)})."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"Cholesky matrix must be numeric, got dtype={matrix.dtype}.")
    matrix = matrix.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Cholesky matrix contains NaN or infinite values.")
    if not np.allclose(matrix, np.tril(matrix), rtol=0.0, atol=float(triangular_atol)):
        # The matrix is guaranteed non-empty above.  Avoid ndarray.max(initial=...)
        # so this validation also works with the older NumPy shipped by some
        # cluster images.
        upper_max = float(np.abs(np.triu(matrix, k=1)).max())
        raise ValueError(f"Cholesky matrix is not lower triangular (max upper value={upper_max:g}).")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError(
            "Cholesky matrix must have a strictly positive diagonal; "
            f"minimum diagonal={float(diagonal.min()):g}."
        )
    return matrix.astype(np.float32, copy=False)


def _validate_action_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError(f"Expected at least two flattened action chunks, got shape={rows.shape}.")
    if not np.isfinite(rows).all():
        raise ValueError("Flattened action chunks contain NaN or infinite values.")
    return rows


def _sanitize_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(matrix).astype(np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (matrix + matrix.T)


def compute_action_covariance(rows: np.ndarray) -> np.ndarray:
    """Estimate covariance, preserving the repository's legacy behavior."""
    rows = _validate_action_rows(rows)
    return _sanitize_symmetric(np.cov(rows, rowvar=False))


def compute_action_correlation(rows: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate a scale-invariant correlation matrix over flattened chunks."""
    rows = _validate_action_rows(rows)
    std = rows.std(axis=0)
    nonconstant = std >= eps
    normalized = np.zeros_like(rows)
    normalized[:, nonconstant] = (rows[:, nonconstant] - rows[:, nonconstant].mean(axis=0)) / std[nonconstant]
    corr = _sanitize_symmetric(np.cov(normalized, rowvar=False))
    diagonal = np.diag(corr).copy()
    safe_diagonal = np.where(diagonal > eps, diagonal, 1.0)
    corr = corr / np.sqrt(np.outer(safe_diagonal, safe_diagonal))
    constant = ~nonconstant
    corr[constant, :] = 0.0
    corr[:, constant] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_action_noise_matrix(rows: np.ndarray, matrix_type: str = "covariance") -> np.ndarray:
    """Select legacy covariance or opt-in scale-invariant correlation."""
    matrix_type = str(matrix_type).lower()
    if matrix_type == "covariance":
        return compute_action_covariance(rows)
    if matrix_type == "correlation":
        return compute_action_correlation(rows)
    raise ValueError(f"Unsupported action-noise matrix_type={matrix_type!r}; expected 'covariance' or 'correlation'.")
