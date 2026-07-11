"""Utilities for estimating correlated flow-matching action noise."""

import numpy as np


def _validate_action_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError(f"Expected at least two flattened action chunks, got shape={rows.shape}.")
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
