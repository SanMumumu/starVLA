"""Focused checks for correlated action-noise statistics."""

import numpy as np

from starVLA.dataloader.action_correlation import compute_action_correlation, compute_action_noise_matrix


def test_action_correlation_is_scale_invariant_and_handles_constants():
    x = np.arange(1, 9, dtype=np.float64)
    rows = np.stack([x, 10.0 * x + 7.0, np.ones_like(x)], axis=1)

    correlation = compute_action_correlation(rows)
    expected = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(correlation, expected, atol=1e-7)
    np.testing.assert_allclose(np.diag(correlation), np.ones(3), atol=1e-7)


def test_action_noise_matrix_defaults_to_legacy_covariance():
    rows = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])

    matrix = compute_action_noise_matrix(rows)

    np.testing.assert_allclose(matrix, np.cov(rows, rowvar=False))
    assert not np.allclose(np.diag(matrix), np.ones(2))


if __name__ == "__main__":
    test_action_correlation_is_scale_invariant_and_handles_constants()
    test_action_noise_matrix_defaults_to_legacy_covariance()
