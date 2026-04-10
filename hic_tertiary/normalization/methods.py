"""
Normalization methods for Hi-C contact matrices.

Implements from scratch (numpy/scipy only):
  • ICE  — Iterative Correction and Eigenvector decomposition
  • KR   — Knight-Ruiz matrix balancing
  • VC   — Vanilla Coverage (sqrt-normalization)
"""
import numpy as np
import warnings


# ── ICE Normalization ──────────────────────────────────────────────────────────

def ice_normalize(
    matrix: np.ndarray,
    max_iter: int = 200,
    tolerance: float = 1e-5,
    ignore_diag: int = 1,
) -> tuple:
    """
    ICE normalization (Imakaev et al. 2012).

    Iteratively scales rows and columns so every bin has equal marginal sums,
    making the matrix doubly stochastic.

    Parameters
    ----------
    matrix      : ndarray (n, n)  raw symmetric contact matrix
    max_iter    : int             maximum iterations
    tolerance   : float           convergence criterion on marginals
    ignore_diag : int             number of diagonals to set zero before normalization

    Returns
    -------
    balanced : ndarray (n, n)   ICE-balanced matrix
    bias     : ndarray (n,)     multiplicative bias vector (1/sqrt(scale per bin))
    """
    m = matrix.astype(float).copy()
    n = m.shape[0]

    # Zero out near-diagonals (self-ligation artefacts)
    for d in range(ignore_diag):
        i = np.arange(n - d)
        m[i, i + d] = 0
        m[i + d, i] = 0

    # Mask bins with zero coverage
    row_sums = m.sum(axis=1)
    valid = row_sums > 0
    bias = np.ones(n)

    for iteration in range(max_iter):
        marg = m.sum(axis=1)
        marg[marg == 0] = 1.0   # avoid div by zero for masked bins

        # Row normalisation
        m = m / marg[:, np.newaxis]
        # Column normalisation (transpose of same operation since matrix is symmetric)
        m = m / marg[np.newaxis, :]

        # Track bias
        bias *= marg

        # Check convergence: marginals should be close to 1 for valid bins
        new_marg = m.sum(axis=1)
        delta = np.abs(new_marg[valid] - new_marg[valid].mean())
        if delta.max() < tolerance:
            break

    return m, bias


# ── KR Normalization ───────────────────────────────────────────────────────────

def kr_normalize(
    matrix: np.ndarray,
    max_iter: int = 300,
    tolerance: float = 1e-6,
) -> tuple:
    """
    Knight-Ruiz (KR) balancing — finds a diagonal scaling vector x such that
    diag(x) @ M @ diag(x) is doubly stochastic.

    Uses the fixed-point Newton iteration described in Knight & Ruiz (2012).

    Returns
    -------
    balanced : ndarray (n, n)
    bias     : ndarray (n,)   balancing vector x (multiply rows AND columns)
    """
    m = matrix.astype(float).copy()
    n = m.shape[0]

    # Mask zero-coverage bins
    row_sums = m.sum(axis=1)
    valid = row_sums > 0

    x = np.ones(n, dtype=float)
    x[~valid] = 0.0

    target = np.ones(n, dtype=float)
    target[~valid] = 0.0

    for _ in range(max_iter):
        # Compute current marginals
        Ax = m @ x          # = M x  (one-sided product because M is symmetric)
        Ax[~valid] = 1.0    # avoid div-by-zero

        # Newton step: x <- x * (target / Ax) element-wise
        x_new = x * (target / Ax)
        x_new[~valid] = 0.0

        if np.linalg.norm(x_new - x) < tolerance:
            x = x_new
            break
        x = x_new

    balanced = x[:, np.newaxis] * m * x[np.newaxis, :]
    return balanced, x


# ── VC (Vanilla Coverage) Normalization ───────────────────────────────────────

def vc_normalize(matrix: np.ndarray, sqrt: bool = True) -> tuple:
    """
    Vanilla Coverage normalization.

    Divides each entry (i,j) by sqrt(coverage_i * coverage_j)  (if sqrt=True)
    or by (coverage_i * coverage_j) otherwise.

    Returns
    -------
    balanced : ndarray (n, n)
    bias     : ndarray (n,)   per-bin coverage
    """
    cov = matrix.sum(axis=1).astype(float)
    cov[cov == 0] = 1.0
    if sqrt:
        scale = np.sqrt(cov)
    else:
        scale = cov
    balanced = matrix / (scale[:, np.newaxis] * scale[np.newaxis, :])
    return balanced, cov


# ── Convenience dispatcher ────────────────────────────────────────────────────

def normalize(matrix: np.ndarray, method: str = "ice", **kwargs) -> tuple:
    """
    Normalise a contact matrix.

    Parameters
    ----------
    method : str    "ice" | "kr" | "vc"

    Returns
    -------
    (balanced_matrix, bias_vector)
    """
    method = method.lower()
    if method == "ice":
        return ice_normalize(matrix, **kwargs)
    elif method == "kr":
        return kr_normalize(matrix, **kwargs)
    elif method == "vc":
        return vc_normalize(matrix, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. "
                         f"Choose from 'ice', 'kr', 'vc'.")
