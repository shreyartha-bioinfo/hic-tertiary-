"""
Shared matrix operations used across all analysis modules.
"""
import numpy as np
from scipy.ndimage import uniform_filter


# ── Diagonal utilities ─────────────────────────────────────────────────────────

def diag_mean(matrix: np.ndarray) -> np.ndarray:
    """
    Return mean value of each diagonal (stratum) of a square matrix.

    Returns
    -------
    means : ndarray, shape (n,)
        means[d] = mean of matrix diagonal at offset d (d=0 is main diagonal).
    """
    n = matrix.shape[0]
    means = np.zeros(n)
    for d in range(n):
        diag = np.diag(matrix, k=d)
        valid = diag[diag > 0]
        means[d] = valid.mean() if len(valid) > 0 else 0.0
    return means


def expected_from_diagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Build an expected-contact matrix where each entry (i,j) equals the
    average contact observed at genomic distance |i-j|.

    Returns
    -------
    expected : ndarray, shape (n, n)
    """
    n = matrix.shape[0]
    means = diag_mean(matrix)
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    dist = np.abs(i_idx - j_idx)
    return means[dist]


def obs_exp(matrix: np.ndarray, pseudocount: float = 1e-10) -> np.ndarray:
    """
    Observed / Expected contact matrix.

    Each entry is divided by the expected contact at that genomic distance.
    """
    exp = expected_from_diagonal(matrix)
    return matrix / (exp + pseudocount)


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_matrix(matrix: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian-equivalent uniform filter for smoothing."""
    size = max(1, int(sigma * 2 + 1))
    return uniform_filter(matrix.astype(float), size=size)


# ── Coverage / masking ────────────────────────────────────────────────────────

def coverage(matrix: np.ndarray) -> np.ndarray:
    """Row sums = total contacts per bin (sequencing coverage proxy)."""
    return matrix.sum(axis=1)


def low_coverage_mask(matrix: np.ndarray, quantile: float = 0.02) -> np.ndarray:
    """
    Boolean mask: True for bins with very low coverage (likely artefacts).
    Bins below the `quantile` coverage threshold are flagged.
    """
    cov = coverage(matrix)
    threshold = np.quantile(cov[cov > 0], quantile)
    return cov < threshold


def remove_low_coverage(matrix: np.ndarray,
                         quantile: float = 0.02) -> tuple:
    """
    Zero-out rows/columns with very low coverage.

    Returns
    -------
    (matrix_filtered, mask) where mask is True for removed bins.
    """
    mask = low_coverage_mask(matrix, quantile)
    m = matrix.copy()
    m[mask, :] = 0
    m[:, mask] = 0
    return m, mask


# ── Bin utilities ─────────────────────────────────────────────────────────────

def bin_genomic_distance(distances: np.ndarray,
                          n_bins: int = 50) -> tuple:
    """
    Bin distances on a log scale.

    Returns
    -------
    (bin_centers, bin_edges)
    """
    valid = distances[distances > 0]
    edges = np.logspace(np.log10(valid.min()), np.log10(valid.max()), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, edges


# ── Symmetrization ────────────────────────────────────────────────────────────

def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Force matrix to be symmetric by averaging with its transpose."""
    return (matrix + matrix.T) / 2.0


# ── Sub-matrix extraction ─────────────────────────────────────────────────────

def extract_square(matrix: np.ndarray,
                   center_i: int,
                   center_j: int,
                   flank: int) -> np.ndarray:
    """
    Extract a (2*flank+1) x (2*flank+1) sub-matrix centred on (center_i, center_j).
    Regions outside the matrix are filled with NaN.
    """
    n = matrix.shape[0]
    size = 2 * flank + 1
    result = np.full((size, size), np.nan)
    for di in range(-flank, flank + 1):
        for dj in range(-flank, flank + 1):
            ii = center_i + di
            jj = center_j + dj
            if 0 <= ii < n and 0 <= jj < n:
                result[di + flank, dj + flank] = matrix[ii, jj]
    return result


# ── Correlation matrix ────────────────────────────────────────────────────────

def pearson_correlation(matrix: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation; NaN entries replaced with 0."""
    corr = np.corrcoef(matrix)
    return np.nan_to_num(corr, nan=0.0)
