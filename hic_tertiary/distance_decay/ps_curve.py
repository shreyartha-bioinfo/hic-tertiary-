"""
Contact probability P(s) vs genomic distance analysis.

Biological question answered
-----------------------------
How does contact frequency decay with genomic distance?
The exponent α in P(s) ~ s^{-α} relates to polymer physics models of chromatin:
  • α ≈ 1.0  → fractal globule model  (Lieberman-Aiden et al. 2009)
  • α ≈ 0.5  → equilibrium globule
  • α ≈ 1.5  → extended or decompacted chromatin
"""
import numpy as np
from scipy.stats import linregress


# ── Core P(s) computation ─────────────────────────────────────────────────────

def compute_ps(
    matrix: np.ndarray,
    resolution: int = 50_000,
    min_dist_bins: int = 2,
    max_dist_bins: int = None,
    log_bins: int = 50,
) -> tuple:
    """
    Compute contact probability P(s) as a function of genomic distance s.

    Uses log-spaced distance bins so short and long distances are equally
    represented on a log-log plot.

    Parameters
    ----------
    matrix         : ndarray (n, n)  symmetric contact matrix
    resolution     : int             bin size in bp
    min_dist_bins  : int             minimum diagonal offset to include
    max_dist_bins  : int             maximum diagonal offset (None = n//2)
    log_bins       : int             number of log-spaced distance bins

    Returns
    -------
    s       : ndarray   genomic distances (bp)
    ps      : ndarray   mean contact probability at each distance
    counts  : ndarray   number of valid diagonal entries per bin
    """
    n = matrix.shape[0]
    max_d = max_dist_bins or (n // 2)
    max_d = min(max_d, n - 1)

    # Collect per-diagonal statistics
    raw_s = []
    raw_ps = []
    for d in range(min_dist_bins, max_d + 1):
        diag = np.diag(matrix, k=d)
        raw_s.append(d * resolution)
        raw_ps.append(float(diag.mean()))

    raw_s = np.array(raw_s, dtype=float)
    raw_ps = np.array(raw_ps, dtype=float)

    # Bin into log-spaced distance bins
    edges = np.logspace(
        np.log10(raw_s.min()),
        np.log10(raw_s.max()),
        log_bins + 1,
    )
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    bin_ps = np.zeros(log_bins)
    bin_counts = np.zeros(log_bins, dtype=int)

    for k in range(log_bins):
        sel = (raw_s >= edges[k]) & (raw_s < edges[k + 1])
        if sel.any():
            bin_ps[k] = raw_ps[sel].mean()
            bin_counts[k] = sel.sum()

    # Remove empty bins
    valid = bin_counts > 0
    return bin_centers[valid], bin_ps[valid], bin_counts[valid]


# ── Power-law fit ─────────────────────────────────────────────────────────────

def fit_power_law(s: np.ndarray, ps: np.ndarray) -> dict:
    """
    Fit P(s) = A * s^{-α} by linear regression in log-log space.

    Returns
    -------
    dict with 'alpha', 'A', 'r_squared', 'fit_ps' (predicted values at input s)
    """
    valid = (s > 0) & (ps > 0)
    log_s = np.log10(s[valid])
    log_ps = np.log10(ps[valid])

    slope, intercept, r, _, _ = linregress(log_s, log_ps)

    alpha = -slope
    A = 10 ** intercept
    fit_ps = A * s ** (-alpha)

    return dict(alpha=alpha, A=A, r_squared=r ** 2, fit_ps=fit_ps)


# ── Slope derivative ──────────────────────────────────────────────────────────

def ps_derivative(s: np.ndarray, ps: np.ndarray) -> tuple:
    """
    Compute the local log-log slope d(log P)/d(log s).

    The slope transitions between regimes (loop extrusion, compartments) are
    visible as features in this curve.
    """
    log_s = np.log10(s)
    log_ps = np.log10(np.clip(ps, 1e-30, None))
    slope = np.gradient(log_ps, log_s)
    return s, slope


# ── Multi-chromosome comparison ───────────────────────────────────────────────

def ps_all_chromosomes(
    matrices: dict,
    chr_names: list,
    resolution: int = 50_000,
    **kwargs,
) -> dict:
    """
    Compute P(s) curves for every chromosome.

    Returns
    -------
    dict {chrom: {'s': ..., 'ps': ..., 'counts': ..., 'fit': ...}}
    """
    results = {}
    for chrom in chr_names:
        s, ps, counts = compute_ps(matrices[chrom], resolution, **kwargs)
        fit = fit_power_law(s, ps)
        results[chrom] = dict(s=s, ps=ps, counts=counts, fit=fit)
    return results
