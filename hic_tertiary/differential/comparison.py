"""
Differential Hi-C Analysis between two conditions.

Biological questions answered
-------------------------------
• Which genomic regions show significantly different contact frequencies?
• Which loci switch between A and B compartments?
• Are TAD boundaries gained or lost?
• What is the overall similarity of two contact maps? (SCC)
• Where are condition-specific loops?

Methods
--------
1. Log2 fold-change contact maps
2. Stratum-adjusted Correlation Coefficient (SCC)  — Yan et al. 2017
3. Differential compartment eigenvectors
4. Differential insulation score (boundary changes)
"""
import numpy as np
from scipy.stats import pearsonr


# ── Log fold-change ────────────────────────────────────────────────────────────

def log_fold_change(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    pseudocount: float = 1.0,
    normalise_depth: bool = True,
) -> np.ndarray:
    """
    Compute log2 fold-change:  log2( B / A )

    Parameters
    ----------
    matrix_a, matrix_b : ndarray (n, n)   contact matrices
    pseudocount        : float            added to avoid log(0)
    normalise_depth    : bool             scale each matrix to unit total before LFC

    Returns
    -------
    lfc : ndarray (n, n)   log2 fold-change (positive → enriched in B)
    """
    a = matrix_a.astype(float) + pseudocount
    b = matrix_b.astype(float) + pseudocount

    if normalise_depth:
        a = a / a.sum() * 1e6
        b = b / b.sum() * 1e6

    return np.log2(b / a)


# ── Stratum-adjusted Correlation Coefficient (SCC) ───────────────────────────

def scc(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    max_dist_bins: int = 100,
) -> dict:
    """
    Stratum-adjusted Correlation Coefficient (Yang et al. 2017 / HiCRep).

    Computes Pearson r between two Hi-C maps at each genomic distance (stratum)
    and returns a weighted average, making the metric robust to the strong
    distance-decay confound.

    Parameters
    ----------
    matrix_a, matrix_b : ndarray (n, n)
    max_dist_bins      : int   maximum diagonal to include

    Returns
    -------
    dict with 'scc', 'per_stratum_r', 'per_stratum_n', 'weights'
    """
    n = matrix_a.shape[0]
    max_d = min(max_dist_bins, n - 1)

    per_r = []
    per_n = []

    for d in range(1, max_d + 1):
        da = np.diag(matrix_a, k=d).astype(float)
        db = np.diag(matrix_b, k=d).astype(float)
        valid = (da > 0) | (db > 0)
        if valid.sum() < 3:
            per_r.append(np.nan)
            per_n.append(0)
            continue
        r, _ = pearsonr(da[valid], db[valid])
        per_r.append(float(r) if np.isfinite(r) else 0.0)
        per_n.append(int(valid.sum()))

    per_r = np.array(per_r, dtype=float)
    per_n = np.array(per_n, dtype=float)

    # Weight by number of valid contacts at each stratum
    weights = per_n / per_n.sum() if per_n.sum() > 0 else per_n
    scc_score = float(np.nansum(per_r * weights))

    return dict(
        scc=scc_score,
        per_stratum_r=per_r,
        per_stratum_n=per_n,
        weights=weights,
    )


# ── Differential compartments ─────────────────────────────────────────────────

def differential_compartments(
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    threshold: float = 0.2,
) -> dict:
    """
    Identify bins that switch compartment between conditions.

    Parameters
    ----------
    comp_a, comp_b : ndarray (n,)   compartment eigenvectors (PC1)
    threshold      : float          minimum |delta| to call a switch

    Returns
    -------
    dict with:
        delta          : ndarray    comp_b − comp_a
        switched       : ndarray    boolean mask
        a_to_b         : ndarray    bins going A→B
        b_to_a         : ndarray    bins going B→A
    """
    delta = comp_b - comp_a
    sign_a = np.sign(comp_a)
    sign_b = np.sign(comp_b)

    switched = (sign_a != sign_b) & (np.abs(delta) > threshold)
    a_to_b = switched & (sign_a > 0) & (sign_b <= 0)
    b_to_a = switched & (sign_a <= 0) & (sign_b > 0)

    return dict(
        delta=delta,
        switched=switched,
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        n_switched=int(switched.sum()),
        n_a_to_b=int(a_to_b.sum()),
        n_b_to_a=int(b_to_a.sum()),
    )


# ── Differential insulation (boundary changes) ────────────────────────────────

def differential_insulation(
    score_a: np.ndarray,
    score_b: np.ndarray,
    threshold: float = 0.3,
) -> dict:
    """
    Compare insulation scores to identify gained / lost TAD boundaries.

    Returns
    -------
    dict with 'delta', 'gained', 'lost' boundary masks
    """
    delta = score_b - score_a

    # Gained boundaries in B: delta strongly negative (new insulation valley)
    gained = delta < -threshold
    # Lost boundaries in B: delta strongly positive (insulation valley filled)
    lost = delta > threshold

    return dict(delta=delta, gained=gained, lost=lost)


# ── Full differential analysis ────────────────────────────────────────────────

def differential_analysis(
    data_a: dict,
    data_b: dict,
    comp_results_a: dict,
    comp_results_b: dict,
    tad_results_a: dict,
    tad_results_b: dict,
    pseudocount: float = 1.0,
    scc_max_dist: int = 100,
) -> dict:
    """
    Run all differential analyses between two Hi-C conditions.

    Returns
    -------
    dict keyed by chromosome, each with:
        lfc        : ndarray   log2 fold-change matrix
        scc_result : dict      SCC metrics
        comp_diff  : dict      compartment switch calls
        ins_diff   : dict      differential insulation
    """
    results = {}
    chr_names = data_a["chr_names"]

    for chrom in chr_names:
        mat_a = data_a["matrices"][chrom]
        mat_b = data_b["matrices"][chrom]

        lfc = log_fold_change(mat_a, mat_b, pseudocount)
        scc_res = scc(mat_a, mat_b, scc_max_dist)

        comp_diff = differential_compartments(
            comp_results_a[chrom]["compartment"],
            comp_results_b[chrom]["compartment"],
        )
        ins_diff = differential_insulation(
            tad_results_a[chrom]["insulation_score"],
            tad_results_b[chrom]["insulation_score"],
        )

        results[chrom] = dict(
            lfc=lfc,
            scc_result=scc_res,
            comp_diff=comp_diff,
            ins_diff=ins_diff,
        )

    return results
