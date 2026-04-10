"""
TAD (Topologically Associating Domain) Analysis.

Biological questions answered
-------------------------------
• Where are TAD boundaries in the genome?
• How large are TADs?
• Which loci are in the same TAD (likely co-regulated)?
• How do TAD boundaries correlate with CTCF binding / insulators?
• Are boundaries conserved between cell types?

Method: Diamond Insulation Score (Crane et al. 2015)
------------------------------------------------------
For each bin i, sum all contacts within a diamond-shaped window
straddling the diagonal. A local minimum in the insulation score
indicates a TAD boundary (contacts cross the boundary less often).
"""
import numpy as np
from scipy.signal import find_peaks


# ── Insulation score ──────────────────────────────────────────────────────────

def insulation_score(
    matrix: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute the Diamond Insulation Score (Crane et al. 2015).

    Parameters
    ----------
    matrix : ndarray (n, n)   balanced, symmetric contact matrix
    window : int              half-size of the diamond window in bins

    Returns
    -------
    score : ndarray (n,)
        Log2-transformed insulation score, normalised to zero mean.
        Minima correspond to TAD boundaries.
    """
    n = matrix.shape[0]
    raw = np.zeros(n)

    for i in range(window, n - window):
        # Diamond: contacts between [i-w, i) and [i, i+w]
        block = matrix[i - window:i, i:i + window]
        raw[i] = block.mean() if block.size > 0 else 0.0

    # Avoid log of zero
    mean_val = raw[raw > 0].mean() if (raw > 0).any() else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.log2(raw / mean_val)
    score[~np.isfinite(score)] = 0.0

    return score


# ── Boundary detection ────────────────────────────────────────────────────────

def find_boundaries(
    score: np.ndarray,
    delta_threshold: float = 0.1,
    min_distance: int = 3,
) -> np.ndarray:
    """
    Detect TAD boundary positions as local minima of the insulation score.

    Parameters
    ----------
    score           : ndarray (n,)   insulation score
    delta_threshold : float          minimum drop relative to flanking max
    min_distance    : int            minimum bins between boundaries

    Returns
    -------
    boundaries : ndarray (k,)  bin indices of detected boundaries
    """
    # find_peaks on inverted score → local minima
    inverted = -score
    peaks, properties = find_peaks(
        inverted,
        prominence=delta_threshold,
        distance=min_distance,
    )
    return peaks


# ── TAD calling from boundaries ───────────────────────────────────────────────

def call_tads(boundaries: np.ndarray, n_bins: int) -> list:
    """
    Convert boundary positions to TAD intervals [start, end).

    Parameters
    ----------
    boundaries : ndarray   sorted boundary bin indices
    n_bins     : int       total number of bins

    Returns
    -------
    tads : list of (start_bin, end_bin) tuples
    """
    all_bounds = np.concatenate([[0], np.sort(boundaries), [n_bins]])
    tads = []
    for k in range(len(all_bounds) - 1):
        s, e = int(all_bounds[k]), int(all_bounds[k + 1])
        if e - s >= 1:
            tads.append((s, e))
    return tads


# ── TAD size distribution ─────────────────────────────────────────────────────

def tad_sizes(tads: list, resolution: int = 50_000) -> np.ndarray:
    """Return TAD sizes in bp."""
    return np.array([(e - s) * resolution for s, e in tads], dtype=float)


# ── Boundary strength (delta score) ──────────────────────────────────────────

def boundary_strength(score: np.ndarray, boundaries: np.ndarray, flank: int = 3) -> np.ndarray:
    """
    Quantify how pronounced each boundary is.

    Strength = mean(score in flanking regions) − score_at_boundary
    """
    strengths = []
    n = len(score)
    for b in boundaries:
        left = score[max(0, b - flank):b]
        right = score[b + 1:min(n, b + flank + 1)]
        flank_mean = np.concatenate([left, right]).mean() if len(left) + len(right) > 0 else 0.0
        strengths.append(flank_mean - score[b])
    return np.array(strengths, dtype=float)


# ── Full TAD analysis ─────────────────────────────────────────────────────────

def tad_analysis(
    matrix: np.ndarray,
    resolution: int = 50_000,
    window: int = 10,
    delta_threshold: float = 0.1,
    min_distance: int = 3,
) -> dict:
    """
    End-to-end TAD analysis for one chromosome.

    Returns
    -------
    dict with:
        insulation_score : ndarray (n,)
        boundaries       : ndarray (k,)  boundary bin indices
        tads             : list[(start, end)]
        tad_sizes_bp     : ndarray (k+1,)
        boundary_strengths: ndarray (k,)
    """
    score = insulation_score(matrix, window)
    bounds = find_boundaries(score, delta_threshold, min_distance)
    tads = call_tads(bounds, matrix.shape[0])
    sizes = tad_sizes(tads, resolution)
    strengths = boundary_strength(score, bounds)

    return dict(
        insulation_score=score,
        boundaries=bounds,
        tads=tads,
        tad_sizes_bp=sizes,
        boundary_strengths=strengths,
    )


def tad_analysis_all_chromosomes(
    matrices: dict,
    chr_names: list,
    resolution: int = 50_000,
    **kwargs,
) -> dict:
    """Run TAD analysis on every chromosome."""
    return {
        chrom: tad_analysis(matrices[chrom], resolution, **kwargs)
        for chrom in chr_names
    }
