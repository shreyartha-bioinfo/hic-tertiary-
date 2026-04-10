"""
Chromatin Loop Analysis.

Biological questions answered
-------------------------------
• Which pairs of genomic loci form stable loops?
• What is the enrichment of contacts at loop anchors over background?
• Are loops preferentially at TAD boundaries (CTCF–CTCF loops)?
• How are loop sizes distributed?

Methods
--------
1. Local enrichment detection: compare each pixel to local background
2. APA (Aggregate Peak Analysis): average O/E contact at putative loop positions
"""
import numpy as np
from scipy.ndimage import maximum_filter

from hic_tertiary.utils.matrix_ops import obs_exp, extract_square


# ── Loop detection ────────────────────────────────────────────────────────────

def detect_loops(
    matrix: np.ndarray,
    min_dist_bins: int = 5,
    max_dist_bins: int = 150,
    enrichment_threshold: float = 2.0,
    local_window: int = 5,
) -> list:
    """
    Detect putative chromatin loops as locally enriched pixels in the O/E matrix.

    Algorithm
    ---------
    1. Compute O/E matrix (correct for distance decay)
    2. Apply maximum filter to find local maxima
    3. Keep pixels that are:
       - Their own local maximum
       - O/E > enrichment_threshold
       - Distance between min_dist_bins and max_dist_bins

    Parameters
    ----------
    matrix              : ndarray (n, n)
    min_dist_bins       : int    minimum loop span in bins
    max_dist_bins       : int    maximum loop span in bins
    enrichment_threshold: float  minimum O/E to call a loop
    local_window        : int    neighbourhood size for local max test

    Returns
    -------
    loops : list of dict, each with keys 'i', 'j', 'enrichment'
    """
    oe = obs_exp(matrix)
    n = oe.shape[0]

    # Locally maximum filter
    neighborhood_size = 2 * local_window + 1
    local_max = maximum_filter(oe, size=neighborhood_size)

    loops = []
    for i in range(n):
        for j in range(i + min_dist_bins, min(n, i + max_dist_bins)):
            val = oe[i, j]
            if val >= enrichment_threshold and val == local_max[i, j]:
                loops.append(dict(i=int(i), j=int(j), enrichment=float(val)))

    return loops


def loop_sizes(loops: list, resolution: int = 50_000) -> np.ndarray:
    """Return loop sizes in bp."""
    if not loops:
        return np.array([])
    return np.array([(lp["j"] - lp["i"]) * resolution for lp in loops], dtype=float)


# ── Aggregate Peak Analysis (APA) ─────────────────────────────────────────────

def apa_pileup(
    matrix: np.ndarray,
    loops: list,
    flank: int = 10,
    normalize: bool = True,
) -> np.ndarray:
    """
    Aggregate Peak Analysis: average sub-matrices centred on each loop.

    Parameters
    ----------
    matrix    : ndarray (n, n)   contact matrix (O/E recommended)
    loops     : list of dict     from detect_loops()
    flank     : int              number of bins around loop anchor
    normalize : bool             if True, divide by corner mean (non-loop background)

    Returns
    -------
    pileup : ndarray (2*flank+1, 2*flank+1)   average pile-up matrix
    """
    if not loops:
        size = 2 * flank + 1
        return np.zeros((size, size))

    oe = obs_exp(matrix)
    stack = []

    for lp in loops:
        sub = extract_square(oe, lp["i"], lp["j"], flank)
        if not np.all(np.isnan(sub)):
            stack.append(sub)

    if not stack:
        return np.zeros((2 * flank + 1, 2 * flank + 1))

    pile = np.nanmean(np.stack(stack, axis=0), axis=0)

    if normalize:
        # Divide by corner quadrants (background estimate)
        q = max(1, flank // 3)
        sz = pile.shape[0]
        corners = np.concatenate([
            pile[:q, :q].ravel(),
            pile[:q, sz - q:].ravel(),
            pile[sz - q:, :q].ravel(),
            pile[sz - q:, sz - q:].ravel(),
        ])
        bg = np.nanmean(corners)
        if bg > 0:
            pile = pile / bg

    return pile


def apa_enrichment(pileup: np.ndarray, flank: int = 10) -> float:
    """
    APA enrichment score: central pixel value over background.
    """
    c = flank   # centre index
    q = max(1, flank // 3)
    sz = pileup.shape[0]
    corners = np.concatenate([
        pileup[:q, :q].ravel(),
        pileup[:q, sz - q:].ravel(),
        pileup[sz - q:, :q].ravel(),
        pileup[sz - q:, sz - q:].ravel(),
    ])
    bg = np.nanmean(corners)
    return float(pileup[c, c] / (bg + 1e-10))


# ── Full loop analysis ────────────────────────────────────────────────────────

def loop_analysis(
    matrix: np.ndarray,
    resolution: int = 50_000,
    flank: int = 10,
    **detect_kwargs,
) -> dict:
    """
    End-to-end loop analysis for one chromosome.

    Returns
    -------
    dict with:
        loops       : list of detected loops
        loop_sizes  : ndarray
        apa         : ndarray (pileup matrix)
        apa_score   : float
    """
    loops = detect_loops(matrix, **detect_kwargs)
    sizes = loop_sizes(loops, resolution)
    oe = obs_exp(matrix)
    pile = apa_pileup(oe, loops, flank=flank, normalize=True)
    score = apa_enrichment(pile, flank)

    return dict(loops=loops, loop_sizes=sizes, apa=pile, apa_score=score)


def loop_analysis_all_chromosomes(
    matrices: dict,
    chr_names: list,
    resolution: int = 50_000,
    **kwargs,
) -> dict:
    """Run loop analysis on every chromosome."""
    return {
        chrom: loop_analysis(matrices[chrom], resolution, **kwargs)
        for chrom in chr_names
    }
