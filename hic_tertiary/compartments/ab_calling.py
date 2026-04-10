"""
A/B Chromatin Compartment Analysis.

Biological questions answered
-------------------------------
• Which genomic regions are transcriptionally active (A compartment)?
• Which are heterochromatic / lamina-associated (B compartment)?
• How strong is compartmentalization?
• Which genes switch compartments between conditions?

Method (Lieberman-Aiden et al. 2009)
--------------------------------------
1. Compute Observed/Expected (O/E) matrix (correct for distance bias)
2. Compute Pearson correlation matrix of O/E rows
3. First principal component (PC1) gives the compartment eigenvector
4. Sign convention: PC1 > 0 → A compartment (correlates with gene density / GC)
"""
import numpy as np
from sklearn.decomposition import PCA

from hic_tertiary.utils.matrix_ops import obs_exp, pearson_correlation


# ── O/E & correlation ─────────────────────────────────────────────────────────

def compute_oe(matrix: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Observed / Expected contact matrix (divides by distance-dependent mean)."""
    return obs_exp(matrix, pseudocount)


def compute_correlation(oe_matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation of O/E rows (n x n)."""
    return pearson_correlation(oe_matrix)


# ── PCA compartment calling ───────────────────────────────────────────────────

def call_compartments(
    matrix: np.ndarray,
    n_components: int = 3,
    flip_sign: bool = True,
) -> dict:
    """
    Call A/B compartments via PCA of the O/E correlation matrix.

    Parameters
    ----------
    matrix       : ndarray (n, n)  balanced contact matrix
    n_components : int             number of PCA components to return
    flip_sign    : bool            if True, flip PC1 so that the compartment with
                                   higher mean coverage is positive (A compartment)

    Returns
    -------
    dict with:
        oe          : ndarray (n, n)   O/E matrix
        correlation : ndarray (n, n)   Pearson correlation of O/E
        eigenvectors: ndarray (n_comp, n)  PCA components
        compartment : ndarray (n,)     PC1 = compartment eigenvector
        labels      : ndarray (n,)     +1=A, -1=B discrete labels
        var_explained: ndarray         variance explained per component
    """
    oe = compute_oe(matrix)
    corr = compute_correlation(oe)

    pca = PCA(n_components=n_components)
    pca.fit(corr)

    pc1 = pca.components_[0]

    # Sign convention: bins with higher coverage (more accessible, A compartment)
    # should be positive
    if flip_sign:
        cov = matrix.sum(axis=1)
        # If the correlation between coverage and PC1 is negative, flip
        if np.corrcoef(cov, pc1)[0, 1] < 0:
            pc1 = -pc1
            pca.components_[0] = pc1

    labels = np.where(pc1 >= 0, 1, -1)   # +1 = A, -1 = B

    return dict(
        oe=oe,
        correlation=corr,
        eigenvectors=pca.components_,
        compartment=pc1,
        labels=labels,
        var_explained=pca.explained_variance_ratio_,
    )


# ── Saddle plot ───────────────────────────────────────────────────────────────

def compute_saddle(
    matrix: np.ndarray,
    compartment: np.ndarray,
    n_quantiles: int = 50,
) -> tuple:
    """
    Compute a saddle plot: average contact enrichment between quantile bins
    of the compartment eigenvector.

    A saddle plot reveals the strength of compartmentalization:
    - Top-left quadrant   : BB interactions (B×B enrichment)
    - Bottom-right quadrant: AA interactions (A×A enrichment)
    - Off-diagonal        : AB interactions (should be depleted)

    Returns
    -------
    saddle  : ndarray (n_quantiles, n_quantiles)   mean O/E in each bin pair
    q_edges : ndarray   quantile edge values of the eigenvector
    """
    oe = compute_oe(matrix)
    n = len(compartment)

    # Assign each bin to a quantile group
    q_edges = np.quantile(compartment, np.linspace(0, 1, n_quantiles + 1))
    group = np.digitize(compartment, q_edges[1:-1])   # 0 .. n_quantiles-1

    saddle = np.zeros((n_quantiles, n_quantiles))
    counts = np.zeros((n_quantiles, n_quantiles), dtype=int)

    for i in range(n):
        for j in range(n):
            gi, gj = group[i], group[j]
            saddle[gi, gj] += oe[i, j]
            counts[gi, gj] += 1

    with np.errstate(invalid="ignore"):
        saddle = np.where(counts > 0, saddle / counts, np.nan)

    return saddle, q_edges


# ── Compartment strength ──────────────────────────────────────────────────────

def compartment_strength(saddle: np.ndarray, n_corner: int = 5) -> dict:
    """
    Quantify compartment strength from a saddle plot.

    Strength = (AA + BB) / (2 * AB)
    where AA, BB, AB are mean O/E in the respective corners of the saddle.
    """
    q = n_corner
    n = saddle.shape[0]
    aa = np.nanmean(saddle[n - q:, n - q:])   # bottom-right (A×A)
    bb = np.nanmean(saddle[:q, :q])            # top-left    (B×B)
    ab = np.nanmean(saddle[:q, n - q:])        # top-right   (A×B)

    strength = (aa + bb) / (2 * ab + 1e-10)
    return dict(AA=float(aa), BB=float(bb), AB=float(ab), strength=float(strength))


# ── Multi-chromosome wrapper ──────────────────────────────────────────────────

def compartments_all_chromosomes(
    matrices: dict,
    chr_names: list,
    n_quantiles: int = 50,
    **kwargs,
) -> dict:
    """Run compartment analysis for every chromosome."""
    results = {}
    for chrom in chr_names:
        mat = matrices[chrom]
        comp = call_compartments(mat, **kwargs)
        saddle, q_edges = compute_saddle(mat, comp["compartment"], n_quantiles)
        strength = compartment_strength(saddle)
        results[chrom] = dict(**comp, saddle=saddle, q_edges=q_edges, strength=strength)
    return results
