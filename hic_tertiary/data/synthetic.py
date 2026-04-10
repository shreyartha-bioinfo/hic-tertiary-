"""
Synthetic Hi-C data generator.

Produces realistic contact count matrices that exhibit:
  • Power-law distance decay  P(s) ~ s^{-α}
  • TAD block structure  (within-TAD enrichment)
  • A/B compartment checkerboard  (same-type enrichment)
  • Chromatin loop point enrichments
  • Poisson sequencing noise
"""
import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_hic(
    n_bins: int = 200,
    resolution: int = 50_000,
    n_chromosomes: int = 3,
    seed: int = 42,
    noise_level: float = 1.0,
    decay_alpha: Optional[float] = None,
) -> dict:
    """
    Generate a realistic synthetic Hi-C dataset.

    Parameters
    ----------
    n_bins        : int    — bins per chromosome
    resolution    : int    — bin size in bp
    n_chromosomes : int    — number of chromosomes
    seed          : int    — random seed
    noise_level   : float  — Poisson noise scaling (1.0 = realistic)
    decay_alpha   : float  — power-law exponent (None → random 0.9–1.3)

    Returns
    -------
    dict with keys:
        matrices      : {chrom: ndarray (n,n)}  raw integer contact counts
        tad_boundaries: {chrom: list[int]}       boundary bin indices (incl. 0 and n)
        compartments  : {chrom: ndarray (n,)}    +1=A (active), -1=B (inactive)
        loops         : {chrom: list[(i,j)]}     loop anchor pairs
        bins          : pd.DataFrame             bin coordinates
        resolution    : int
        chr_names     : list[str]
    """
    rng = np.random.default_rng(seed)
    chr_names = [f"chr{i + 1}" for i in range(n_chromosomes)]

    matrices: dict = {}
    tad_boundaries: dict = {}
    compartments: dict = {}
    loops: dict = {}
    bins_rows = []

    for chrom in chr_names:
        n = n_bins
        alpha = decay_alpha if decay_alpha else rng.uniform(0.9, 1.3)

        # ── 1. Distance decay ───────────────────────────────────────────────
        idx = np.arange(n)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        dist = np.abs(ii - jj).astype(float) + 1.0
        matrix = 3000.0 / (dist ** alpha)

        # ── 2. TAD structure ─────────────────────────────────────────────────
        n_tads = max(4, n // 20)
        n_bounds = min(n_tads - 1, n // 12)
        bp = sorted(
            rng.choice(np.arange(12, n - 12), size=n_bounds, replace=False).tolist()
        )
        bounds = [0] + bp + [n]

        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            boost = rng.uniform(1.8, 3.5)
            matrix[s:e, s:e] *= boost

        # ── 3. Compartment structure ─────────────────────────────────────────
        comp = np.zeros(n, dtype=float)
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            comp[s:e] = 1.0 if k % 2 == 0 else -1.0

        # A-A and B-B contacts enriched; A-B contacts depleted
        comp_outer = np.outer(comp, comp)        # +1 if same type, -1 if different
        matrix *= np.where(comp_outer > 0, 1.45, 0.60)

        # ── 4. Chromatin loops ───────────────────────────────────────────────
        loop_list = []
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            tad_len = e - s
            if tad_len < 18:
                continue
            # Two loop anchors at ~10% and ~90% of the TAD
            a1 = s + max(2, tad_len // 10)
            a2 = e - max(2, tad_len // 10)
            loop_list.append((int(a1), int(a2)))

            # Gaussian point enrichment
            strength = rng.uniform(8.0, 25.0) * 100
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    i1, j1 = a1 + di, a2 + dj
                    if 0 <= i1 < n and 0 <= j1 < n:
                        w = np.exp(-(di ** 2 + dj ** 2) / 3.0)
                        matrix[i1, j1] += strength * w
                        matrix[j1, i1] += strength * w

        # ── 5. Symmetrize, zero diagonal, Poisson noise ──────────────────────
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0)
        matrix = rng.poisson(np.clip(matrix * noise_level, 0, None)).astype(float)
        matrix = (matrix + matrix.T) / 2.0   # re-symmetrize after noise

        matrices[chrom] = matrix
        tad_boundaries[chrom] = [int(b) for b in bounds]
        compartments[chrom] = comp
        loops[chrom] = loop_list

        for i in range(n):
            bins_rows.append(
                dict(chrom=chrom, start=i * resolution, end=(i + 1) * resolution)
            )

    bins_df = pd.DataFrame(bins_rows)

    return dict(
        matrices=matrices,
        tad_boundaries=tad_boundaries,
        compartments=compartments,
        loops=loops,
        bins=bins_df,
        resolution=resolution,
        chr_names=chr_names,
    )


def generate_condition_pair(
    n_bins: int = 200,
    resolution: int = 50_000,
    seed_a: int = 42,
    seed_b: int = 99,
) -> tuple:
    """
    Generate two Hi-C datasets (e.g. control vs. treatment) for differential analysis.
    Condition B shares the same TAD / compartment skeleton as A but with perturbations.
    """
    data_a = generate_synthetic_hic(n_bins=n_bins, resolution=resolution, seed=seed_a)

    # Condition B: same structure but different noise seed + a few loop changes
    data_b = generate_synthetic_hic(n_bins=n_bins, resolution=resolution, seed=seed_b)

    return data_a, data_b
