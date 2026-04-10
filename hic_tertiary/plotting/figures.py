"""
Publication-quality Hi-C figure generation.

All plot functions follow the same convention:
    plot_*(data, ..., ax=None, save_path=None) -> matplotlib Figure

If ax=None a new figure is created; if save_path is given the figure is saved.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm
import seaborn as sns
from pathlib import Path

# ── palette ───────────────────────────────────────────────────────────────────
CMAP_CONTACT = "YlOrRd"
CMAP_OE = "RdBu_r"
CMAP_DIV = "RdBu_r"
COMPARTMENT_COLORS = {1: "#E74C3C", -1: "#3498DB"}   # A=red, B=blue

sns.set_theme(style="ticks", font_scale=1.1)


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contact map
# ─────────────────────────────────────────────────────────────────────────────

def plot_contact_map(
    matrix: np.ndarray,
    resolution: int = 50_000,
    title: str = "Hi-C Contact Map",
    vmin=None,
    vmax=None,
    log_scale: bool = True,
    ax=None,
    save_path: str = None,
):
    """
    Heat-map of the raw or normalised contact matrix.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    m = matrix.copy()
    m[m == 0] = np.nan
    if log_scale:
        vmin = vmin or 1
        norm = LogNorm(vmin=vmin, vmax=vmax or np.nanpercentile(m, 99))
    else:
        norm = None

    n = matrix.shape[0]
    extent_mb = n * resolution / 1e6
    im = ax.imshow(
        m, cmap=CMAP_CONTACT, norm=norm,
        aspect="auto",
        extent=[0, extent_mb, extent_mb, 0],
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, shrink=0.7, label="Contacts" + (" (log)" if log_scale else ""))
    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Genomic position (Mb)")
    ax.set_title(title)

    if standalone:
        fig.tight_layout()
        return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. QC panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_qc_panel(
    matrix: np.ndarray,
    qc_metrics: dict,
    resolution: int = 50_000,
    chrom: str = "chr1",
    save_path: str = None,
):
    """
    Four-panel QC figure:
      (a) Coverage per bin  (b) Distance decay (log-log)
      (c) Cis/trans ratio   (d) Data completeness heatmap
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # (a) Coverage histogram
    ax_a = fig.add_subplot(gs[0, 0])
    cov = matrix.sum(axis=1)
    ax_a.hist(cov, bins=30, color="#2ECC71", edgecolor="white", linewidth=0.5)
    ax_a.axvline(np.median(cov), color="k", linestyle="--", label=f"Median={np.median(cov):.0f}")
    ax_a.set_xlabel("Coverage (total contacts per bin)")
    ax_a.set_ylabel("Number of bins")
    ax_a.set_title("(a) Per-bin Coverage")
    ax_a.legend(fontsize=9)

    # (b) Distance decay — raw
    dd = qc_metrics[chrom]["distance_decay"]
    s = dd["distances"]
    ps = dd["contacts"]
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.loglog(s / 1e6, ps, "o-", ms=3, lw=1.5, color="#E67E22")
    ax_b.set_xlabel("Genomic distance (Mb)")
    ax_b.set_ylabel("Mean contact count")
    ax_b.set_title("(b) Distance Decay (raw)")

    # (c) Cis / trans bar
    ax_c = fig.add_subplot(gs[1, 0])
    gbl = qc_metrics["global"]
    bars = ax_c.bar(
        ["Cis", "Trans"],
        [gbl["cis_fraction"] * 100, (1 - gbl["cis_fraction"]) * 100],
        color=["#3498DB", "#E74C3C"],
    )
    for bar in bars:
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_c.set_ylim(0, 110)
    ax_c.set_ylabel("% of all contacts")
    ax_c.set_title("(c) Cis / Trans Ratio")

    # (d) Completeness heat-map (down-sampled)
    ax_d = fig.add_subplot(gs[1, 1])
    step = max(1, matrix.shape[0] // 50)
    m_ds = matrix[::step, ::step]
    nonzero = (m_ds > 0).astype(float)
    im = ax_d.imshow(nonzero, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_d, shrink=0.7, label="Non-zero")
    compl = qc_metrics[chrom]["completeness"]["completeness"]
    ax_d.set_title(f"(d) Data Completeness  ({compl * 100:.1f}% filled)")
    ax_d.set_xlabel("Bin (down-sampled)")
    ax_d.set_ylabel("Bin (down-sampled)")

    fig.suptitle(f"Quality Control — {chrom}", fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normalization comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_normalization_comparison(
    raw: np.ndarray,
    ice: np.ndarray,
    kr: np.ndarray,
    vc: np.ndarray,
    bias_ice: np.ndarray,
    save_path: str = None,
):
    """
    Three-panel: raw vs ICE vs KR vs VC contact maps + bias histogram.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    mats = [raw, ice, kr, vc]
    labels = ["Raw", "ICE", "KR", "VC"]
    for ax, m, lbl in zip(axes, mats, labels):
        m_plot = m.copy()
        m_plot[m_plot <= 0] = np.nan
        clip = np.nanpercentile(m_plot, 99)
        ax.imshow(
            m_plot,
            cmap=CMAP_CONTACT,
            norm=LogNorm(vmin=1e-4, vmax=clip),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(lbl, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Normalization Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. P(s) curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_ps_curves(
    ps_results: dict,
    save_path: str = None,
):
    """
    Log-log P(s) curves for all chromosomes with power-law fits.
    Also shows the derivative d(log P)/d(log s).
    """
    chrs = list(ps_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(chrs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for chrom, col in zip(chrs, colors):
        res = ps_results[chrom]
        s_mb = res["s"] / 1e6
        ps = res["ps"]
        fit = res["fit"]

        ax1.loglog(s_mb, ps, "o", ms=3, color=col, alpha=0.7, label=chrom)
        ax1.loglog(s_mb, fit["fit_ps"], "-", color=col, lw=1.8,
                   label=f"{chrom} α={fit['alpha']:.2f}")

        # Derivative
        log_s = np.log10(res["s"])
        log_ps = np.log10(np.clip(res["ps"], 1e-30, None))
        slope = np.gradient(log_ps, log_s)
        ax2.semilogx(s_mb, slope, "-", color=col, lw=1.8, label=chrom)

    ax1.set_xlabel("Genomic distance (Mb)")
    ax1.set_ylabel("P(s) — mean contact")
    ax1.set_title("P(s) Contact Probability Curve")
    ax1.legend(fontsize=8, ncol=2)

    ax2.axhline(-1.0, color="gray", linestyle=":", label="α=1 (fractal globule)")
    ax2.axhline(-0.5, color="gray", linestyle="--", label="α=0.5 (equil. globule)")
    ax2.set_xlabel("Genomic distance (Mb)")
    ax2.set_ylabel("d log P / d log s  (local exponent)")
    ax2.set_title("P(s) Derivative — Regime Detection")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compartment analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_compartment_map(
    matrix: np.ndarray,
    comp_result: dict,
    resolution: int = 50_000,
    chrom: str = "chr1",
    save_path: str = None,
):
    """
    Three-panel compartment figure:
      (a) O/E contact map  (b) Compartment eigenvector track  (c) Saddle plot
    """
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1.5], wspace=0.35)

    # (a) O/E map
    ax_a = fig.add_subplot(gs[0])
    oe = comp_result["oe"]
    n = oe.shape[0]
    ext = n * resolution / 1e6
    clip = np.nanpercentile(np.abs(oe), 98)
    im_a = ax_a.imshow(
        oe, cmap=CMAP_OE, vmin=-clip, vmax=clip,
        aspect="auto", extent=[0, ext, ext, 0],
    )
    plt.colorbar(im_a, ax=ax_a, shrink=0.7, label="O/E")
    ax_a.set_title(f"(a) O/E Matrix — {chrom}")
    ax_a.set_xlabel("Mb"); ax_a.set_ylabel("Mb")

    # (b) Eigenvector track
    ax_b = fig.add_subplot(gs[1])
    ev = comp_result["compartment"]
    pos = np.arange(len(ev)) * resolution / 1e6
    ax_b.fill_between(pos, ev, 0, where=ev >= 0, color="#E74C3C", alpha=0.7, label="A compartment")
    ax_b.fill_between(pos, ev, 0, where=ev < 0, color="#3498DB", alpha=0.7, label="B compartment")
    ax_b.axhline(0, color="k", lw=0.8)
    ax_b.set_xlabel("Genomic position (Mb)")
    ax_b.set_ylabel("PC1 (compartment eigenvector)")
    ax_b.set_title("(b) A/B Compartment Eigenvector")
    ax_b.legend(fontsize=9)

    explained = comp_result["var_explained"]
    ax_b.text(
        0.97, 0.97,
        f"PC1 explains {explained[0] * 100:.1f}% var.",
        transform=ax_b.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
    )

    # (c) Saddle plot
    ax_c = fig.add_subplot(gs[2])
    saddle = comp_result["saddle"]
    strength = comp_result["strength"]
    vabs = np.nanpercentile(np.abs(saddle), 95)
    im_c = ax_c.imshow(
        saddle, cmap=CMAP_OE, vmin=0.2, vmax=3.0, aspect="auto",
    )
    plt.colorbar(im_c, ax=ax_c, shrink=0.7, label="Mean O/E")
    ax_c.set_title(
        f"(c) Saddle Plot\nStrength={strength['strength']:.2f}",
        fontsize=10,
    )
    ax_c.set_xlabel("Bins sorted by PC1\n← B          A →")
    ax_c.set_ylabel("← B          A →")

    fig.suptitle(f"Compartment Analysis — {chrom}", fontsize=14, fontweight="bold")
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TAD analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_tad_analysis(
    matrix: np.ndarray,
    tad_result: dict,
    resolution: int = 50_000,
    chrom: str = "chr1",
    region_bins: tuple = None,
    save_path: str = None,
):
    """
    Three-panel TAD figure:
      (a) Contact map with boundary lines
      (b) Insulation score track
      (c) TAD size distribution
    """
    if region_bins:
        s_b, e_b = region_bins
        mat = matrix[s_b:e_b, s_b:e_b]
        score = tad_result["insulation_score"][s_b:e_b]
        bounds = tad_result["boundaries"]
        bounds = bounds[(bounds >= s_b) & (bounds < e_b)] - s_b
        offset = s_b
    else:
        mat = matrix
        score = tad_result["insulation_score"]
        bounds = tad_result["boundaries"]
        offset = 0

    n = mat.shape[0]
    ext = n * resolution / 1e6
    base = offset * resolution / 1e6

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1.5], wspace=0.4)

    # (a) Contact map + boundaries
    ax_a = fig.add_subplot(gs[0])
    m_plot = mat.copy().astype(float)
    m_plot[m_plot == 0] = np.nan
    clip = np.nanpercentile(m_plot, 98)
    ax_a.imshow(
        m_plot, cmap=CMAP_CONTACT,
        norm=LogNorm(vmin=1, vmax=clip),
        aspect="auto",
        extent=[base, base + ext, base + ext, base],
    )
    for b in bounds:
        bp_mb = (b + offset) * resolution / 1e6
        ax_a.axhline(bp_mb, color="cyan", lw=0.8, alpha=0.9)
        ax_a.axvline(bp_mb, color="cyan", lw=0.8, alpha=0.9)
    ax_a.set_title(f"(a) TAD Boundaries — {chrom}")
    ax_a.set_xlabel("Mb"); ax_a.set_ylabel("Mb")

    # (b) Insulation score track
    ax_b = fig.add_subplot(gs[1])
    pos = (np.arange(len(score)) + offset) * resolution / 1e6
    ax_b.plot(pos, score, color="#2C3E50", lw=1.5, label="Insulation Score")
    ax_b.fill_between(pos, score, score.min(), alpha=0.2, color="#2C3E50")
    for b in bounds:
        bp_mb = (b + offset) * resolution / 1e6
        ax_b.axvline(bp_mb, color="red", lw=0.8, alpha=0.8)
    ax_b.set_xlabel("Genomic position (Mb)")
    ax_b.set_ylabel("log₂ Insulation Score")
    ax_b.set_title(f"(b) Insulation Score\n({len(bounds)} boundaries called)")
    ax_b.legend(fontsize=9)

    # (c) TAD size histogram
    ax_c = fig.add_subplot(gs[2])
    sizes_mb = tad_result["tad_sizes_bp"] / 1e6
    if len(sizes_mb) > 0:
        ax_c.hist(sizes_mb, bins=15, color="#9B59B6", edgecolor="white")
        ax_c.axvline(np.median(sizes_mb), color="k", linestyle="--",
                     label=f"Median={np.median(sizes_mb):.2f} Mb")
        ax_c.set_xlabel("TAD size (Mb)")
        ax_c.set_ylabel("Count")
        ax_c.set_title(f"(c) TAD Size Distribution\n(n={len(sizes_mb)} TADs)")
        ax_c.legend(fontsize=9)
    else:
        ax_c.text(0.5, 0.5, "No TADs detected", ha="center", va="center",
                  transform=ax_c.transAxes)

    fig.suptitle(f"TAD Analysis — {chrom}", fontsize=14, fontweight="bold")
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Loop analysis (APA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loop_analysis(
    loop_result: dict,
    resolution: int = 50_000,
    chrom: str = "chr1",
    save_path: str = None,
):
    """
    Two-panel loop figure:
      (a) APA pile-up  (b) Loop size distribution
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) APA pile-up
    pile = loop_result["apa"]
    vmax = np.nanpercentile(pile, 98)
    im = ax_a.imshow(
        pile, cmap="hot", aspect="equal", vmin=0, vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax_a, shrink=0.7, label="Mean O/E (norm.)")
    sz = pile.shape[0]
    ax_a.axhline(sz // 2, color="cyan", lw=0.7, linestyle="--")
    ax_a.axvline(sz // 2, color="cyan", lw=0.7, linestyle="--")
    n_loops = len(loop_result["loops"])
    score = loop_result["apa_score"]
    ax_a.set_title(
        f"(a) APA Pile-up — {chrom}\n"
        f"n={n_loops} loops, enrichment={score:.2f}×",
        fontsize=11,
    )
    ax_a.set_xticks([]); ax_a.set_yticks([])

    # (b) Loop size distribution
    sizes = loop_result["loop_sizes"] / 1e6
    if len(sizes) > 0:
        ax_b.hist(sizes, bins=20, color="#1ABC9C", edgecolor="white")
        ax_b.axvline(np.median(sizes), color="k", linestyle="--",
                     label=f"Median={np.median(sizes):.2f} Mb")
        ax_b.set_xlabel("Loop size (Mb)")
        ax_b.set_ylabel("Number of loops")
        ax_b.set_title(f"(b) Loop Size Distribution\n({chrom})")
        ax_b.legend(fontsize=9)
    else:
        ax_b.text(0.5, 0.5, "No loops detected", ha="center", va="center",
                  transform=ax_b.transAxes)

    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Differential analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_differential(
    diff_result_chrom: dict,
    comp_diff: dict,
    chrom: str = "chr1",
    resolution: int = 50_000,
    save_path: str = None,
):
    """
    Three-panel differential figure:
      (a) Log2 FC contact map  (b) Compartment delta track  (c) SCC per stratum
    """
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1.5], wspace=0.4)

    lfc = diff_result_chrom["lfc"]
    scc_res = diff_result_chrom["scc_result"]
    comp_d = diff_result_chrom["comp_diff"]
    ins_d = diff_result_chrom["ins_diff"]

    # (a) LFC map
    ax_a = fig.add_subplot(gs[0])
    n = lfc.shape[0]
    ext = n * resolution / 1e6
    clip = np.nanpercentile(np.abs(lfc), 97)
    im_a = ax_a.imshow(
        lfc, cmap=CMAP_DIV,
        norm=TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip),
        aspect="auto", extent=[0, ext, ext, 0],
    )
    plt.colorbar(im_a, ax=ax_a, shrink=0.7, label="log₂ FC (B/A)")
    ax_a.set_title(f"(a) Differential Contacts — {chrom}")
    ax_a.set_xlabel("Mb"); ax_a.set_ylabel("Mb")

    # (b) Compartment delta
    ax_b = fig.add_subplot(gs[1])
    pos = np.arange(len(comp_d["delta"])) * resolution / 1e6
    delta = comp_d["delta"]
    ax_b.fill_between(pos, delta, 0, where=delta >= 0, color="#E74C3C", alpha=0.7,
                      label=f"B→A switches ({comp_d['n_b_to_a']})")
    ax_b.fill_between(pos, delta, 0, where=delta < 0, color="#3498DB", alpha=0.7,
                      label=f"A→B switches ({comp_d['n_a_to_b']})")
    ax_b.axhline(0, color="k", lw=0.8)

    # Mark insulation changes
    gained_pos = pos[ins_d["gained"]]
    lost_pos = pos[ins_d["lost"]]
    if gained_pos.size:
        ax_b.scatter(gained_pos, np.zeros(len(gained_pos)) + delta.min() * 0.9,
                     marker="^", color="green", s=30, label="Gained boundary", zorder=5)
    if lost_pos.size:
        ax_b.scatter(lost_pos, np.zeros(len(lost_pos)) + delta.max() * 0.9,
                     marker="v", color="orange", s=30, label="Lost boundary", zorder=5)

    ax_b.set_xlabel("Genomic position (Mb)")
    ax_b.set_ylabel("ΔPC1 (Condition B − A)")
    ax_b.set_title(f"(b) Compartment Switches\n({comp_d['n_switched']} bins switched)")
    ax_b.legend(fontsize=8)

    # (c) SCC per stratum
    ax_c = fig.add_subplot(gs[2])
    r = scc_res["per_stratum_r"]
    strata = np.arange(1, len(r) + 1)
    valid = ~np.isnan(r)
    ax_c.scatter(strata[valid], r[valid], s=8, alpha=0.7, color="#2C3E50")
    ax_c.axhline(scc_res["scc"], color="red", lw=1.5, label=f"SCC={scc_res['scc']:.3f}")
    ax_c.set_xlabel("Stratum (genomic distance in bins)")
    ax_c.set_ylabel("Pearson r")
    ax_c.set_title(f"(c) SCC per Stratum")
    ax_c.legend(fontsize=9)
    ax_c.set_ylim(-0.2, 1.1)

    fig.suptitle(f"Differential Analysis A vs B — {chrom}",
                 fontsize=14, fontweight="bold")
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary overview figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_overview(
    matrix: np.ndarray,
    comp_result: dict,
    tad_result: dict,
    loop_result: dict,
    resolution: int = 50_000,
    chrom: str = "chr1",
    save_path: str = None,
):
    """
    Comprehensive single-chromosome overview figure with:
    - Contact map (upper)
    - Compartment eigenvector track
    - Insulation score track
    - Loop position markers
    """
    fig = plt.figure(figsize=(14, 10))

    # Layout: large map on top, three annotation tracks below
    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[4, 1, 1, 1],
        hspace=0.05,
    )

    n = matrix.shape[0]
    pos = np.arange(n) * resolution / 1e6
    ext = n * resolution / 1e6

    # (a) Contact map
    ax_map = fig.add_subplot(gs[0])
    m_plot = matrix.astype(float)
    m_plot[m_plot == 0] = np.nan
    clip = np.nanpercentile(m_plot, 98)
    ax_map.imshow(
        m_plot, cmap=CMAP_CONTACT,
        norm=LogNorm(vmin=1, vmax=clip),
        aspect="auto", extent=[0, ext, ext, 0],
    )
    # TAD boundaries
    for b in tad_result["boundaries"]:
        bp_mb = b * resolution / 1e6
        ax_map.axhline(bp_mb, color="cyan", lw=0.6, alpha=0.8)
        ax_map.axvline(bp_mb, color="cyan", lw=0.6, alpha=0.8)
    # Loop dots
    for lp in loop_result["loops"][:50]:   # plot max 50 loops
        x = lp["j"] * resolution / 1e6
        y = lp["i"] * resolution / 1e6
        ax_map.plot(x, y, "o", ms=3, color="lime", alpha=0.6)
        ax_map.plot(y, x, "o", ms=3, color="lime", alpha=0.6)
    ax_map.set_title(f"Hi-C Overview — {chrom}  (cyan=TAD boundaries, green=loops)",
                     fontsize=13, fontweight="bold")
    ax_map.set_xlim(0, ext)
    ax_map.set_ylim(ext, 0)
    ax_map.set_xticklabels([])

    # (b) Compartment track
    ax_comp = fig.add_subplot(gs[1], sharex=ax_map)
    ev = comp_result["compartment"]
    ax_comp.fill_between(pos, ev, 0, where=ev >= 0, color="#E74C3C", alpha=0.8)
    ax_comp.fill_between(pos, ev, 0, where=ev < 0, color="#3498DB", alpha=0.8)
    ax_comp.axhline(0, color="k", lw=0.5)
    ax_comp.set_ylabel("PC1", fontsize=9)
    ax_comp.set_yticklabels([])
    ax_comp.set_xticklabels([])
    ax_comp.text(0.01, 0.85, "Compartments (A=red, B=blue)", transform=ax_comp.transAxes,
                 fontsize=8)

    # (c) Insulation score
    ax_ins = fig.add_subplot(gs[2], sharex=ax_map)
    score = tad_result["insulation_score"]
    ax_ins.plot(pos, score, color="#2C3E50", lw=1.2)
    ax_ins.fill_between(pos, score, score.min(), alpha=0.15, color="#2C3E50")
    for b in tad_result["boundaries"]:
        ax_ins.axvline(b * resolution / 1e6, color="red", lw=0.6, alpha=0.7)
    ax_ins.set_ylabel("Insulation", fontsize=9)
    ax_ins.set_xticklabels([])
    ax_ins.text(0.01, 0.8, "TAD Insulation Score (red=boundaries)",
                transform=ax_ins.transAxes, fontsize=8)

    # (d) Loop strength track
    ax_loop = fig.add_subplot(gs[3], sharex=ax_map)
    if loop_result["loops"]:
        loop_positions = np.array([(lp["i"] + lp["j"]) / 2 * resolution / 1e6
                                   for lp in loop_result["loops"]])
        loop_scores = np.array([lp["enrichment"] for lp in loop_result["loops"]])
        ax_loop.vlines(loop_positions, 0, loop_scores, color="#1ABC9C", lw=1.5, alpha=0.7)
    ax_loop.set_xlabel("Genomic position (Mb)")
    ax_loop.set_ylabel("O/E", fontsize=9)
    ax_loop.text(0.01, 0.8, "Loop Enrichment", transform=ax_loop.transAxes, fontsize=8)
    ax_loop.set_xlim(0, ext)

    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Multi-chromosome SCC heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_scc_heatmap(
    scc_scores: dict,
    save_path: str = None,
):
    """
    Bar chart of SCC scores per chromosome (A vs B comparison).
    """
    chroms = list(scc_scores.keys())
    scores = [scc_scores[c]["scc_result"]["scc"] for c in chroms]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(chroms, scores, color=["#3498DB", "#E74C3C", "#2ECC71"][:len(chroms)])
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="k", linestyle=":", lw=0.8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Stratum-adjusted Correlation Coefficient")
    ax.set_title("SCC: Overall Hi-C Map Similarity (Condition A vs B)\n"
                 "SCC=1 means identical maps; SCC≈0 means uncorrelated")
    fig.tight_layout()
    return _save(fig, save_path)
