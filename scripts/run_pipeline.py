#!/usr/bin/env python3
"""
Hi-C Tertiary Analysis Pipeline — Main Runner
=============================================

Usage
-----
    # Full pipeline on synthetic data (default, no real data required)
    python scripts/run_pipeline.py

    # Use a real numpy contact matrix
    python scripts/run_pipeline.py --matrix data/chr1.npy --chrom chr1

    # Change output directory
    python scripts/run_pipeline.py --outdir my_results

    # Change normalization method
    python scripts/run_pipeline.py --norm kr
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Allow running from repo root without installing the package ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hic_tertiary.data.synthetic import generate_synthetic_hic, generate_condition_pair
from hic_tertiary.qc.metrics import run_qc
from hic_tertiary.normalization.methods import normalize
from hic_tertiary.distance_decay.ps_curve import ps_all_chromosomes
from hic_tertiary.compartments.ab_calling import compartments_all_chromosomes
from hic_tertiary.tads.insulation import tad_analysis_all_chromosomes
from hic_tertiary.loops.enrichment import loop_analysis_all_chromosomes
from hic_tertiary.differential.comparison import differential_analysis
from hic_tertiary.plotting import figures as figs


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def banner(text: str):
    width = 64
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def step(text: str):
    print(f"\n  ▸ {text}")


def done(text: str, elapsed: float):
    print(f"    ✓ {text}  ({elapsed:.1f}s)")


def save_json(obj, path: Path):
    """Save a JSON-serializable summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Not serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, default=_convert, indent=2)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict, outdir: Path, norm_method: str = "ice"):
    banner("Hi-C Tertiary Analysis Pipeline — Starting")

    plot_dir = outdir / "plots"
    data_dir = outdir / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Generate / load data ───────────────────────────────────────────────
    step("Generating synthetic Hi-C data (conditions A and B) …")
    t0 = time.time()
    syn = cfg["synthetic"]
    data_a, data_b = generate_condition_pair(
        n_bins=syn["n_bins"],
        resolution=cfg["data"]["resolution"],
        seed_a=syn["seed"],
        seed_b=syn["seed"] + 100,
    )
    chr_names = data_a["chr_names"]
    resolution = data_a["resolution"]
    done(f"{len(chr_names)} chromosomes × {syn['n_bins']} bins each", time.time() - t0)

    # ── 1. QC ─────────────────────────────────────────────────────────────────
    step("Running QC …")
    t0 = time.time()
    qc_a = run_qc(data_a)
    chrom_qc = chr_names[0]
    fig = figs.plot_qc_panel(
        data_a["matrices"][chrom_qc],
        qc_a,
        resolution=resolution,
        chrom=chrom_qc,
        save_path=str(plot_dir / "01_qc_panel.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    summary_qc = {c: {k: v for k, v in qc_a[c].items() if k != "distance_decay"}
                  for c in chr_names}
    summary_qc["global"] = qc_a["global"]
    save_json(summary_qc, data_dir / "qc_summary.json")
    done(
        f"Cis fraction={qc_a['global']['cis_fraction'] * 100:.1f}%, "
        f"completeness={qc_a[chrom_qc]['completeness']['completeness'] * 100:.1f}%",
        time.time() - t0,
    )

    # ── 2. Normalization ──────────────────────────────────────────────────────
    step(f"Normalizing matrices (method: {norm_method}) …")
    t0 = time.time()
    norm_a, norm_b = {}, {}
    bias_a_all = {}
    for chrom in chr_names:
        mat_a, bias_a = normalize(data_a["matrices"][chrom], method=norm_method)
        mat_b, _ = normalize(data_b["matrices"][chrom], method=norm_method)
        norm_a[chrom] = mat_a
        norm_b[chrom] = mat_b
        bias_a_all[chrom] = bias_a

    # Also compute KR and VC for comparison figure (first chrom only)
    from hic_tertiary.normalization.methods import kr_normalize, vc_normalize
    raw_c = data_a["matrices"][chrom_qc]
    ice_c, bias_ice_c = normalize(raw_c, method="ice")
    kr_c, _ = kr_normalize(raw_c)
    vc_c, _ = vc_normalize(raw_c)
    fig = figs.plot_normalization_comparison(
        raw_c, ice_c, kr_c, vc_c, bias_ice_c,
        save_path=str(plot_dir / "02_normalization_comparison.png"),
    )
    plt.close(fig)
    done("Saved normalization comparison", time.time() - t0)

    # ── 3. Distance Decay / P(s) ──────────────────────────────────────────────
    step("Computing P(s) curves …")
    t0 = time.time()
    ps_results = ps_all_chromosomes(norm_a, chr_names, resolution)
    fig = figs.plot_ps_curves(ps_results, save_path=str(plot_dir / "03_ps_curves.png"))
    plt.close(fig)
    ps_summary = {
        c: {"alpha": float(ps_results[c]["fit"]["alpha"]),
            "r_squared": float(ps_results[c]["fit"]["r_squared"])}
        for c in chr_names
    }
    save_json(ps_summary, data_dir / "ps_curve_fits.json")
    done(
        "  ".join(f"{c}: α={ps_results[c]['fit']['alpha']:.3f}" for c in chr_names),
        time.time() - t0,
    )

    # ── 4. Compartments ───────────────────────────────────────────────────────
    step("Calling A/B compartments …")
    t0 = time.time()
    data_a_norm = dict(**data_a, matrices=norm_a)
    data_b_norm = dict(**data_b, matrices=norm_b)
    comp_a = compartments_all_chromosomes(norm_a, chr_names)
    comp_b = compartments_all_chromosomes(norm_b, chr_names)
    for chrom in chr_names:
        fig = figs.plot_compartment_map(
            norm_a[chrom], comp_a[chrom],
            resolution=resolution, chrom=chrom,
            save_path=str(plot_dir / f"04_compartments_{chrom}.png"),
        )
        plt.close(fig)
    comp_summary = {
        c: {
            "var_explained_pc1": float(comp_a[c]["var_explained"][0]),
            "n_A_bins": int((comp_a[c]["labels"] == 1).sum()),
            "n_B_bins": int((comp_a[c]["labels"] == -1).sum()),
            "compartment_strength": float(comp_a[c]["strength"]["strength"]),
        }
        for c in chr_names
    }
    save_json(comp_summary, data_dir / "compartments_summary.json")
    done(
        "Compartments called for all chromosomes. "
        f"Avg strength={np.mean([comp_summary[c]['compartment_strength'] for c in chr_names]):.2f}",
        time.time() - t0,
    )

    # ── 5. TADs ───────────────────────────────────────────────────────────────
    step("Detecting TADs (insulation score method) …")
    t0 = time.time()
    tad_cfg = cfg["tads"]
    tad_a = tad_analysis_all_chromosomes(
        norm_a, chr_names, resolution,
        window=tad_cfg["insulation_window"],
        delta_threshold=tad_cfg["boundary_delta_threshold"],
        min_distance=tad_cfg["min_tad_size_bins"],
    )
    for chrom in chr_names:
        fig = figs.plot_tad_analysis(
            norm_a[chrom], tad_a[chrom],
            resolution=resolution, chrom=chrom,
            save_path=str(plot_dir / f"05_tads_{chrom}.png"),
        )
        plt.close(fig)
    tad_summary = {
        c: {
            "n_tads": len(tad_a[c]["tads"]),
            "n_boundaries": len(tad_a[c]["boundaries"]),
            "median_tad_size_mb": (
                float(np.median(tad_a[c]["tad_sizes_bp"]) / 1e6)
                if len(tad_a[c]["tad_sizes_bp"]) else 0.0
            ),
        }
        for c in chr_names
    }
    save_json(tad_summary, data_dir / "tads_summary.json")
    done(
        "  ".join(
            f"{c}: {tad_summary[c]['n_tads']} TADs, "
            f"median {tad_summary[c]['median_tad_size_mb']:.2f} Mb"
            for c in chr_names
        ),
        time.time() - t0,
    )

    # ── 6. Loops ──────────────────────────────────────────────────────────────
    step("Detecting chromatin loops …")
    t0 = time.time()
    loop_cfg = cfg["loops"]
    loop_a = loop_analysis_all_chromosomes(
        norm_a, chr_names, resolution,
        min_dist_bins=loop_cfg["min_dist_bins"],
        max_dist_bins=loop_cfg["max_dist_bins"],
        enrichment_threshold=loop_cfg["enrichment_threshold"],
        flank=loop_cfg["apa_flank"],
    )
    for chrom in chr_names:
        fig = figs.plot_loop_analysis(
            loop_a[chrom], resolution=resolution, chrom=chrom,
            save_path=str(plot_dir / f"06_loops_{chrom}.png"),
        )
        plt.close(fig)
    loop_summary = {
        c: {
            "n_loops": len(loop_a[c]["loops"]),
            "apa_score": float(loop_a[c]["apa_score"]),
            "median_loop_size_mb": (
                float(np.median(loop_a[c]["loop_sizes"]) / 1e6)
                if len(loop_a[c]["loop_sizes"]) else 0.0
            ),
        }
        for c in chr_names
    }
    save_json(loop_summary, data_dir / "loops_summary.json")
    done(
        "  ".join(f"{c}: {loop_summary[c]['n_loops']} loops" for c in chr_names),
        time.time() - t0,
    )

    # ── 7. Differential analysis ──────────────────────────────────────────────
    step("Running differential analysis (A vs B) …")
    t0 = time.time()
    tad_b = tad_analysis_all_chromosomes(
        norm_b, chr_names, resolution,
        window=tad_cfg["insulation_window"],
        delta_threshold=tad_cfg["boundary_delta_threshold"],
    )
    diff_cfg = cfg["differential"]
    diff_results = differential_analysis(
        data_a_norm, data_b_norm,
        comp_a, comp_b,
        tad_a, tad_b,
        pseudocount=diff_cfg["pseudocount"],
        scc_max_dist=diff_cfg["scc_max_dist_bins"],
    )
    for chrom in chr_names:
        fig = figs.plot_differential(
            diff_results[chrom],
            diff_results[chrom]["comp_diff"],
            chrom=chrom, resolution=resolution,
            save_path=str(plot_dir / f"07_differential_{chrom}.png"),
        )
        plt.close(fig)
    scc_scores = {c: diff_results[c] for c in chr_names}
    fig = figs.plot_scc_heatmap(scc_scores,
                                save_path=str(plot_dir / "08_scc_summary.png"))
    plt.close(fig)
    diff_summary = {
        c: {
            "scc": float(diff_results[c]["scc_result"]["scc"]),
            "n_compartment_switches": int(diff_results[c]["comp_diff"]["n_switched"]),
            "n_gained_boundaries": int(diff_results[c]["ins_diff"]["gained"].sum()),
            "n_lost_boundaries": int(diff_results[c]["ins_diff"]["lost"].sum()),
        }
        for c in chr_names
    }
    save_json(diff_summary, data_dir / "differential_summary.json")
    done(
        "  ".join(f"{c}: SCC={diff_summary[c]['scc']:.3f}" for c in chr_names),
        time.time() - t0,
    )

    # ── 8. Summary overview ───────────────────────────────────────────────────
    step("Generating summary overview figures …")
    t0 = time.time()
    for chrom in chr_names:
        fig = figs.plot_summary_overview(
            norm_a[chrom], comp_a[chrom], tad_a[chrom], loop_a[chrom],
            resolution=resolution, chrom=chrom,
            save_path=str(plot_dir / f"09_overview_{chrom}.png"),
        )
        plt.close(fig)
    done("Overview figures saved", time.time() - t0)

    # ── Print final report ────────────────────────────────────────────────────
    banner("Pipeline Complete — Summary Report")
    print(f"\n  Output directory : {outdir}")
    print(f"  Plots            : {plot_dir}")
    print(f"  Data summaries   : {data_dir}")
    print()

    for chrom in chr_names:
        print(f"  ── {chrom} ────────────────────────────────────────────")
        print(f"     Compartments : {comp_summary[chrom]['n_A_bins']} A bins / "
              f"{comp_summary[chrom]['n_B_bins']} B bins  "
              f"(strength={comp_summary[chrom]['compartment_strength']:.2f})")
        print(f"     TADs         : {tad_summary[chrom]['n_tads']} TADs, "
              f"median size={tad_summary[chrom]['median_tad_size_mb']:.2f} Mb")
        print(f"     Loops        : {loop_summary[chrom]['n_loops']} loops, "
              f"APA enrichment={loop_summary[chrom]['apa_score']:.2f}×")
        print(f"     P(s) α       : {ps_summary[chrom]['alpha']:.3f}")
        print(f"     SCC (A vs B) : {diff_summary[chrom]['scc']:.3f}")
        print(f"     Compartment switches : {diff_summary[chrom]['n_compartment_switches']} bins")
        print()

    plot_files = sorted(plot_dir.glob("*.png"))
    print(f"  Plots generated ({len(plot_files)}):")
    for pf in plot_files:
        print(f"     {pf.name}")

    print()
    banner("Questions you can now answer with these results")
    QUESTIONS = [
        "Where are A (active, gene-rich) vs B (inactive, heterochromatic) chromatin compartments?",
        "What is the fractal dimension of chromatin? (from P(s) exponent α)",
        "How large are topologically associating domains (TADs)?",
        "Which genomic loci are most insulated (strong TAD boundaries)?",
        "Which pairs of loci form stable chromatin loops?",
        "Are loop anchors preferentially at TAD boundaries (CTCF sites)?",
        "Which genomic loci show differential contact frequency between conditions?",
        "Which bins switch compartment between A and B across conditions?",
        "Are TAD boundaries gained or lost between conditions?",
        "How similar are the two Hi-C maps overall? (SCC score)",
        "Which enhancers are in the same TAD as their target genes?",
        "Do disease-associated SNPs fall at TAD boundaries or loop anchors?",
        "What is the contact probability between two specific loci of interest?",
    ]
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  {i:>2}. {q}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hi-C Tertiary Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--outdir", default="results",
                        help="Output directory")
    parser.add_argument("--norm", default=None,
                        choices=["ice", "kr", "vc"],
                        help="Normalization method (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    norm_method = args.norm or cfg["normalization"]["method"]
    outdir = Path(args.outdir)

    run_pipeline(cfg, outdir, norm_method)


if __name__ == "__main__":
    main()
