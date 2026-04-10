# Hi-C Tertiary Analysis Pipeline

A comprehensive, self-contained pipeline for tertiary analysis of Hi-C contact count data.  
All core algorithms are implemented from scratch in **NumPy/SciPy** — no proprietary tools required.

---

## What this pipeline covers

| Step | Analysis | Key output | Biological question |
|------|----------|------------|---------------------|
| 1 | Quality Control | Coverage, cis/trans ratio, completeness | Is the data usable? |
| 2 | Normalization | ICE / KR / VC | How to remove sequencing bias? |
| 3 | Distance Decay | P(s) curves + power-law fit | What polymer model fits the chromatin? |
| 4 | Compartments | A/B calling via PCA, saddle plots | Where is active vs. heterochromatin? |
| 5 | TADs | Insulation score, boundary detection | Where are domain boundaries? |
| 6 | Loops | Local enrichment, APA pile-up | Which loci form stable loops? |
| 7 | Differential | Log FC, SCC, compartment switches | What changes between conditions? |
| 8 | Overview | Multi-track summary figure | Full picture for a chromosome |

---

## Questions you can answer with Hi-C

1. Where are **A (active, gene-rich)** vs **B (inactive, heterochromatic)** chromatin compartments?
2. What is the **fractal dimension** of chromatin? (from P(s) exponent α)
3. How large are **topologically associating domains (TADs)**?
4. Which loci are the most **insulated** (strongest TAD boundaries)?
5. Which pairs of loci form stable **chromatin loops**?
6. Are loop anchors preferentially at **TAD boundaries** (CTCF sites)?
7. Which loci show **differential contact frequency** between conditions?
8. Which genomic bins **switch compartment** (A↔B) across conditions?
9. Are TAD boundaries **gained or lost** between conditions?
10. How **similar** are two Hi-C maps overall? (SCC score)
11. Which **enhancers** are in the same TAD as their target genes?
12. Do **disease-associated SNPs** fall at TAD boundaries or loop anchors?
13. What is the **contact probability** between two specific loci?

---

## Repository structure

```
hic-tertiary-/
├── hic_tertiary/              # Python package
│   ├── data/
│   │   └── synthetic.py       # Realistic synthetic Hi-C data generator
│   ├── utils/
│   │   └── matrix_ops.py      # Shared: O/E, diagonals, coverage, smoothing
│   ├── qc/
│   │   └── metrics.py         # Coverage, cis/trans, completeness
│   ├── normalization/
│   │   └── methods.py         # ICE, KR, VC normalization
│   ├── distance_decay/
│   │   └── ps_curve.py        # P(s) curves, power-law fit, derivative
│   ├── compartments/
│   │   └── ab_calling.py      # O/E, PCA, A/B calling, saddle plots
│   ├── tads/
│   │   └── insulation.py      # Diamond insulation score, TAD calling
│   ├── loops/
│   │   └── enrichment.py      # Loop detection, APA pile-up
│   ├── differential/
│   │   └── comparison.py      # Log FC, SCC, differential compartments
│   └── plotting/
│       └── figures.py         # 10 publication-quality figure types
│
├── scripts/
│   └── run_pipeline.py        # Single-command CLI pipeline runner
│
├── notebooks/
│   └── hic_tertiary_analysis.ipynb   # Interactive step-by-step analysis
│
├── workflow/
│   └── Snakefile              # Snakemake DAG for production runs
│
├── config/
│   └── config.yaml            # All tunable parameters
│
└── requirements.txt
```

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full pipeline (synthetic data, no real data needed)

```bash
python scripts/run_pipeline.py
```

Results land in `results/plots/` (PNG figures) and `results/data/` (JSON summaries).

### Use a real contact matrix

```bash
# Change normalization
python scripts/run_pipeline.py --norm kr
```

### Interactive analysis (Jupyter)

```bash
jupyter lab notebooks/hic_tertiary_analysis.ipynb
```

### Snakemake workflow

```bash
snakemake --cores 4 --snakefile workflow/Snakefile
```

---

## Figures generated

| File | Content |
|------|---------|
| `01_qc_panel.png` | 4-panel QC: coverage, distance decay, cis/trans, completeness |
| `02_normalization_comparison.png` | Raw vs ICE vs KR vs VC contact maps |
| `03_ps_curves.png` | P(s) log-log curves + power-law fits + derivative |
| `04_compartments_chrN.png` | O/E map, PC1 eigenvector track, saddle plot |
| `05_tads_chrN.png` | TAD boundary map, insulation score, size histogram |
| `06_loops_chrN.png` | APA pile-up, loop size distribution |
| `07_differential_chrN.png` | Log FC map, compartment switch track, SCC per stratum |
| `08_scc_summary.png` | SCC scores per chromosome |
| `09_overview_chrN.png` | Full overview: map + compartment + insulation + loop tracks |

---

## Algorithms implemented

| Algorithm | Reference |
|-----------|-----------|
| ICE normalization | Imakaev et al., *Nature Methods* 2012 |
| KR balancing | Knight & Ruiz, *IMA Journal* 2012 |
| P(s) curve | Lieberman-Aiden et al., *Science* 2009 |
| A/B compartments via PCA | Lieberman-Aiden et al., *Science* 2009 |
| Diamond insulation score | Crane et al., *Nature* 2015 |
| APA pile-up | Rao et al., *Cell* 2014 |
| Stratum-adjusted Correlation (SCC) | Yang et al., *Nature Methods* 2017 |

---

## Configuration

Edit `config/config.yaml` to change bin resolution, normalization method, TAD/loop thresholds, and output format.

---

## Extending the pipeline

Each module is a standalone Python file with clear input/output contracts:

```python
# All analysis functions follow this pattern:
result_dict = analysis_function(matrix_ndarray, resolution, **params)

# All plot functions:
fig = plot_*(data, ..., save_path="path/to/figure.png")
```
