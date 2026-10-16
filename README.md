<div align="center">

```
 ██████╗ ███████╗ ██████╗  █████╗ 
 ██╔══██╗██╔════╝██╔════╝ ██╔══██╗
 ██████╔╝█████╗  ██║  ███╗███████║
 ██╔═══╝ ██╔══╝  ██║   ██║██╔══██║
 ██║     ███████╗╚██████╔╝██║  ██║   ██╗ █████╗ ██╗  ██╗
 ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝ ██╔══██╗╚██╗██╔╝
                                         █████╔╝  ╚███╔╝ 
                                         ██╔══╝   ██╔╝   
                                         ╚═╝      ╚═╝    
```

**Peptide Evolution via Genetic Algorithm**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/omicsoldier/PEGA.py)
[![Predictors](https://img.shields.io/badge/predictors-10-teal.svg)](https://github.com/omicsoldier/PEGA.py)

*Multi-predictor ensemble scoring for antimicrobial peptide research*

</div>

---

## Overview

PEGA.py integrates **10 published AMP predictors** into a single, parallelized scoring
pipeline. Given a FASTA file, it returns predictor scores and validated ensemble scores,
saved automatically as TSV files. It supports datasets from a few sequences up to
**1M+ sequences** via chunked screening mode.

---

## Predictors

| ID | Name | Method | Dependency |
|----|------|--------|------------|
| 1 | AMPnet | CNN (TensorFlow, batch) | `pega_env` |
| 2 | amPEPpy | Random Forest | `pega_env` |
| 3 | ampir (mature) | SVM | `pega_env` + R |
| 4 | ampir (precursor) | SVM | `pega_env` + R |
| 5 | AMP_CG | ESM-2 Transformer (PyTorch) | `pega_env` |
| 6 | Macrel | SVM + physico-chemical | `macrel_env` |
| 7 | AMPlify (balanced) | Attentive deep learning | `amplify_env` |
| 8 | AMPlify (imbalanced) | Attentive deep learning | `amplify_env` |
| 9 | modlAMP RF | Random Forest (PepCATS) | `pega_env` |
| 10 | modlAMP SVM | SVM (PepCATS) | `pega_env` |
| 11 | HemoPI2 (RF) | Random Forest — **haemolysis**, not activity | `hemo_env` |

Predictors 1–10 score antimicrobial activity. Predictor 11 scores a
different endpoint — haemolytic toxicity — and is described under
[Output](#hemolysis-columns).

## Validated Ensembles

Every activity has two ensembles. Use `_class` to label peptides at a 0.5
threshold, and `_screen` to rank candidates for virtual screening — they
were optimised for different objectives (MCC vs. BEDROC) and generally
select different predictors and transformations.

**Classification (MCC-optimised).** MCC below is the nested-CV value, i.e.
out-of-fold, not the in-sample figure.

| Ensemble | Predictors | Task | Strategy | MCC | AUC |
|----------|-----------|------|----------|-----|-----|
| `ensemble_AMP_class_score` | AMPnet, Macrel, AMPlify | Antimicrobial | mean | 0.839 | 0.962 |
| `ensemble_AVP_class_score` | AMP_CG, AMPlify, modlAMP RF | Antiviral | weighted mean | 0.412 | 0.765 |
| `ensemble_AFP_class_score` | Macrel, AMP_CG, ampir, modlAMP RF | Antifungal | vote | 0.700 | 0.881 |
| `ensemble_APP_class_score` | Macrel, AMPnet | Antiparasitic | weighted mean | 0.623 | 0.867 |
| `ensemble_ABP_class_score` | ampir, AMP_CG, AMPnet, modlAMP RF | Antibacterial | weighted mean | 0.854 | 0.967 |

**Screening (BEDROC-optimised).**

| Ensemble | Predictors | Task | Strategy | BEDROC | AUPRC |
|----------|-----------|------|----------|--------|-------|
| `ensemble_AMP_screen_score` | AMPnet, Macrel, AMPlify ×2 | Antimicrobial | weighted mean | 0.891 | 0.844 |
| `ensemble_AVP_screen_score` | AMP_CG, ampir, modlAMP SVM | Antiviral | weighted mean | 0.412 | 0.271 |
| `ensemble_AFP_screen_score` | Macrel, AMP_CG, AMPnet, ampir | Antifungal | median | 0.809 | 0.751 |
| `ensemble_APP_screen_score` | Macrel, AMPnet, modlAMP SVM, ampir | Antiparasitic | median | 0.621 | 0.533 |
| `ensemble_ABP_screen_score` | AMP_CG, ampir, AMPlify, AMPnet, Macrel | Antibacterial | weighted mean | 0.890 | 0.836 |

> **Do not deploy the antiviral ensembles.** Neither beats the single
> `AMP_CG_score` predictor out of fold (classification delta MCC = −0.015;
> screening ties it at 0.412 vs 0.406). Use `AMP_CG_score` alone for AVP.
> `PEGA list --ensembles` prints this and the other caveats.
>
> `ensemble_AFP_class_score` uses a **vote**, so it only takes 5 distinct
> values (0, 0.25, 0.5, 0.75, 1). It labels well but cannot rank — use
> `ensemble_AFP_screen_score` for screening.

---

## Installation — Local

### Requirements

- Linux 64-bit
- Miniconda or Anaconda
- mamba (recommended)
- git with Git LFS

### Step 1 — Clone

```bash
git clone https://github.com/omicsoldier/PEGA.py.git
cd PEGA.py
```

> Model weights (~106 MB) are stored via Git LFS. If `git-lfs` isn't
> installed when you clone, you'll silently get small placeholder files
> instead of the real weights — run `PEGA download-models` (Step 2 below)
> to fix this regardless of whether LFS worked.

> **Remote machines / SSH nodes**: the installation procedure is identical —
> run all steps on the target node after connecting via SSH.
> No scheduler-specific configuration is needed.

### Step 2 — Main environment

```bash
conda env create -f envs/pega_env.yml
conda activate pega_env
pip install -e .
PEGA download-models
PEGA --version
PEGA list
```

> `PEGA download-models` always verifies the model files, even if
> `git clone` already ran — it detects and replaces Git LFS pointer stubs
> automatically, so this step is safe (and a no-op) whether or not LFS
> worked during cloning.

> The yml pins **Python 3.10** — required for TensorFlow and modlAMP
> compatibility. Do not change this to 3.11/3.12.
>
> **No root/sudo required** anywhere in this installation — conda
> environments, pip packages, and R packages (via CRAN) all install into
> your own user-owned prefixes. This works the same way on a shared
> cluster node as on a personal machine.

### Step 3 — ampir (R package)

```bash
conda activate pega_env

# Update r-rlang first — ampir requires >=1.1.7, conda may install an older version
conda install -n pega_env -c conda-forge r-rlang -y

# Install R dependencies
conda install -n pega_env -c conda-forge \
    r-caret r-data.table r-recipes r-modelmetrics r-ipred -y

# Install ampir from CRAN
$CONDA_PREFIX/bin/Rscript -e \
    'install.packages("ampir", repos="https://cloud.r-project.org")'
```

> **If you see** `namespace 'rlang' X.X.X is already loaded, but >= 1.1.7 is required`:
> the `r-rlang` update above fixes this. If the error persists, restart the R session
> and try again.

### Step 4 — amPEPpy

amPEPpy is not published on PyPI, so it isn't in `pega_env.yml` — install
it from its git repository after the environment is created:

```bash
conda activate pega_env
pip install git+https://github.com/tlawrence3/amPEPpy.git
```

> If this fails on a cluster node with restricted outbound git access,
> download the repository as a tarball on a machine with normal GitHub
> access and `pip install` the local copy instead.

### Step 5 — External predictor environments

**Macrel** (predictor 6):
```bash
conda env create -f envs/macrel_env.yml
```

> Use the pinned `envs/macrel_env.yml`, not a bare `conda create -n
> macrel_env macrel`. Macrel bundles a pre-trained scikit-learn model, and
> loading it with a different scikit-learn/numpy/scipy version than the
> one it was validated against can silently change its predictions for
> the same input — no error, just wrong scores. This has been observed
> in practice between two machines both running `macrel=1.6.0`, purely
> from unpinned transitive dependencies resolving differently.
>
> **Always verify after installing** (or after recreating this
> environment on a new node) by scoring a known reference AMP and
> confirming the result looks sane, e.g.:
> ```bash
> echo -e ">magainin2\nGIGKFLHSAKKFGKAFVGEIMNS" > _ref_amp.fasta
> conda run -n macrel_env macrel peptides --fasta _ref_amp.fasta --output _ref_out --keep-negatives
> gzip -d _ref_out/macrel.out.prediction.gz && cat _ref_out/macrel.out.prediction
> rm -rf _ref_amp.fasta _ref_out
> ```
> `is_AMP` should be `True` with a clearly positive `AMP_probability`
> (well above 0). If you see `False` with a value near 0, your
> `macrel_env` dependencies have drifted — recreate it from the pinned
> yml above rather than trusting the scores.

**AMPlify** (predictors 7 & 8) — requires Python 3.6, use mamba:
```bash
conda create -n amplify_env python=3.6 -y
conda activate amplify_env
mamba install bioconda::amplify
conda deactivate
```

**HemoPI2** (predictor 11, haemolysis):
```bash
conda env create -f envs/hemo_env.yml
```

> This needs its own environment and cannot go in `pega_env`. HemoPI2
> pins `scikit-learn==1.3.1`, which forces `numpy<2`; installing it
> alongside `pega_env`'s pinned `numpy=2.2` downgrades numpy and breaks
> the TensorFlow/PyTorch stack built on it.
>
> Verify with a known haemolytic peptide — melittin should score high
> and a polyanionic control low:
> ```bash
> printf '>melittin\nGIGAVLKVLTTGLPALISWIKRKRQQ\n>neg\nEEEEEEDDDDDDEEEEEEDDDDDD\n' > _hemo.fasta
> conda run -n hemo_env hemopi2_classification -i _hemo.fasta -m 1 -j 1 -wd . -o _hemo.csv
> cat _hemo.csv && rm -f _hemo.fasta _hemo.csv
> ```
> Expected: melittin ≈ 0.97 (`Hemolytic`), the negative ≈ 0.40
> (`Non-Hemolytic`). If melittin comes out low, the environment has
> drifted — recreate it from the pinned yml.

---

## Quick Start

```bash
conda activate pega_env

# Check predictors
PEGA list

# Validate FASTA
PEGA validate --fasta sequences.fasta

# Score (all predictors + ensembles)
PEGA score --fasta sequences.fasta --jobs -1 --quiet

# Score — ensemble columns only (faster)
PEGA score --fasta sequences.fasta \
    --ensembles AMP_class AFP_class APP_class ABP_class \
    --no-individual-scores \
    --jobs -1 --quiet

# Large dataset screening
PEGA screen --fasta large.fasta \
    --chunk 50000 \
    --ensembles AMP_screen AFP_screen APP_screen ABP_screen \
    --no-individual-scores \
    --quiet
```

---

## Output

### `PEGA score` → single TSV

```
PEGA_results_YYYYMMDD_HHMMSS.tsv
```

Columns: `seq_name` | `original_header` | activity scores | ensemble scores | hemolysis scores

### `PEGA screen` → folder with 3 files

```
PEGA_screen_YYYYMMDD_HHMMSS/
├── names.tsv       # pep_id | original_header | length
├── rejected.tsv    # original_header | length | reason
├── rejected.fasta  # FASTA of rejected sequences
└── scores.tsv      # pep_id | activity scores | ensemble scores | hemolysis scores
```

Join scores ↔ names on `pep_id`:

```python
import pandas as pd
names  = pd.read_csv("PEGA_screen_*/names.tsv", sep="\t")
scores = pd.read_csv("PEGA_screen_*/scores.tsv", sep="\t")
df     = scores.merge(names, on="pep_id")
```

All scores are probabilities in `[0, 1]`. Higher = more likely antimicrobial
— except for the two haemolysis columns.

<a id="hemolysis-columns"></a>
### Hemolysis columns — higher is *worse*

Two columns score haemolytic toxicity rather than antimicrobial activity:
the probability that the peptide lyses red blood cells. They are triage
flags, to help you drop promising-but-haemolytic candidates.

| Column | Source | Notes |
|--------|--------|-------|
| `hemolysis_macrel_score` | Macrel's `Hemolytic_probability` | Free — Macrel already runs for `Macrel_score` |
| `hemopi2_score` | HemoPI2 Random Forest (`hemo_env`) | AUC 0.888 on the authors' independent set |

**Neither is consumed by any ensemble**, and both are dropped by
`--no-individual-scores`, which keeps only `ensemble_*` columns. They are
placed at the end of the table, after the ensembles, so the activity
columns read as one block.

Two caveats on `hemopi2_score`: HemoPI2 truncates sequences longer than
40 residues to their first 40 before scoring, and its own
hemolytic/non-hemolytic cutoff is 0.46, not 0.5.

Why these two and not the others: of the published haemolysis predictors,
most are web servers with no installable code (HemoPred, HAPPENN,
HLPpred-Fuse), and several have code or services that no longer run —
the HemoPI2 authors report being unable to evaluate HLPpred-Fuse,
HemoNet, hemolytic-Pred or Capecchi's RNN for exactly that reason. A web
server is also unusable here: no batching, no offline reproducibility.

---

## Python API

```python
import pega

# Score with all predictors
df = pega.score("sequences.fasta")

# Score specific predictors
df = pega.score("sequences.fasta",
                predictors=["ampnet", "amp_cg", "modlamp_rf"])

# Score and export
df = pega.score("sequences.fasta", export_tsv="results.tsv")
```

---

## Project structure

```
PEGA/
├── envs/
│   ├── pega_env.yml          # main environment
│   ├── macrel_env.yml        # Macrel predictor
│   └── amplify_env.yml       # AMPlify predictor
├── pega/
│   ├── predictors/           # one module per predictor
│   ├── operators/            # GA crossover, mutation, selection
│   ├── population/           # GA population management
│   ├── preprocess/           # FASTA validation
│   ├── ensemble.py           # validated ensemble scores
│   ├── utils.py              # scoring + screen orchestration
│   └── cli.py                # PEGA CLI
├── tests/
│   └── data/test.fasta
└── pyproject.toml
```

---

## Citation

```bibtex
@software{pega2025,
  author = {omicsoldier},
  title  = {PEGA.py: Peptide Evolution via Genetic Algorithm},
  year   = {2025},
  url    = {https://github.com/omicsoldier/PEGA.py}
}
```

Please also cite the original publication for each predictor used.

---

## License

MIT — see [LICENSE](LICENSE).
