#!/usr/bin/env bash
# =============================================================================
# PEGA — Environment Setup Script
# =============================================================================
# Creates the six conda environments required by PEGA predictors.
#
# Usage:
#   bash setup_environments.sh            # create all environments
#   bash setup_environments.sh macrel     # create only macrel_env
#   bash setup_environments.sh amplify    # create only amplify_env
#   bash setup_environments.sh pega       # create only pega_env
#   bash setup_environments.sh hemo       # create only hemo_env
#   bash setup_environments.sh tox        # create only tox_env
#   bash setup_environments.sh toxipep    # create only toxipep_env
#   bash setup_environments.sh --status   # show environment status
#
# After running this script, activate the main environment and install PEGA:
#   conda activate pega_env
#   pip install -e .
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${SCRIPT_DIR}/envs"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC}  $*"; }
fail() { echo -e "  ${RED}✗${NC}  $*"; }
info() { echo -e "  ${YELLOW}→${NC}  $*"; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

show_status() {
    echo ""
    echo "PEGA environment status"
    echo "────────────────────────────────────────"
    for name in pega_env macrel_env amplify_env hemo_env tox_env toxipep_env; do
        if env_exists "$name"; then
            ok "$name  [exists]"
        else
            fail "$name  [not found]"
        fi
    done

    # ampir inside pega_env
    if env_exists "pega_env"; then
        if conda run -n pega_env Rscript -e \
            'if (!requireNamespace("ampir", quietly=TRUE)) quit(status=1)' \
            &>/dev/null; then
            ok "ampir R package  [installed inside pega_env]"
        else
            fail "ampir R package  [not installed]"
        fi

        # AlgPred3 also lives inside pega_env rather than one of its own.
        # Looked up rather than run: "algpred3 -h" exits 0 but leaves a
        # tmp/ directory behind in whatever directory it was called from.
        if conda run -n pega_env bash -c 'command -v algpred3' &>/dev/null; then
            ok "algpred3  [installed inside pega_env]"
        else
            fail "algpred3  [not installed]"
        fi
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# pega_env
# ---------------------------------------------------------------------------

create_pega_env() {
    echo ""
    echo "── pega_env ──────────────────────────────"
    if env_exists "pega_env"; then
        ok "pega_env already exists — skipping."
        info "To recreate: conda env remove -n pega_env && bash setup_environments.sh pega"
        return 0
    fi

    info "Creating pega_env from envs/pega_env.yml ..."
    conda env create -f "${ENVS_DIR}/pega_env.yml"
    ok "pega_env created."

    local RSCRIPT="$HOME/miniconda3/envs/pega_env/bin/Rscript"

    # Step 1: install compiled R dependencies via conda (avoids source build failures)
    info "Installing R dependency packages via conda ..."
    conda install -n pega_env -c conda-forge \
        r-caret r-data.table r-recipes r-modelmetrics r-ipred -y

    # Step 2: install ampir from CRAN using the conda R binary directly
    info "Installing ampir from CRAN ..."
    "$RSCRIPT" -e 'install.packages("ampir", repos="https://cloud.r-project.org")'

    if "$RSCRIPT" -e 'library(ampir)' &>/dev/null; then
        ok "ampir installed successfully."
    else
        fail "ampir installation failed. Try manually:"
        info "  conda install -n pega_env -c conda-forge r-caret r-data.table r-recipes r-modelmetrics r-ipred -y"
        info "  \$CONDA_PREFIX/bin/Rscript -e 'install.packages(\"ampir\", repos=\"https://cloud.r-project.org\")'"
    fi

    warm_up_algpred3
}

# AlgPred3's wheel carries no model: it fetches ~210 MB (model, feature
# binary, column list) into its own site-packages the first time it runs.
# Doing that here means a screening run never stops to download, and a
# machine without network access finds out now rather than mid-run.
warm_up_algpred3() {
    info "Warming up AlgPred3 (downloads ~210 MB on first use) ..."

    local warm_dir
    warm_dir="$(mktemp -d)"
    printf '>warmup\nGIGKFLHSAKKFGKAFVGEIMNS\n' > "${warm_dir}/warmup.fasta"

    # Run inside the temp directory: AlgPred3 and pfeature_comp drop
    # scratch files into the working directory.
    if ( cd "${warm_dir}" && conda run -n pega_env algpred3 \
            -i "${warm_dir}/warmup.fasta" \
            -o "${warm_dir}/warmup.csv" -j pred ) &>/dev/null \
       && [ -s "${warm_dir}/warmup.csv" ]; then
        ok "AlgPred3 ready."
    else
        fail "AlgPred3 could not score a test sequence."
        info "  Try: conda run -n pega_env algpred3 -i seqs.fasta -o out.csv -j pred"
        info "  It needs network access on first use, and scikit-learn,"
        info "  which it does not declare as a dependency."
    fi
    rm -rf "${warm_dir}"
}

# ---------------------------------------------------------------------------
# macrel_env
# ---------------------------------------------------------------------------

create_macrel_env() {
    echo ""
    echo "── macrel_env ────────────────────────────"
    if env_exists "macrel_env"; then
        ok "macrel_env already exists — skipping."
        return 0
    fi
    # Use the pinned envs/macrel_env.yml, not a bare `conda create -c bioconda
    # macrel`: an unpinned install lets conda resolve a different, untested
    # dependency tree, which can silently change Macrel's predictions (see
    # README troubleshooting notes and envs/macrel_env.yml for details).
    info "Creating macrel_env from envs/macrel_env.yml ..."
    conda env create -f "${ENVS_DIR}/macrel_env.yml"
    ok "macrel_env created."

    # Quick smoke test
    if conda run -n macrel_env macrel --version &>/dev/null; then
        ok "macrel is functional."
    else
        fail "macrel installed but 'macrel --version' failed — check the environment."
    fi
}

# ---------------------------------------------------------------------------
# amplify_env
# ---------------------------------------------------------------------------

create_amplify_env() {
    echo ""
    echo "── amplify_env ───────────────────────────"
    if env_exists "amplify_env"; then
        ok "amplify_env already exists — skipping."
        return 0
    fi

    # AMPlify requires Python 3.6.
    # mamba resolves old dependency trees that conda/libmamba cannot.
    # Requires mamba: conda install -n base -c conda-forge mamba

    if ! command -v mamba &>/dev/null; then
        fail "mamba not found. Install it first:"
        info "conda install -n base -c conda-forge mamba"
        return 1
    fi

    info "Step 1/2: Creating Python 3.6 base environment ..."
    conda create -n amplify_env python=3.6 -y

    # NOT "conda activate amplify_env && mamba install ...": conda
    # activate is a shell function that only exists after conda's shell
    # hook is sourced, so in a script it fails with CommandNotFoundError
    # and the && short-circuits — mamba never runs and AMPlify is never
    # installed. mamba takes -n directly.
    info "Step 2/2: Installing AMPlify via mamba (bioconda::amplify) ..."
    mamba install -n amplify_env -y bioconda::amplify

    # Quick smoke test
    if conda run -n amplify_env AMPlify --help &>/dev/null; then
        ok "amplify_env created and AMPlify is functional."
    else
        fail "AMPlify installed but 'AMPlify --help' failed — check the environment."
    fi
}

# ---------------------------------------------------------------------------
# hemo_env
# ---------------------------------------------------------------------------

create_hemo_env() {
    echo ""
    echo "── hemo_env ──────────────────────────────"
    if env_exists "hemo_env"; then
        ok "hemo_env already exists — skipping."
        return 0
    fi
    # HemoPI2 is kept out of pega_env on purpose: it pins
    # scikit-learn==1.3.1, which forces numpy<2 and would downgrade
    # pega_env's numpy 2.2, breaking its TensorFlow/PyTorch stack.
    info "Creating hemo_env from envs/hemo_env.yml ..."
    conda env create -f "${ENVS_DIR}/hemo_env.yml"
    ok "hemo_env created."

    # Quick smoke test
    if conda run -n hemo_env hemopi2_classification -h &>/dev/null; then
        ok "HemoPI2 is functional."
    else
        fail "HemoPI2 installed but 'hemopi2_classification -h' failed — check the environment."
    fi
}

# ---------------------------------------------------------------------------
# tox_env
# ---------------------------------------------------------------------------

create_tox_env() {
    echo ""
    echo "── tox_env ───────────────────────────────"
    if env_exists "tox_env"; then
        ok "tox_env already exists — skipping."
        return 0
    fi
    # ToxinPred3 needs its own environment: its models are pickles fitted
    # under scikit-learn 1.0.2 and fail to load under >= 1.3, which rules
    # out both pega_env (>=1.3) and hemo_env (==1.3.1).
    info "Creating tox_env from envs/tox_env.yml ..."
    conda env create -f "${ENVS_DIR}/tox_env.yml"
    ok "tox_env created."

    # Quick smoke test
    if conda run -n tox_env toxinpred3 -h &>/dev/null && conda run -n tox_env algpred2 -h &>/dev/null; then
        ok "ToxinPred3 and AlgPred2 are functional."
    else
        fail "tox_env created but toxinpred3 or algpred2 failed — check the environment."
    fi
}

# ---------------------------------------------------------------------------
# toxipep_env
# ---------------------------------------------------------------------------

create_toxipep_env() {
    echo ""
    echo "── toxipep_env ───────────────────────────"
    if env_exists "toxipep_env"; then
        ok "toxipep_env already exists — skipping."
    else
        info "Creating toxipep_env from envs/toxipep_env.yml ..."
        conda env create -f "${ENVS_DIR}/toxipep_env.yml"
        ok "toxipep_env created."
    fi

    # ToxiPep is cloned, not vendored: its repository carries no licence
    # file, so redistributing its code and weights inside this MIT project
    # would be unlicensed. third_party/ is gitignored.
    local tp_dir="${SCRIPT_DIR}/third_party/ToxiPep"
    if [[ -f "${tp_dir}/Code/best_model_0.9.pth" ]]; then
        ok "ToxiPep already cloned at third_party/ToxiPep — skipping."
    else
        info "Cloning ToxiPep into third_party/ToxiPep ..."
        mkdir -p "${SCRIPT_DIR}/third_party"
        if git clone --depth 1 https://github.com/GGCL7/ToxiPep "${tp_dir}"; then
            ok "ToxiPep cloned."
        else
            fail "Could not clone ToxiPep — the predictor will report itself unavailable."
            return 0
        fi
    fi

    if conda run -n toxipep_env python -c "import torch, rdkit" &>/dev/null; then
        ok "ToxiPep is functional."
    else
        fail "toxipep_env created but 'import torch, rdkit' failed — check the environment."
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo ""
echo "  ____  _____ ____    _"
echo " |  _ \| ____/ ___|  / \\"
echo " | |_) |  _|| |  _  / _ \\"
echo " |  __/| |__| |_| |/ ___ \\"
echo " |_|   |_____\____/_/   \_\\"
echo ""
echo " PEGA — Environment Setup"
echo ""

if ! command -v conda &>/dev/null; then
    fail "conda not found on PATH."
    echo "     Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

TARGET="${1:-all}"

case "$TARGET" in
    --status)   show_status ;;
    all)        create_pega_env; create_macrel_env; create_amplify_env; create_hemo_env; create_tox_env; create_toxipep_env ;;
    pega)       create_pega_env ;;
    macrel)     create_macrel_env ;;
    amplify)    create_amplify_env ;;
    hemo)       create_hemo_env ;;
    tox)        create_tox_env ;;
    toxipep)    create_toxipep_env ;;
    *)
        echo "Unknown argument: $TARGET"
        echo "Usage: bash setup_environments.sh [all|pega|macrel|amplify|hemo|tox|toxipep|--status]"
        exit 1
        ;;
esac

echo ""
echo "────────────────────────────────────────"
show_status

echo " Next steps:"
echo "   conda activate pega_env"
echo "   pip install -e ."
echo "   pega list"
echo ""
