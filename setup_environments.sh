#!/usr/bin/env bash
# =============================================================================
# PEGA — Environment Setup Script
# =============================================================================
# Creates the four conda environments required by PEGA predictors.
#
# Usage:
#   bash setup_environments.sh            # create all environments
#   bash setup_environments.sh macrel     # create only macrel_env
#   bash setup_environments.sh amplify    # create only amplify_env
#   bash setup_environments.sh pega       # create only pega_env
#   bash setup_environments.sh hemo       # create only hemo_env
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
    for name in pega_env macrel_env amplify_env hemo_env; do
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

    info "Step 2/2: Installing AMPlify via mamba (bioconda::amplify) ..."
    conda activate amplify_env && mamba install bioconda::amplify -y

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
    all)        create_pega_env; create_macrel_env; create_amplify_env; create_hemo_env ;;
    pega)       create_pega_env ;;
    macrel)     create_macrel_env ;;
    amplify)    create_amplify_env ;;
    hemo)       create_hemo_env ;;
    *)
        echo "Unknown argument: $TARGET"
        echo "Usage: bash setup_environments.sh [all|pega|macrel|amplify|hemo|--status]"
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
