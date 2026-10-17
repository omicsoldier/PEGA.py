"""
pega.setup_envs
===============
Environment setup utilities for PEGA.

Creates and validates the conda environments required by all predictors.
Can be invoked from the command line via ``pega setup`` or used
programmatically.

Environments managed
--------------------
pega_env      Main environment — all pip-installable predictors.
macrel_env    Macrel predictor (bioconda).
amplify_env   AMPlify predictor (bioconda).
hemo_env      HemoPI2 haemolysis predictor (pip).
tox_env       ToxinPred3 toxicity + AlgPred2 allergenicity (pip).
toxipep_env   ToxiPep toxicity predictor (pip + user checkout).

R / ampir is handled separately since it requires a system R installation.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ENVS_DIR = Path(__file__).resolve().parent.parent / "envs"

_ENVIRONMENTS = {
    "pega_env": {
        "yml": _ENVS_DIR / "pega_env.yml",
        "description": "Main PEGA environment (TensorFlow, PyTorch, modlAMP, amPEPpy)",
        "check_cmd": ["python", "--version"],
    },
    "macrel_env": {
        "yml": _ENVS_DIR / "macrel_env.yml",
        "description": "Macrel predictor environment (bioconda)",
        "check_cmd": ["macrel", "--version"],
    },
    "amplify_env": {
        "yml": _ENVS_DIR / "amplify_env.yml",
        "description": "AMPlify predictor environment (bioconda)",
        "check_cmd": ["AMPlify", "--help"],
        # The yml only provides the Python 3.6 interpreter; conda cannot
        # resolve AMPlify's own dependency tree.
        "note": "AMPlify itself still needs: mamba install -n amplify_env "
                "bioconda::amplify",
    },
    "hemo_env": {
        "yml": _ENVS_DIR / "hemo_env.yml",
        "description": "HemoPI2 haemolysis predictor environment (pip)",
        "check_cmd": ["hemopi2_classification", "-h"],
    },
    "tox_env": {
        "yml": _ENVS_DIR / "tox_env.yml",
        "description": "ToxinPred3 + AlgPred2 environment (pip)",
        "check_cmd": ["toxinpred3", "-h"],
    },
    "toxipep_env": {
        "yml": _ENVS_DIR / "toxipep_env.yml",
        "description": "ToxiPep toxicity predictor environment (pip)",
        "check_cmd": ["python", "-c", "import torch, rdkit"],
        # ToxiPep has no licence file upstream, so it is cloned rather
        # than vendored or pip-installed.
        "note": "ToxiPep itself still needs: bash setup_environments.sh "
                "toxipep (clones it into third_party/)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conda_available() -> bool:
    return shutil.which("conda") is not None


def _env_exists(name: str) -> bool:
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
    )
    return name in result.stdout


def _env_functional(name: str, check_cmd: list[str]) -> bool:
    result = subprocess.run(
        ["conda", "run", "-n", name] + check_cmd,
        capture_output=True,
    )
    return result.returncode == 0


def _r_available() -> bool:
    return shutil.which("Rscript") is not None


def _ampir_installed() -> bool:
    if not _r_available():
        return False
    result = subprocess.run(
        ["Rscript", "-e",
         'if (!requireNamespace("ampir", quietly=TRUE)) quit(status=1)'],
        capture_output=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def environment_status() -> dict[str, dict]:
    """Return the installation status of all PEGA environments.

    Returns
    -------
    dict
        Keys are environment names.  Each value is a dict with
        ``exists``, ``functional``, and ``description`` keys.
    """
    if not _conda_available():
        return {
            name: {"exists": False, "functional": False, "description": info["description"]}
            for name, info in _ENVIRONMENTS.items()
        }

    status = {}
    for name, info in _ENVIRONMENTS.items():
        exists = _env_exists(name)
        functional = _env_functional(name, info["check_cmd"]) if exists else False
        status[name] = {
            "exists": exists,
            "functional": functional,
            "description": info["description"],
        }

    # R / ampir
    status["r_ampir"] = {
        "exists": _r_available(),
        "functional": _ampir_installed(),
        "description": "R + ampir package (predictors 3 & 4)",
    }

    return status


def print_status() -> None:
    """Print a formatted status table for all environments."""
    if not _conda_available():
        print("  conda not found on PATH.  Please install Miniconda or Anaconda.")
        return

    status = environment_status()
    header = f"  {'Environment':<20}  {'Exists':<8}  {'Functional':<12}  Description"
    sep = "  " + "-" * 72
    print(header)
    print(sep)
    for name, info in status.items():
        exists_str = "yes" if info["exists"] else "no"
        func_str = "yes" if info["functional"] else "no"
        print(
            f"  {name:<20}  {exists_str:<8}  {func_str:<12}  {info['description']}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def create_environment(name: str, force: bool = False) -> bool:
    """Create a single conda environment from its YAML file.

    Parameters
    ----------
    name:
        Environment name — any key of :data:`_ENVIRONMENTS`.
    force:
        If ``True``, remove and recreate the environment if it already
        exists.

    Returns
    -------
    bool
        ``True`` if the environment was created successfully.
    """
    if name not in _ENVIRONMENTS:
        print(f"  Unknown environment: {name}", file=sys.stderr)
        return False

    info = _ENVIRONMENTS[name]
    yml_path = info["yml"]

    if not yml_path.exists():
        print(
            f"  YAML file not found: {yml_path}\n"
            "  Make sure the envs/ directory is present in the repository root.",
            file=sys.stderr,
        )
        return False

    if _env_exists(name):
        if not force:
            print(f"  {name} already exists — skipping.  Use --force to recreate.")
            return True
        print(f"  Removing existing {name} environment...")
        subprocess.run(["conda", "env", "remove", "-n", name, "-y"], check=True)

    print(f"  Creating {name} from {yml_path.name}...")
    result = subprocess.run(
        ["conda", "env", "create", "-f", str(yml_path)],
        text=True,
    )

    if result.returncode != 0:
        print(f"  Failed to create {name}.", file=sys.stderr)
        return False

    print(f"  {name} created successfully.")
    note = info.get("note")
    if note:
        print(f"  Note: {note}")
    return True


def install_r_ampir() -> bool:
    """Install the ampir R package if R is available.

    Returns
    -------
    bool
        ``True`` if ampir is available after this call.
    """
    if not _r_available():
        print(
            "  R not found on PATH.\n"
            "  Install R from https://cran.r-project.org/ and re-run this command.",
            file=sys.stderr,
        )
        return False

    if _ampir_installed():
        print("  ampir already installed — skipping.")
        return True

    print("  Installing ampir R package...")
    result = subprocess.run(
        ["Rscript", "-e", 'install.packages("ampir", repos="https://cloud.r-project.org")'],
        text=True,
    )

    if result.returncode != 0:
        print("  Failed to install ampir.", file=sys.stderr)
        return False

    print("  ampir installed successfully.")
    return True


def setup_all(
    envs: list[str] | None = None,
    include_r: bool = True,
    force: bool = False,
) -> None:
    """Create all PEGA conda environments.

    Parameters
    ----------
    envs:
        List of environment names to create.  When ``None``, all three
        conda environments are created.
    include_r:
        If ``True`` (default), also attempt to install the R ampir package.
    force:
        If ``True``, recreate environments that already exist.
    """
    if not _conda_available():
        print(
            "conda is not available on PATH.\n"
            "Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html",
            file=sys.stderr,
        )
        return

    target_envs = envs or list(_ENVIRONMENTS.keys())

    print(f"\nSetting up {len(target_envs)} conda environment(s)...\n")
    results: dict[str, bool] = {}

    for name in target_envs:
        print(f"[{name}]")
        results[name] = create_environment(name, force=force)
        print()

    if include_r:
        print("[r_ampir]")
        results["r_ampir"] = install_r_ampir()
        print()

    # Summary
    ok = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    print("Setup complete.")
    if ok:
        print(f"  Installed: {', '.join(ok)}")
    if failed:
        print(f"  Failed:    {', '.join(failed)}", file=sys.stderr)
