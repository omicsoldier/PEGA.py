"""
pega.predictors.ampactipred
===========================
AMPActiPred predictor — antibacterial activity, deep forest (conda).

An activity score, not a liability: the probability that a peptide is
antibacterial, from the first of AMPActiPred's three stages. Stages two
and three (which bacterial strains, and the activity level against each)
are not wired in — see "Stages" below.

Environment
-----------
Needs ``ampactipred_env`` and cannot share any existing one.
``deep-forest`` ships no wheel past **CPython 3.9** for any of its
versions, and 0.1.5 imports ``sklearn.utils.fixes._joblib_parallel_args``,
which scikit-learn removed in 1.1 — so this is Python 3.9 with
``scikit-learn==1.0.2``, while ``pega_env`` is Python 3.10 and
``tox_env``, which pins the same scikit-learn, is Python 3.10 too.

Checkout and weights
--------------------
The repository carries no weights and no licence, so PEGA neither
vendors the code nor redistributes the models.
``bash setup_environments.sh ampactipred`` clones the repository into
``third_party/AMPActiPred`` and downloads the first-stage model from the
authors' server into ``third_party/AMPActiPred/model/``. Point
``PEGA_AMPACTIPRED_DIR`` at a checkout elsewhere if you already have one.

Why PEGA does not run predict.py
--------------------------------
Three reasons, each verified against the published file:

1. **It does not compile.** ``GAAC``, ``GDPC`` and ``GTPC`` were
   committed with their bodies flattened to column 0 — loop and
   ``if``/``else`` bodies included — so ``python predict.py`` fails with
   ``IndentationError`` before doing anything. They are dead code for
   prediction, reached only from ``insert_gngram``, which
   ``construct_features`` never calls.
2. **Its model paths are absolute and hardcoded** to
   ``/home/AMPActiPred/model/...``; the README tells you to edit the
   source.
3. **It reports a label, not a probability.** Stage one is read through
   ``predict`` and written to JSON as ``isABP: true/false``, whereas PEGA
   needs a [0, 1] score, which ``predict_proba`` gives.

So :mod:`pega.runners.ampactipred` reuses the authors' feature code —
reimplementing 553 descriptors would risk silently disagreeing with what
the model was trained on — and does its own input reading and model call
around it.

Stages
------
Stage two is a per-strain classifier for ten species (*E. coli*,
*S. aureus*, *P. aeruginosa*, *B. subtilis*, *S. epidermidis*,
*M. luteus*, *K. pneumoniae*, *E. faecalis*, *S. typhimurium*,
*A. baumannii*) and stage three regresses an activity level per strain.
Neither is wired in yet. Stage two would add ten activity columns whose
own cutoffs run from 0.05 to 0.85, which PEGA's verdict — a flat
``>= 0.5`` on any activity column — would read wrongly; stage three
returns an activity level, not a probability in [0, 1], so it does not
fit the column convention. Both are worth adding deliberately rather
than by extension.

Reference
---------
Yao L. et al. (2024). AMPActiPred: A three-stage framework for
predicting antibacterial peptides and activity levels with deep forest.
*Protein Science* 33:e5006.
https://github.com/lantianyao/AMPActiPred
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pega.base import BasePredictor

_CONDA_ENV = "ampactipred_env"
_DIR_VAR = "PEGA_AMPACTIPRED_DIR"

#: Where ``setup_environments.sh ampactipred`` clones the repository.
#: Kept inside the project (and gitignored) rather than vendored, so no
#: unlicensed code is redistributed, while still needing nothing from the
#: user.
_VENDOR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "AMPActiPred"

#: Subdirectory of the checkout holding the first-stage model, named as
#: the authors' tarball extracts.
_MODEL_SUBDIR = Path("model") / "first_stage_model_final"

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "ampactipred.py"


def _usable(path: Path) -> bool:
    """Whether ``path`` is a checkout with the feature code and the model."""
    model = path / _MODEL_SUBDIR
    return (
        (path / "predict.py").is_file()
        and (path / "data" / "PAAC.txt").is_file()
        # deep-forest writes these three next to each other; a half-extracted
        # tarball would otherwise only fail once a run was under way.
        and (model / "param.pkl").is_file()
        and (model / "binner.pkl").is_file()
        and (model / "estimator").is_dir()
    )


def _ampactipred_dir() -> Path | None:
    """Return a usable AMPActiPred checkout, or None if there is none.

    Checks ``PEGA_AMPACTIPRED_DIR`` first so an existing clone elsewhere
    still works, then the location the setup script clones into.
    """
    raw = os.environ.get(_DIR_VAR)
    if raw:
        path = Path(raw).expanduser()
        return path if _usable(path) else None
    if _usable(_VENDOR_DIR):
        return _VENDOR_DIR
    return None


class AMPActiPredPredictor(BasePredictor):
    """AMPActiPred antibacterial predictor (conda + cloned checkout).

    Requires::

        bash setup_environments.sh ampactipred
    """

    name = "ampactipred"
    display_name = "AMPActiPred"
    predictor_id = 17
    description = "Deep forest antibacterial predictor (AMPActiPred, conda)."
    category = "conda"

    @classmethod
    def score_column(cls) -> str:
        return "AMPActiPred_score"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _env_ready(cls) -> bool:
        """Whether ampactipred_env exists and imports. Cached: spawns conda."""
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV,
             "python", "-c", "import deepforest, modlamp"],
            capture_output=True, check=False,
        )
        return result.returncode == 0

    @classmethod
    def is_available(cls) -> bool:
        # The checkout lookup stays uncached, as ToxiPep's does: unlike a
        # conda environment, the clone can appear (or the variable be set)
        # after this is first called, and a cached False would never recover.
        return _ampactipred_dir() is not None and cls._env_ready()

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with AMPActiPred's first stage.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMPActiPred_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        col = self.score_column()

        checkout = _ampactipred_dir()
        if checkout is None:
            raise RuntimeError(
                "No usable AMPActiPred checkout found.\n"
                f"Expected it at {_VENDOR_DIR}, which "
                "'bash setup_environments.sh ampactipred' creates, or set "
                f"{_DIR_VAR} to a clone elsewhere.\n"
                "The clone needs the first-stage model extracted into "
                f"{_MODEL_SUBDIR}/."
            )

        with tempfile.TemporaryDirectory(prefix="pega_ampactipred_") as work_dir:
            result_file = os.path.join(work_dir, "ampactipred_out.csv")
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "python", str(_RUNNER),
                "--repo", str(checkout),
                "--model-dir", str(checkout / _MODEL_SUBDIR),
                "--fasta", str(fasta_path),
                "--out", result_file,
            ]

            # The runner chdirs into the checkout itself (AMPActiPred reads
            # data/PAAC.txt relative to the working directory); work_dir is
            # the cwd only so that anything written before that lands here.
            run = self._run_subprocess_with_progress(
                cmd, desc="AMPActiPred", cwd=work_dir
            )
            if run.returncode != 0:
                raise RuntimeError(
                    f"AMPActiPred failed (exit {run.returncode}).\n"
                    f"{run.stderr}\n"
                    "Check: bash setup_environments.sh ampactipred"
                )

            if not os.path.exists(result_file):
                raise FileNotFoundError(
                    f"AMPActiPred output not found: {result_file}\n"
                    "Check your AMPActiPred installation."
                )

            raw = pd.read_csv(result_file)

        if col not in raw.columns:
            raise RuntimeError(
                f"AMPActiPred output has no '{col}' column "
                f"(got {list(raw.columns)})."
            )

        raw["seq_name"] = raw["seq_name"].astype(str)
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

        return raw.dropna(subset=[col]).reset_index(drop=True)
