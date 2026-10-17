"""
pega.predictors.toxipep
=======================
ToxiPep predictor — peptide toxicity via sequence + atomic graph (conda).

A liability score like the haemolysis columns: higher is worse, and no
ensemble consumes it. It complements ToxinPred3 rather than replacing it
— on a 25-peptide control panel ToxiPep ranked all six known toxins
(melittin, mastoparan, protegrin-1, polyphemusin-1, apamin, conotoxin GI)
above 0.75 and the five realistic inert peptides below 0.38.

Not vendored, but set up for you
-------------------------------
ToxiPep's repository carries no licence file, so by default all rights
are reserved and its code and weights cannot be redistributed inside this
MIT project. Nothing is asked of you either: ``setup_environments.sh
toxipep`` clones it into ``third_party/ToxiPep`` (gitignored) and this
predictor looks there automatically.

Set ``PEGA_TOXIPEP_DIR`` only to point at a clone kept somewhere else; it
overrides the default location. With neither present the predictor simply
reports itself unavailable and the rest of PEGA runs normally.

The reported number is not the toxicity probability
---------------------------------------------------
``predict.py`` writes ``sequence, label, value`` where the value is::

    adjusted_prob = prob if pred == 1 else 1 - prob

i.e. the model's confidence in *whichever* class it chose, not P(toxic).
Every row therefore reads above 0.5. Taken at face value the column
inverts: on the control panel LL-37 wrote 0.990 and an inert peptide
0.952, both of which the model had in fact called non-toxic. PEGA
recovers the real P(toxic) using the label in the second column.

Other integration constraints
-----------------------------
* The weights were saved on a CUDA device and ``predict.py`` calls
  ``torch.load`` with no ``map_location``, so it cannot load on a
  CPU-only machine. PEGA runs it through a wrapper that defaults
  ``map_location`` to CPU rather than editing the upstream file.
* The output carries no sequence identifiers, only sequences — and those
  are rebuilt from token indices, so unknown residues come back altered.
  PEGA maps results to names by position, which ToxiPep preserves.
* Sequences are padded to 51 tokens; anything longer than 50 residues
  makes ``predict.py`` fail rather than truncate.

Reference
---------
https://github.com/GGCL7/ToxiPep
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pega.base import BasePredictor

_CONDA_ENV = "toxipep_env"
_DIR_VAR = "PEGA_TOXIPEP_DIR"

# Runs upstream predict.py unmodified, with torch.load defaulted to CPU.
_RUNNER = """\
import os, runpy, sys, torch
# predict.py does "from model import ...", and running it from elsewhere
# puts this wrapper's directory on sys.path instead of the checkout's.
sys.path.insert(0, os.getcwd())
_orig = torch.load
def _load(*a, **k):
    k.setdefault("map_location", "cpu")
    return _orig(*a, **k)
torch.load = _load
sys.argv = ["predict.py", "-i", sys.argv[1], "-o", sys.argv[2]]
runpy.run_path("predict.py", run_name="__main__")
"""


#: Where ``setup_environments.sh toxipep`` clones ToxiPep. Kept inside the
#: repository (and gitignored) rather than vendored, so no unlicensed code
#: is redistributed, while still needing nothing from the user.
_VENDOR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "ToxiPep" / "Code"


def _usable(path: Path) -> bool:
    return (path / "predict.py").is_file() and (path / "best_model_0.9.pth").is_file()


def _toxipep_dir() -> Path | None:
    """Return a usable ToxiPep checkout, or None if there is none.

    Checks ``PEGA_TOXIPEP_DIR`` first so an existing clone elsewhere still
    works, then the location the setup script clones into.
    """
    raw = os.environ.get(_DIR_VAR)
    if raw:
        path = Path(raw).expanduser()
        return path if _usable(path) else None
    if _usable(_VENDOR_DIR):
        return _VENDOR_DIR
    return None


class ToxiPepPredictor(BasePredictor):
    """ToxiPep toxicity predictor (conda + user-provided checkout)."""

    name = "toxipep"
    display_name = "ToxiPep"
    predictor_id = 13
    description = "BiGRU/Transformer + atomic-graph toxicity predictor (ToxiPep, conda)."
    category = "conda"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _env_ready(cls) -> bool:
        """Whether toxipep_env exists and imports. Cached: spawns conda."""
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "python", "-c", "import torch, rdkit"],
            capture_output=True,
        )
        return result.returncode == 0

    @classmethod
    def is_available(cls) -> bool:
        # The checkout lookup stays uncached: unlike a conda environment,
        # the clone can appear (or PEGA_TOXIPEP_DIR be set) after this is
        # first called, and a cached False would then never recover.
        return _toxipep_dir() is not None and cls._env_ready()

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ToxiPep.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "toxipep_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        checkout = _toxipep_dir()
        if checkout is None:
            raise RuntimeError(
                "No usable ToxiPep checkout found.\n"
                f"Expected it at {_VENDOR_DIR}, which "
                "'bash setup_environments.sh toxipep' creates, or set "
                f"{_DIR_VAR} to a clone elsewhere."
            )

        names = [
            line[1:].split()[0]
            for line in Path(fasta_path).read_text().splitlines()
            if line.startswith(">")
        ]

        with tempfile.TemporaryDirectory(prefix="pega_toxipep_") as tmp:
            runner = os.path.join(tmp, "run_toxipep.py")
            Path(runner).write_text(_RUNNER)
            out_file = os.path.join(tmp, "toxipep_out.txt")

            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "python", runner, str(fasta_path), out_file,
            ]
            # cwd is the checkout so predict.py finds its sibling modules
            # and its weights, which it opens by relative path.
            run = self._run_subprocess_with_progress(
                cmd, desc="ToxiPep", cwd=str(checkout)
            )
            if run.returncode != 0:
                raise RuntimeError(
                    f"ToxiPep failed (exit {run.returncode}).\n"
                    f"{run.stderr}\n"
                    "Note: ToxiPep rejects sequences longer than 50 residues."
                )

            if not os.path.exists(out_file):
                raise FileNotFoundError(f"ToxiPep output not found: {out_file}")
            rows = [
                line.split(",")
                for line in Path(out_file).read_text().splitlines()
                if line.strip()
            ]

        if len(rows) != len(names):
            raise RuntimeError(
                f"ToxiPep returned {len(rows)} rows for {len(names)} sequences; "
                "results cannot be matched to names by position."
            )

        scores = []
        for label, value in ((r[1], r[2]) for r in rows):
            label = int(label)
            value = float(value)
            # Undo predict.py's "confidence in the predicted class" to get
            # the probability of the toxic class.
            scores.append(value if label == 1 else 1.0 - value)

        return pd.DataFrame({"seq_name": names, "toxipep_score": scores})
