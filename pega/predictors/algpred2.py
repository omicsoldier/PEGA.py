"""
pega.predictors.algpred2
========================
AlgPred 2.0 predictor — allergenicity (conda).

A liability score: higher means more likely to provoke an allergic
response, so higher is worse and no ensemble consumes it. It answers a
third question, separate from haemolysis (does it lyse red blood cells)
and toxicity (does it look like a venom or bacterial toxin).

Environment
-----------
Shares ``tox_env`` with ToxinPred3 rather than taking one of its own:
both are Raghava-group packages whose pickled models need
``scikit-learn==1.0.2`` and ``numpy==1.22.4``. Verified that installing
both together leaves ToxinPred3's scores unchanged.

Model
-----
PEGA runs the plain machine-learning model (``-m 1``, the default).
``-m 2`` is a hybrid that folds in MERCI motif hits and a BLAST search,
the same construction PEGA avoids in HemoPI2 and ToxinPred3: it is built
for labelling at a threshold and flattens the score a ranked screen
needs.

Output file
-----------
``algpred2`` writes a CSV with the columns ``ID``, ``Sequence``,
``ML_Score``, ``Prediction``. ``ML_Score`` is a [0, 1] probability; the
tool's own allergen cutoff is 0.3.

Caveats
-------
``-d 2`` is passed explicitly. AlgPred2 defaults to ``-d 1``, which
returns only the rows it calls allergenic — that would silently drop
every non-allergenic sequence.

AlgPred2 writes a scratch file named ``Sequence_1`` into the process
working directory and does not clean it up, so PEGA runs it inside a
temporary directory.

AlgPred2 was built for allergenic *proteins*; short peptides are near
the edge of what it was trained on, so read the score as a flag to
follow up rather than a verdict.

Installation
------------
    conda env create -f envs/tox_env.yml

Reference
---------
Sharma N. et al. (2021). Front Immunol 12:601980.
https://github.com/raghavagps/algpred2
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pega.base import BasePredictor

_CONDA_ENV = "tox_env"

_SCORE_COL = "ML_Score"


class AlgPred2Predictor(BasePredictor):
    """AlgPred 2.0 allergenicity predictor (conda).

    Requires::

        conda env create -f envs/tox_env.yml
    """

    name = "algpred2"
    display_name = "AlgPred2"
    predictor_id = 14
    description = "Allergenicity predictor (AlgPred 2.0, conda)."
    category = "conda"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_available(cls) -> bool:
        # Cached for the same reason as Macrel's: this spawns a "conda run"
        # subprocess and availability cannot change mid-process.
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "algpred2", "-h"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``algpred2``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "algpred2_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        with tempfile.TemporaryDirectory(prefix="pega_algpred2_") as work_dir:
            result_file = os.path.join(work_dir, "algpred2_out.csv")
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "algpred2",
                "-i", str(fasta_path),
                "-o", result_file,
                "-d", "2",
            ]

            # cwd is the temp dir so the stray "Sequence_1" scratch file
            # lands there and goes away with it.
            run = self._run_subprocess_with_progress(
                cmd, desc="AlgPred2", cwd=work_dir
            )
            if run.returncode != 0:
                raise RuntimeError(
                    f"AlgPred2 failed (exit {run.returncode}).\n"
                    f"{run.stderr}\n"
                    "Check: conda env create -f envs/tox_env.yml"
                )

            if not os.path.exists(result_file):
                raise FileNotFoundError(
                    f"AlgPred2 output not found: {result_file}\n"
                    "Check your AlgPred2 installation."
                )

            raw = pd.read_csv(result_file)

        if _SCORE_COL not in raw.columns:
            raise RuntimeError(
                f"AlgPred2 output has no '{_SCORE_COL}' column "
                f"(got {list(raw.columns)})."
            )

        scores = raw[["ID", _SCORE_COL]].rename(
            columns={"ID": "seq_name", _SCORE_COL: "algpred2_score"}
        )
        scores["seq_name"] = scores["seq_name"].astype(str)
        scores["algpred2_score"] = pd.to_numeric(
            scores["algpred2_score"], errors="coerce"
        )

        return scores.dropna(subset=["algpred2_score"]).reset_index(drop=True)
