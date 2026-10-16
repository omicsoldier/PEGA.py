"""
pega.predictors.hemopi2
=======================
HemoPI2 predictor — haemolytic activity against mammalian RBCs (conda).

This scores a different endpoint from every other PEGA predictor: it is
the probability that a peptide lyses red blood cells, so **higher is
worse**. It is a toxicity flag for triaging candidates, not an activity
score, and no ensemble consumes it.

Model
-----
PEGA runs the Random Forest classifier (``-m 1``), which reaches AUC
0.888 on the authors' independent dataset. HemoPI2 also ships ESM2-t6
and MERCI-hybrid models that score higher (0.919 and 0.921), at the cost
of a much heavier environment and far slower per-sequence inference.

Output file
-----------
``hemopi2_classification`` writes a CSV into its ``-wd`` directory with
the columns ``SeqID``, ``Sequence``, ``ML Score``, ``Prediction``.
``ML Score`` is a [0, 1] probability rounded to 3 decimals; PEGA reads
it as is. ``SeqID`` already has its leading ``>`` stripped.

Caveats
-------
``-d 2`` is passed explicitly: under ``-d 1`` HemoPI2 keeps only the rows
it calls haemolytic, which would silently drop sequences.

HemoPI2 truncates any sequence longer than 40 residues to its first 40
before scoring, so long peptides are scored on a prefix. Row count is
unaffected.

Installation
------------
    conda env create -f envs/hemo_env.yml

Reference
---------
Rathore A.S. et al. (2025). Commun Biol 8:176.
https://github.com/raghavagps/hemopi2
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pega.base import BasePredictor

_CONDA_ENV = "hemo_env"

# -m 1 selects the Random Forest classifier; see the module docstring for
# why the heavier ESM2/MERCI models are not the default.
_MODEL_RF = "1"


class HemoPI2Predictor(BasePredictor):
    """HemoPI2 Random Forest haemolysis predictor (conda).

    Requires::

        conda env create -f envs/hemo_env.yml
    """

    name = "hemopi2"
    display_name = "HemoPI2 (RF)"
    predictor_id = 11
    description = "Random Forest haemolytic-activity predictor (HemoPI2, conda)."
    category = "conda"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_available(cls) -> bool:
        # Cached for the same reason as Macrel's: this spawns a "conda run"
        # subprocess and availability cannot change mid-process.
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "hemopi2_classification", "-h"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``hemopi2_classification``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "hemopi2_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        with tempfile.TemporaryDirectory(prefix="pega_hemopi2_") as work_dir:
            # -o takes a bare filename: HemoPI2 writes it inside -wd, so an
            # absolute path here would be appended to the working directory.
            result_name = "hemopi2_out.csv"
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "hemopi2_classification",
                "-i", str(fasta_path),
                "-wd", work_dir,
                "-o", result_name,
                "-j", "1",
                "-m", _MODEL_RF,
                "-d", "2",
            ]

            run = self._run_subprocess_with_progress(cmd, desc="HemoPI2")
            if run.returncode != 0:
                raise RuntimeError(
                    f"HemoPI2 failed (exit {run.returncode}).\n"
                    f"{run.stderr}\n"
                    "Check: conda env create -f envs/hemo_env.yml"
                )

            result_file = os.path.join(work_dir, result_name)
            if not os.path.exists(result_file):
                raise FileNotFoundError(
                    f"HemoPI2 output not found: {result_file}\n"
                    "Check your HemoPI2 installation."
                )

            raw = pd.read_csv(result_file)

        scores = raw[["SeqID", "ML Score"]].rename(
            columns={"SeqID": "seq_name", "ML Score": "hemopi2_score"}
        )
        scores["seq_name"] = scores["seq_name"].astype(str)
        scores["hemopi2_score"] = pd.to_numeric(
            scores["hemopi2_score"], errors="coerce"
        )

        return scores.dropna(subset=["hemopi2_score"]).reset_index(drop=True)
