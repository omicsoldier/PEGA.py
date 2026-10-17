"""
pega.predictors.macrel
======================
Macrel predictor — SVM with physico-chemical and structural features (conda).

Output file
-----------
Macrel writes ``macrel.out.prediction.gz`` inside the output directory.
PEGA decompresses it and reads it with ``skiprows=2, header=None``.

Output columns (0-indexed, after skiprows=2)
--------------------------------------------
0  seq_name     ← used by PEGA
1  sequence
2  AMP_family
3  is_AMP label
4  AMP_probability  ← used by PEGA, as ``Macrel_score``
5  Hemolytic label
6  Hemolytic_probability  ← used by PEGA, as ``hemolysis_macrel_score``

Both probabilities come straight from the ONNX models'
``output_probability`` and are already on the [0, 1] scale PEGA expects,
so they are used as is.

``hemolysis_macrel_score`` is reported alongside the AMP score but is a
separate endpoint: it predicts haemolytic activity, not antimicrobial
activity, and no ensemble consumes it.

Indices 4 and 6 hold these two columns only because ``score`` passes
``--keep-negatives``, which keeps the ``is_AMP`` column at index 3.
Without that flag Macrel drops ``is_AMP`` and every index past 3 shifts
down by one.

Installation
------------
    conda create --name macrel_env -c bioconda macrel

Reference
---------
Santos-Júnior C.D. et al. (2020). PeerJ 8:e10555.
https://github.com/BigDataBiology/macrel
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pega.base import BasePredictor

_CONDA_ENV = "macrel_env"


class MacrelPredictor(BasePredictor):
    """Macrel SVM AMP predictor (conda).

    Requires::

        conda create --name macrel_env -c bioconda macrel
    """

    name = "macrel"
    display_name = "Macrel"
    predictor_id = 6
    description = "SVM with physico-chemical features via the Macrel tool (conda)."
    category = "conda"

    @classmethod
    def score_column(cls) -> str:
        return "Macrel_score"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_available(cls) -> bool:
        # Cached — spawns a "conda run" subprocess, and availability can't
        # change mid-process, so repeated calls (e.g. once per chunk in
        # screen_sequences) would otherwise re-spawn it needlessly.
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "macrel", "--version"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``macrel peptides``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "Macrel_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        # Scratch lives in a private temp dir. It used to be a relative
        # "temp_macrel" in the process working directory, which collided
        # between concurrent runs and, worse, was removed with rmtree in a
        # finally block — deleting a user directory that happened to share
        # the name. Macrel creates --output itself, so name a subdirectory
        # that does not exist yet.
        with tempfile.TemporaryDirectory(prefix="pega_macrel_") as scratch:
            output_dir = os.path.join(scratch, "out")
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "macrel", "peptides",
                "--fasta", str(fasta_path),
                "--output", output_dir,
                "--keep-negatives",
            ]

            result = self._run_subprocess_with_progress(cmd, desc="Macrel")
            if result.returncode != 0:
                raise RuntimeError(
                    f"Macrel failed (exit {result.returncode}).\n"
                    f"{result.stderr}\n"
                    "Check: conda env create -f envs/macrel_env.yml"
                )

            # Locate and decompress the output file
            gz_file = os.path.join(output_dir, "macrel.out.prediction.gz")
            if not os.path.exists(gz_file):
                raise FileNotFoundError(
                    f"Macrel output not found: {gz_file}\n"
                    "Check your Macrel installation."
                )

            subprocess.run(["gzip", "-d", gz_file], check=True)

            result_file = os.path.join(output_dir, "macrel.out.prediction")
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"Decompressed file not found: {result_file}")

            # skiprows=2 skips the comment line and column header
            # Columns: 0=seq_name, 1=sequence, 2=AMP_family, 3=is_AMP,
            # 4=AMP_probability, 5=Hemolytic, 6=Hemolytic_probability
            # (indices 4 and 6 rely on --keep-negatives above)
            raw = pd.read_csv(result_file, sep="\t", skiprows=2, header=None)
            scores = raw.iloc[:, [0, 4, 6]].rename(columns={
                0: "seq_name",
                4: "Macrel_score",
                6: "hemolysis_macrel_score",
            })
            for col in ("Macrel_score", "hemolysis_macrel_score"):
                scores[col] = pd.to_numeric(scores[col], errors="coerce")

            # Only the AMP score decides whether a row is usable; a row with
            # a valid AMP score but an unparsable hemolysis value keeps the
            # former and carries NaN for the latter.
            return scores.dropna(subset=["Macrel_score"]).reset_index(drop=True)
