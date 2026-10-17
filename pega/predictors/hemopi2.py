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

import abc
import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from pega.base import BasePredictor

_CONDA_ENV = "hemo_env"

# -m 1 selects the Random Forest classifier; see the module docstring for
# why the heavier ESM2/MERCI models are not the default.
_MODEL_RF = "1"
_MODEL_ESM = "3"


class _HemoPI2Base(BasePredictor, abc.ABC):
    """Shared plumbing for HemoPI2's two non-hybrid models."""

    category = "conda"

    #: HemoPI2 -m value, and the score column its CSV carries.
    _model: str
    _csv_column: str
    #: Sequences per invocation, or None to pass the whole file at once.
    _max_batch: int | None = None

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

        if self._max_batch is None:
            return self._score_batch(fasta_path)

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if len(records) <= self._max_batch:
            return self._score_batch(fasta_path)

        # Sequentially, on purpose: the point is to bound peak memory, so
        # running the shards concurrently would defeat it.
        frames = []
        with tempfile.TemporaryDirectory(prefix="pega_hemopi2_shards_") as shards:
            for start in range(0, len(records), self._max_batch):
                shard = os.path.join(shards, f"shard_{start}.fasta")
                SeqIO.write(records[start:start + self._max_batch], shard, "fasta")
                frames.append(self._score_batch(Path(shard)))
        return pd.concat(frames, ignore_index=True)

    def _score_batch(self, fasta_path: Path) -> pd.DataFrame:
        """Run HemoPI2 once over every sequence in ``fasta_path``."""
        with tempfile.TemporaryDirectory(prefix="pega_hemopi2_") as work_dir:
            # -o takes a bare filename: HemoPI2 writes it inside -wd, so an
            # absolute path here would be appended to the working directory.
            result_name = f"{self.name}_out.csv"
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "hemopi2_classification",
                "-i", str(fasta_path),
                "-wd", work_dir,
                "-o", result_name,
                "-j", "1",
                "-m", self._model,
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

        col = self.score_column()
        scores = raw[["SeqID", self._csv_column]].rename(
            columns={"SeqID": "seq_name", self._csv_column: col}
        )
        scores["seq_name"] = scores["seq_name"].astype(str)
        scores[col] = pd.to_numeric(scores[col], errors="coerce")

        return scores.dropna(subset=[col]).reset_index(drop=True)


class HemoPI2Predictor(_HemoPI2Base):
    """HemoPI2 Random Forest haemolysis predictor (conda)."""

    name = "hemopi2"
    display_name = "HemoPI2 (RF)"
    predictor_id = 11
    description = "Random Forest haemolytic-activity predictor (HemoPI2, conda)."
    _model = _MODEL_RF
    _csv_column = "ML Score"


class HemoPI2ESMPredictor(_HemoPI2Base):
    """HemoPI2 ESM2-t6 haemolysis predictor (conda).

    Reported alongside the Random Forest rather than instead of it: the
    two agree only moderately (Spearman 0.66 across a 25-peptide panel),
    so the disagreement carries information. Where they differ on
    low-complexity sequences, trust the RF — ESM2 called poly-Gly,
    poly-Ser and poly-Ala haemolytic, which a genetic algorithm will
    produce in quantity.
    """

    name = "hemopi2_esm"
    display_name = "HemoPI2 (ESM2)"
    predictor_id = 15
    description = "ESM2-t6 haemolytic-activity predictor (HemoPI2, conda)."
    _model = _MODEL_ESM
    _csv_column = "ESM Score"
    # HemoPI2 tokenises the whole input file and pushes it through ESM2 in
    # a single forward pass, with no batching of its own and the device
    # hardcoded to CUDA when one is present. Peak memory therefore scales
    # with the file, and a screening chunk of 10k sequences exhausts an
    # 8 GB card outright. Feeding it in fixed slices bounds that; the RF
    # model needs none of this and keeps _max_batch = None.
    _max_batch = 512
