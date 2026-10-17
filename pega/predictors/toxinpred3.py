"""
pega.predictors.toxinpred3
==========================
ToxinPred3 predictor — general peptide toxicity (conda).

Like the haemolysis predictors this scores a liability rather than an
activity, so **higher is worse** and no ensemble consumes it.

What "toxic" means here
-----------------------
ToxinPred3 is trained to recognise *toxins* — venom and bacterial toxin
peptides — not mammalian-cell safety. That is a different question from
haemolysis, and the two disagree in practice: magainin2, a benign frog
AMP, scores 1.00 here while scoring 0.72 on HemoPI2, and melittin scores
0.82 here against 0.97 on HemoPI2. Many membrane-active AMPs sit in
ToxinPred's positive set, so a high score on a designed AMP is expected
and is not on its own evidence of a safety problem. Read it next to
the haemolysis columns, not instead of them.

Model
-----
PEGA runs the plain Extra Trees model (``-m 1``). ToxinPred3 defaults to
``-m 2``, a hybrid that adds MERCI motif hits to the ML score, which PEGA
deliberately avoids: a negative motif subtracts 0.5 outright, which flips
melittin — bee venom — from 0.82 to 0.32 and labels it non-toxic. The
hybrid also collapses to identical values for the many peptides that
match no motif, destroying the ranking a screen depends on.

Output file
-----------
``toxinpred3`` writes a CSV with the columns ``ID``, ``Sequence``,
``ML Score``, ``Prediction``, ``PPV``. ``ML Score`` is a [0, 1]
probability; PEGA reads it as is. ``ID`` has no leading ``>``.

Caveats
-------
``-d 2`` is passed explicitly: under ``-d 1`` ToxinPred3 returns only the
rows it calls toxic, which would silently drop sequences.

ToxinPred3 writes a scratch file named ``Sequence_1`` into the process
working directory and does not clean it up, so PEGA runs it inside a
temporary directory.

Installation
------------
    conda env create -f envs/tox_env.yml

Reference
---------
Rathore A.S. et al. (2024). Comput Biol Med 179:108926.
https://github.com/raghavagps/toxinpred3
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

# -m 1 selects the plain Extra Trees model; see the module docstring for
# why the MERCI hybrid that ToxinPred3 defaults to is not used.
_MODEL_ET = "1"


class ToxinPred3Predictor(BasePredictor):
    """ToxinPred3 Extra Trees toxicity predictor (conda).

    Requires::

        conda env create -f envs/tox_env.yml
    """

    name = "toxinpred3"
    display_name = "ToxinPred3 (ET)"
    predictor_id = 12
    description = "Extra Trees toxicity predictor (ToxinPred3, conda)."
    category = "conda"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_available(cls) -> bool:
        # Cached for the same reason as Macrel's: this spawns a "conda run"
        # subprocess and availability cannot change mid-process.
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "toxinpred3", "-h"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``toxinpred3``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "toxinpred3_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        with tempfile.TemporaryDirectory(prefix="pega_toxinpred3_") as work_dir:
            result_file = os.path.join(work_dir, "toxinpred3_out.csv")
            cmd = [
                "conda", "run", "-n", _CONDA_ENV,
                "toxinpred3",
                "-i", str(fasta_path),
                "-o", result_file,
                "-m", _MODEL_ET,
                "-d", "2",
            ]

            # cwd is the temp dir so the stray "Sequence_1" scratch file
            # lands there and goes away with it.
            run = self._run_subprocess_with_progress(
                cmd, desc="ToxinPred3", cwd=work_dir
            )
            if run.returncode != 0:
                raise RuntimeError(
                    f"ToxinPred3 failed (exit {run.returncode}).\n"
                    f"{run.stderr}\n"
                    "Check: conda env create -f envs/tox_env.yml"
                )

            if not os.path.exists(result_file):
                raise FileNotFoundError(
                    f"ToxinPred3 output not found: {result_file}\n"
                    "Check your ToxinPred3 installation."
                )

            raw = pd.read_csv(result_file)

        scores = raw[["ID", "ML Score"]].rename(
            columns={"ID": "seq_name", "ML Score": "toxinpred3_score"}
        )
        scores["seq_name"] = scores["seq_name"].astype(str)
        scores["toxinpred3_score"] = pd.to_numeric(
            scores["toxinpred3_score"], errors="coerce"
        )

        return scores.dropna(subset=["toxinpred3_score"]).reset_index(drop=True)
