"""
pega.predictors.algpred3
========================
AlgPred 3.0 predictor — allergenicity, trained on peptides (pip).

A liability score: higher means more likely to provoke an allergic
response, so higher is worse and no ensemble consumes it.

Why alongside AlgPred2
----------------------
They are not two versions of one model, they are two training sets.
AlgPred2 learned from allergenic *proteins*, so PEGA-sized peptides sit
at the edge of what it saw; AlgPred3 was trained on curated allergenic
and non-allergenic *peptides*, which is the length range PEGA screens.
Both columns are reported rather than one replacing the other, for the
same reason HemoPI2's two models are: where they disagree, the
disagreement is informative.

Environment
-----------
Runs inside ``pega_env`` — no environment of its own. Its only declared
dependencies are pandas and joblib, and the model (an Extra Trees
classifier) was fitted under scikit-learn 1.4.2, which loads cleanly on
the ``scikit-learn >= 1.3`` that ``pega_env`` already pins. Verified
loading and scoring under scikit-learn 1.9.1 with numpy 2.4.

scikit-learn is an *undeclared* dependency of the package — it is needed
to unpickle the model but is absent from its requirements — so a bare
``pip install algpred3`` into an environment without it fails at
``joblib.load`` with ``No module named 'sklearn'``.

First run
---------
The wheel is 6 kB. On its first invocation AlgPred3 downloads ~210 MB
into its own site-packages directory: the model (163 MB), the
``pfeature_comp`` feature binary (45 MB) and the list of trained feature
columns. That needs network access and a writable site-packages, and it
happens inside whichever run touches it first — ``setup_environments.sh``
warms it up at install time so a screening run does not stop to download.

Sharding
--------
AlgPred3 costs ~0.19 s per sequence (measured over 10, 200 and 1000
sequences: ~3.5 s of fixed overhead, then a flat per-sequence cost),
which puts it just behind AMPlify as PEGA's slowest predictor. The work
is ``pfeature_comp`` computing seven feature families, and it is strictly
single-threaded: a 200-sequence run spends 41.4 s of user time in 41.6 s
of wall time. Two concurrent runs of that same file finished in 43 s
total and returned byte-identical scores, so PEGA splits the input and
runs several at once, as it already does for AMPlify.

Each shard loads its own copy of the 163 MB model and peaks around
475 MiB, which is what caps the default shard count; override it with
``PEGA_ALGPRED3_SHARDS``.

Output file
-----------
``algpred3 -j pred`` writes a CSV with the columns ``Sequence_ID``,
``Probability``, ``Status``. ``Probability`` is the [0, 1] allergen
probability and is what PEGA reads; ``Status`` is that probability cut at
``-t`` (default 0.5), which PEGA ignores.

Caveats
-------
AlgPred3 drops any sequence containing a non-canonical residue instead of
scoring it, logging it to ``stand_error.log``. PEGA validates the FASTA
upstream, and merges on sequence name, so a dropped row surfaces as a
missing value rather than a misaligned column.

Both AlgPred3 and the ``pfeature_comp`` binary it shells out to litter
the process working directory — ``tmp/``, ``stand_error.log`` and a set
of fixed-name ``sam_allcomp.*`` feature files, the last of which two
concurrent runs would overwrite for each other. Every invocation
therefore gets a temporary directory of its own, which also keeps
AlgPred3's ``tmp/`` cleanup from reaching a directory of the user's by
that name.

Reference
---------
Kumar N. et al. AlgPred3: Prediction of Allergenic Peptides and
Identification of Allergen-Specific Sequence Motifs.
https://github.com/raghavagps/algpred3
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from pega.base import BasePredictor

_SEQ_COL = "Sequence_ID"
_SCORE_COL = "Probability"

# Below this many sequences a shard spends a visible fraction of its life
# loading the 163 MB model rather than scoring, so splitting further buys
# nothing. Same reasoning, and same figure, as AMPlify's.
_MIN_SEQS_PER_SHARD = 500


def _default_shards() -> int:
    """Number of concurrent AlgPred3 subprocesses to run.

    A quarter of the cores, as for AMPlify: PEGA already runs predictors
    concurrently above this, and each shard holds ~475 MiB. Override with
    ``PEGA_ALGPRED3_SHARDS``.
    """
    override = os.environ.get("PEGA_ALGPRED3_SHARDS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, cpus // 4)


def _read_output(result_file: str) -> pd.DataFrame:
    """Parse one AlgPred3 CSV into ``seq_name`` / ``Probability`` rows."""
    if not os.path.exists(result_file):
        raise FileNotFoundError(
            f"AlgPred3 output not found: {result_file}\n"
            "Check your AlgPred3 installation."
        )

    raw = pd.read_csv(result_file)
    missing = {_SEQ_COL, _SCORE_COL} - set(raw.columns)
    if missing:
        raise RuntimeError(
            f"AlgPred3 output is missing {sorted(missing)} "
            f"(got {list(raw.columns)})."
        )
    return raw[[_SEQ_COL, _SCORE_COL]]


def _failed(returncode: int, stderr: str) -> RuntimeError:
    return RuntimeError(
        f"AlgPred3 failed (exit {returncode}).\n"
        f"{stderr}\n"
        "Check: pip install algpred3 scikit-learn\n"
        "(scikit-learn is needed to load the model but is not declared "
        "as a dependency of algpred3.)"
    )


def _run_batch(fasta_path: Path, work_dir: str, progress: bool) -> pd.DataFrame:
    """Run one AlgPred3 subprocess over ``fasta_path``, inside ``work_dir``.

    ``work_dir`` is both the output location and the process working
    directory — see the module docstring on the scratch files it drops.
    """
    result_file = os.path.join(work_dir, "algpred3_out.csv")
    cmd = [
        "algpred3",
        "-i", str(fasta_path.resolve()),
        "-o", result_file,
        "-j", "pred",
    ]

    if progress:
        run = BasePredictor._run_subprocess_with_progress(
            cmd, desc="AlgPred3", cwd=work_dir
        )
        if run.returncode != 0:
            raise _failed(run.returncode, run.stderr)
    else:
        # No progress bar on the sharded path: several bars drawing at
        # once is noise. stdout/stderr go to files rather than PIPE for
        # the deadlock reason spelled out in BasePredictor's helper —
        # AlgPred3 forks pfeature_comp, which would hold the write end.
        out_path = os.path.join(work_dir, "stdout.txt")
        err_path = os.path.join(work_dir, "stderr.txt")
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            proc = subprocess.run(
                cmd, stdout=out_f, stderr=err_f, cwd=work_dir, check=False,
            )
        if proc.returncode != 0:
            with open(err_path, errors="replace") as err_f:
                raise _failed(proc.returncode, err_f.read())

    return _read_output(result_file)


class AlgPred3Predictor(BasePredictor):
    """AlgPred 3.0 allergenicity predictor (pip, inside ``pega_env``).

    Requires::

        pip install algpred3
    """

    name = "algpred3"
    display_name = "AlgPred3"
    predictor_id = 16
    description = "Peptide allergenicity predictor (AlgPred 3.0, pip)."
    category = "pip"

    @classmethod
    def is_available(cls) -> bool:
        # A PATH check only, deliberately: "algpred3 -h" exits 0 but
        # creates a tmp/ directory in the working directory as a side
        # effect of importing its CLI module, and an availability check
        # has no business leaving that behind.
        return cls._executable_on_path("algpred3")

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score sequences with ``algpred3 -j pred``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "algpred3_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        n_shards = min(
            _default_shards(),
            max(1, len(records) // _MIN_SEQS_PER_SHARD),
        )

        with tempfile.TemporaryDirectory(prefix="pega_algpred3_") as parent:
            if n_shards <= 1:
                raw = _run_batch(fasta_path, parent, progress=True)
            else:
                raw = pd.concat(
                    self._run_shards(records, n_shards, parent),
                    ignore_index=True,
                )

        col = self.score_column()
        scores = raw.rename(
            columns={_SEQ_COL: "seq_name", _SCORE_COL: col}
        )
        scores["seq_name"] = scores["seq_name"].astype(str)
        scores[col] = pd.to_numeric(scores[col], errors="coerce")

        return scores.dropna(subset=[col]).reset_index(drop=True)

    @staticmethod
    def _run_shards(records: list, n_shards: int, parent: str) -> list[pd.DataFrame]:
        """Split ``records`` into ``n_shards`` FASTA files and score them at once."""
        shard_size = math.ceil(len(records) / n_shards)
        jobs = []
        for start in range(0, len(records), shard_size):
            # One directory per shard: the scratch files AlgPred3 and
            # pfeature_comp write have fixed names, so shards sharing a
            # directory would overwrite each other's.
            shard_dir = os.path.join(parent, f"shard_{start}")
            os.makedirs(shard_dir, exist_ok=True)
            shard_fasta = Path(shard_dir) / "shard.fasta"
            SeqIO.write(records[start:start + shard_size], str(shard_fasta), "fasta")
            jobs.append((shard_fasta, shard_dir))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [
                pool.submit(_run_batch, shard_fasta, shard_dir, False)
                for shard_fasta, shard_dir in jobs
            ]
            return [f.result() for f in futures]
