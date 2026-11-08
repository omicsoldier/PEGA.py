"""
pega.runners.ampactipred
========================
Score sequences with AMPActiPred's first-stage model.

Runs under ``ampactipred_env``, launched by
``pega.predictors.ampactipred`` through ``conda run``. It reuses the
authors' own feature code — reimplementing it would risk silently
disagreeing with what the model was trained on — but does its own
input reading and its own call into the model, for three reasons the
module docstring of the predictor spells out: ``predict.py`` does not
compile as published, its model paths are absolute and hardcoded, and it
reports a 0/1 label where PEGA needs a probability.

Usage::

    python -m pega.runners.ampactipred --repo <clone> --model-dir <dir> \\
        --fasta <in.fasta> --out <out.csv>
"""

from __future__ import annotations

import argparse
import os
import sys

#: Column written to the output CSV, matching the predictor's score column.
SCORE_COLUMN = "AMPActiPred_score"


def load_definitions(repo: str) -> dict:
    """Return ``predict.py``'s definitions, executed in a fresh namespace.

    The published ``predict.py`` does not compile: ``GAAC``, ``GDPC`` and
    ``GTPC`` were committed with their bodies flattened to column 0, loop
    and ``if``/``else`` bodies included, so the original indentation
    cannot be recovered mechanically. Those three are reached only from
    ``insert_gngram``, which the prediction path never calls, so the run
    of broken definitions is cut out instead: from ``def GAAC(`` up to
    the last column-0 ``def`` before indentation resumes.

    The script's own main block is dropped too — everything from
    ``seqlist = []`` on — since it loads models from absolute paths under
    ``/home/AMPActiPred`` and writes JSON.
    """
    path = os.path.join(repo, "predict.py")
    # newline="" keeps the file's mixed CRLF/LF endings out of the way of
    # the column-0 test below.
    with open(path, newline="") as fh:
        lines = [line.rstrip("\r") for line in fh.read().split("\n")]

    def column_zero(i: int) -> bool:
        return bool(lines[i].strip()) and not lines[i].startswith((" ", "\t"))

    try:
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("def GAAC("))
        first_indented = next(i for i in range(start + 1, len(lines))
                              if lines[i].strip() and not column_zero(i))
        end = max(i for i in range(start + 1, first_indented)
                  if lines[i].startswith("def "))
        main_start = next(i for i, line in enumerate(lines)
                          if line.startswith("seqlist = []"))
    except (StopIteration, ValueError) as exc:
        raise RuntimeError(
            f"{path} does not have the structure PEGA patches around "
            "(upstream may have fixed or rewritten it). Re-clone "
            "AMPActiPred, or report this."
        ) from exc

    source = "\n".join(lines[:start] + lines[end:main_start])
    namespace = {"__name__": "ampactipred_defs", "__file__": path}
    # predict.py parses its arguments at import time; give argparse
    # something valid so the definitions below it can be reached.
    saved_argv = sys.argv
    sys.argv = ["predict.py", "-i", "in.fasta", "-o", "out.json"]
    try:
        exec(compile(source, path, "exec"), namespace)  # noqa: S102
    finally:
        sys.argv = saved_argv

    missing = [n for n in ("construct_features",) if n not in namespace]
    if missing:
        raise RuntimeError(f"{path} defines no {missing} — unexpected layout.")
    return namespace


def read_sequences(fasta_path: str):
    """Return a ``(Id, Sequence)`` DataFrame of str, not bytes.

    Deliberately not the authors' ``read_fasta``: it reads
    ``record.seq._data``, which is ``bytes`` from Biopython 1.78 onwards,
    and every descriptor downstream (``Counter``, ``len``, n-gram counts)
    would then compute silently wrong numbers rather than fail. It also
    opens the file in ``"rU"`` mode, which Python 3.11 removed.
    """
    import pandas as pd
    from Bio import SeqIO

    rows = [(record.id, str(record.seq))
            for record in SeqIO.parse(fasta_path, "fasta")]
    if not rows:
        raise RuntimeError(f"No sequences found in {fasta_path}.")
    return pd.DataFrame(rows, columns=["Id", "Sequence"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True,
                        help="AMPActiPred checkout (holds predict.py and data/).")
    parser.add_argument("--model-dir", required=True,
                        help="First-stage model directory.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    repo = os.path.abspath(args.repo)
    fasta = os.path.abspath(args.fasta)
    out = os.path.abspath(args.out)
    model_dir = os.path.abspath(args.model_dir)

    # paac() builds its data path from os.path.realpath('__file__') — the
    # string literal, not the variable — so it resolves data/PAAC.txt
    # against the working directory. Hence the chdir, not a nicety.
    os.chdir(repo)

    namespace = load_definitions(repo)
    seq_df = read_sequences(fasta)
    features = namespace["construct_features"](seq_df)

    # The script drops the Id and Sequence columns the same way.
    matrix = features.iloc[:, 2:].values

    from deepforest import CascadeForestClassifier
    model = CascadeForestClassifier()
    model.load(model_dir)
    proba = model.predict_proba(matrix)

    if len(proba) != len(seq_df):
        raise RuntimeError(
            f"AMPActiPred scored {len(proba)} of {len(seq_df)} sequences; "
            "rows cannot be matched to names by position."
        )

    # Column 1 is P(antibacterial); the script thresholds this and reports
    # only the label.
    with open(out, "w") as fh:
        fh.write(f"seq_name,{SCORE_COLUMN}\n")
        fh.writelines(
            f"{name},{float(row[1]):.6f}\n"
            for name, row in zip(features["Id"], proba)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
