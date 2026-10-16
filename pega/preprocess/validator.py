"""
pega.preprocess.validator
=========================
FASTA sequence validation for PEGA.
"""

from __future__ import annotations

from pathlib import Path

from Bio import SeqIO

_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
_AMBIGUOUS = set("BJOUXZ")
_MIN_LEN = 10


def _fasta_stats_from_path(fasta_path: Path) -> dict:
    """Return length statistics for sequences in a FASTA file."""
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    lengths = [len(r.seq) for r in records]
    return {
        "n": len(records),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }


def validate_fasta(
    fasta_path: str | Path,
    verbose: bool = True,
) -> tuple[bool, str]:
    """Validate a FASTA file for use with PEGA predictors.

    Parameters
    ----------
    fasta_path:
        Path to the FASTA file to validate.
    verbose:
        If ``True``, print a full report. If ``False``, return a one-line
        status string only.

    Returns
    -------
    tuple[bool, str]
        ``(valid, status_message)`` — ``valid`` is ``False`` only when
        critical errors (non-ASCII or invalid characters) are found.
    """
    path = Path(fasta_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return False, f"File not found: {path}"

    if path.stat().st_size == 0:
        return False, "File is empty."

    try:
        path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        return False, f"Non-ASCII characters detected: {exc}"

    records = list(SeqIO.parse(str(path), "fasta"))
    if not records:
        return False, "No sequences found. Check FASTA format."

    # Duplicate seq_names
    ids = [r.id for r in records]
    dup_ids = [i for i in set(ids) if ids.count(i) > 1]

    short, noncanonical, ambiguous, stop_codon = [], {}, {}, []

    for rec in records:
        seq = str(rec.seq).upper()
        if seq.endswith("*"):
            stop_codon.append(rec.id)
            seq = seq[:-1]
        if len(seq) < _MIN_LEN:
            short.append(rec.id)
        chars = set(seq)
        bad = chars - _CANONICAL - _AMBIGUOUS - {"*"}
        amb = chars & _AMBIGUOUS
        if bad:
            noncanonical[rec.id] = bad
        if amb:
            ambiguous[rec.id] = amb

    if noncanonical:
        chars = sorted({c for v in noncanonical.values() for c in v})
        errors.append(
            f"{len(noncanonical)} sequence(s) with invalid characters {chars}"
        )

    if dup_ids:
        warnings.append(
            f"{len(dup_ids)} duplicate sequence name(s) — PEGA renames them "
            "internally, so results will not carry the original name twice"
        )
    if short:
        warnings.append(f"{len(short)} sequence(s) < {_MIN_LEN} residues (may return NaN)")
    if ambiguous:
        chars = sorted({c for v in ambiguous.values() for c in v})
        warnings.append(f"{len(ambiguous)} sequence(s) with ambiguous AAs {chars}")
    if stop_codon:
        warnings.append(f"{len(stop_codon)} sequence(s) with trailing stop codon (*)")

    valid = len(errors) == 0

    # Build status line
    if errors:
        status = "INVALID — " + "; ".join(errors)
    elif warnings:
        status = "OK (warnings) — " + "; ".join(warnings)
    else:
        status = "OK — all sequences are valid."

    if verbose:
        lengths = [len(r.seq) for r in records]
        n = len(records)
        print(f"  File       : {path.name}")
        print(f"  Sequences  : {n}  "
              f"(len: {min(lengths)}–{max(lengths)}, avg {round(sum(lengths)/n,1)})")
        for w in warnings:
            print(f"  [WARNING]  {w}")
        for e in errors:
            print(f"  [ERROR]    {e}")
        print(f"  Status     : {status}\n")

    return valid, status
