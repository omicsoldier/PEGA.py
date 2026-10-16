"""
pega.utils
==========
Main scoring orchestrator for PEGA.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from pega.registry import registry

_CANONICAL_SET = set("ACDEFGHIKLMNPQRSTVWY")
_console = Console()

# Liability columns: these score a risk (haemolysis, toxicity) rather than
# antimicrobial activity, so higher is worse. They are moved to the end of
# the output so the activity scores and ensembles read as one block. Add
# new haemolysis or toxicity columns here.
_LIABILITY_COLUMNS = (
    "hemolysis_macrel_score",
    "hemopi2_score",
    "toxinpred3_score",
)


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------


def _count_fasta_seqs(fasta_path: Path) -> int:
    """Count sequences in a FASTA file with a fast single-pass line scan."""
    with open(fasta_path, "r", errors="replace") as f:
        return sum(1 for line in f if line.startswith(">"))


def _prepare_fasta(fasta_path: Path) -> tuple[Path, dict]:
    """Assign sequential IDs and return a temp FASTA + id_map.

    Returns
    -------
    tuple[Path, dict[str, dict]]
        (temp_fasta_path, {pep_id: {original_header, length}})
    """
    import tempfile
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    width   = len(str(len(records)))
    id_map  = {}
    new_recs = []

    for i, rec in enumerate(records, 1):
        pep_id   = f"pep_{i:0{width}d}"
        original = rec.description if rec.description != rec.id else rec.id
        id_map[pep_id] = {"original_header": original, "length": len(rec.seq)}
        new_recs.append(SeqRecord(rec.seq, id=pep_id, description=""))

    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        n_dup = len(ids) - len(set(ids))
        print(f"  [info] {n_dup} duplicate seq_name(s) → assigned sequential IDs.",
              file=sys.stderr)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, prefix="pega_prep_"
    )
    SeqIO.write(new_recs, tmp, "fasta")
    tmp.close()
    return Path(tmp.name), id_map


def _iter_fasta_chunks(fasta_path: Path, chunk_size: int):
    """Yield (valid_records, id_map, rejected) per chunk.

    Validates canonical AAs lazily. Non-canonical sequences go to
    ``rejected`` and never receive a pep_id.

    Yields
    ------
    tuple[list[SeqRecord], dict[str, dict], list[dict]]
        - valid_records : SeqRecords with pep_XXXXXXXX IDs.
        - id_map        : {pep_id: {original_header, length}}.
        - rejected      : [{original_header, sequence, length, reason}].
    """
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord

    counter  = 0
    valid:    list = []
    id_map:   dict = {}
    rejected: list = []

    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        seq      = str(rec.seq).upper()
        original = rec.description if rec.description != rec.id else rec.id
        invalid  = set(seq) - _CANONICAL_SET

        if invalid:
            rejected.append({
                "original_header": original,
                "sequence":        seq,
                "length":          len(seq),
                "reason":          f"non_canonical_aa:{sorted(invalid)}",
            })
        else:
            counter += 1
            pep_id  = f"pep_{counter:08d}"
            id_map[pep_id] = {"original_header": original, "length": len(seq)}
            valid.append(SeqRecord(rec.seq, id=pep_id, description=""))

        if len(valid) >= chunk_size:
            yield valid, id_map, rejected
            valid    = []
            id_map   = {}
            rejected = []

    if valid or rejected:
        yield valid, id_map, rejected


def _progress(current: int, total: int, name: str,
              elapsed: float | None = None) -> None:
    width    = len(str(total))
    time_str = f"  {elapsed:.1f}s" if elapsed is not None else ""
    if name.endswith(" ✓"):
        name = name[:-2] + " [green]✓[/green]"
    elif name.endswith(" ✗"):
        name = name[:-2] + " [red]✗[/red]"
    _console.print(f"  \\[{current:{width}}/{total}]  {name}{time_str}")


# ---------------------------------------------------------------------------
# Screening (chunked, 1M+ sequences)
# ---------------------------------------------------------------------------


def screen_sequences(
    fasta_path: str | Path,
    output_dir: str | Path | None = None,
    chunk_size: int = 50_000,
    predictor_names: list[str] | None = None,
    ensemble_names: list[str] | None = None,
    no_individual_scores: bool = False,
    jobs: int = -1,
) -> dict[str, Path]:
    """Score a large FASTA file in chunks — optimised for 1M+ sequences.

    Output folder (``<output_dir>/``) contains:

    ``names.tsv``
        ``pep_id`` | ``original_header`` | ``length``.
        Maps sequential IDs to original FASTA headers.

    ``rejected.tsv``
        ``original_header`` | ``length`` | ``reason``.
        Sequences excluded due to non-canonical amino acids.

    ``rejected.fasta``
        FASTA file of the rejected sequences.

    ``scores.tsv``
        ``pep_id`` | predictor scores | ensemble scores.
        Join with ``names.tsv`` on ``pep_id`` to recover original headers.

    Parameters
    ----------
    fasta_path : str or Path
        Input FASTA file (any size).
    output_dir : str or Path or None
        Output directory. Default: ``PEGA_screen_YYYYMMDD_HHMMSS/``.
    chunk_size : int
        Valid sequences per chunk. Default 50,000.
        Reduce to 10,000 if RAM < 16 GB.
    predictor_names : list[str] or None
        Predictors to run. ``None`` = all available.
    ensemble_names : list[str] or None
        Ensembles to compute. ``None`` = all.
        Auto-selects required predictors when specified.
    no_individual_scores : bool
        Keep only ensemble columns in ``scores.tsv``.
    jobs : int
        Parallel workers. Default -1 = 75%% of CPU threads.

    Returns
    -------
    dict[str, Path]
        Keys: ``"names"``, ``"rejected_tsv"``, ``"rejected_fasta"``,
        ``"scores"``, ``"dir"``.
    """
    import tempfile
    from Bio import SeqIO as _SeqIO
    from Bio.SeqRecord import SeqRecord as _SeqRecord
    from Bio.Seq import Seq as _Seq

    fasta_path = Path(fasta_path)

    # ── Output folder ──────────────────────────────────────────────────
    if output_dir is None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"PEGA_screen_{timestamp}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_names    = output_dir / "names.tsv"
    out_rej_tsv  = output_dir / "rejected.tsv"
    out_rej_fasta= output_dir / "rejected.fasta"
    out_scores   = output_dir / "scores.tsv"

    # ── Resolve predictor set ──────────────────────────────────────────
    # Only auto-restrict predictors when specific ensembles were requested;
    # ensemble_names == [] means "skip all ensembles", not "skip all predictors".
    if ensemble_names and predictor_names is None:
        from pega.ensemble import required_predictors
        needed_cols     = required_predictors(ensemble_names)
        predictor_names = [
            cls.name for cls in registry.list_available()
            if cls.score_column() in needed_cols
        ]

    # ── Header ────────────────────────────────────────────────────────
    print(f"  Input     : {fasta_path.name}")
    print(f"  Output    : {output_dir}/")
    print(f"  Chunk size: {chunk_size:,} valid sequences")
    print()

    print(f"  Counting sequences ...", end=" ", flush=True)
    total_in_file = _count_fasta_seqs(fasta_path)
    print(f"{total_in_file:,}")
    print()

    headers_written = {"names": False, "rej_tsv": False, "scores": False}
    total_valid     = 0
    total_rejected  = 0
    chunk_num       = 0
    t_start         = time.perf_counter()
    chunk_times: list[float] = []

    for valid_records, id_map, rejected in _iter_fasta_chunks(fasta_path, chunk_size):

        # ── names.tsv — pep_id | original_header | length ─────────────
        if id_map:
            names_df = pd.DataFrame([
                {"pep_id": pid,
                 "original_header": v["original_header"],
                 "length": v["length"]}
                for pid, v in id_map.items()
            ])
            names_df.to_csv(
                out_names, sep="\t", index=False,
                mode="a" if headers_written["names"] else "w",
                header=not headers_written["names"],
            )
            headers_written["names"] = True

        # ── rejected.tsv — original_header | length | reason ──────────
        if rejected:
            rej_df = pd.DataFrame([
                {"original_header": r["original_header"],
                 "length":          r["length"],
                 "reason":          r["reason"]}
                for r in rejected
            ])
            rej_df.to_csv(
                out_rej_tsv, sep="\t", index=False,
                mode="a" if headers_written["rej_tsv"] else "w",
                header=not headers_written["rej_tsv"],
            )
            headers_written["rej_tsv"] = True

            # rejected.fasta
            rej_recs = [
                _SeqRecord(_Seq(r["sequence"]),
                           id=r["original_header"].split()[0],
                           description=" ".join(r["original_header"].split()[1:]))
                for r in rejected
            ]
            with open(out_rej_fasta, "a") as fh:
                _SeqIO.write(rej_recs, fh, "fasta")

            total_rejected += len(rejected)

        if not valid_records:
            continue

        # ── Score chunk ────────────────────────────────────────────────
        chunk_num += 1
        n_chunk    = len(valid_records)
        t0         = time.perf_counter()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False, prefix="pega_chunk_"
        ) as tmp:
            _SeqIO.write(valid_records, tmp, "fasta")
            tmp_path = Path(tmp.name)

        try:
            df = calculate_scores(
                fasta_path=tmp_path,
                predictor_names=predictor_names,
                ensemble_names=ensemble_names,
                no_individual_scores=no_individual_scores,
                jobs=jobs,
                validate=False,
                quiet=True,
                export_tsv=None,
                _skip_prepare=True,
                _no_export=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # scores.tsv — pep_id only (no original_header)
        if "original_header" in df.columns:
            df = df.drop(columns=["original_header"])

        df.to_csv(
            out_scores, sep="\t", index=False,
            mode="a" if headers_written["scores"] else "w",
            header=not headers_written["scores"],
        )
        headers_written["scores"] = True
        total_valid += n_chunk

        elapsed = time.perf_counter() - t0
        chunk_times.append(elapsed)

        throughput = n_chunk / elapsed
        avg_time   = sum(chunk_times[-5:]) / len(chunk_times[-5:])
        seqs_done  = total_valid + total_rejected
        seqs_left  = max(0, total_in_file - seqs_done)
        eta_s      = avg_time * (seqs_left / max(chunk_size, 1))

        if   eta_s < 60:    eta_str = f"{eta_s:.0f}s"
        elif eta_s < 3600:  eta_str = f"{eta_s/60:.1f}min"
        else:                eta_str = f"{eta_s/3600:.1f}h"

        pct = 100 * seqs_done / total_in_file
        print(
            f"  Chunk {chunk_num:>4}  "
            f"valid={total_valid:>8,}  rejected={total_rejected:>6,}  "
            f"({pct:5.1f}%)  {throughput:,.0f} seq/s  ETA {eta_str}"
        )

    wall = time.perf_counter() - t_start
    print(f"\n  Done in {wall:.1f}s  ({wall/60:.1f} min)")
    print(f"  Valid    : {total_valid:,}  → {out_scores.name}")
    print(f"  Rejected : {total_rejected:,} → {out_rej_tsv.name} + {out_rej_fasta.name}")
    print(f"  Names    : {total_valid:,}  → {out_names.name}")
    print(f"  Folder   : {output_dir}/")

    return {
        "names":          out_names,
        "rejected_tsv":   out_rej_tsv,
        "rejected_fasta": out_rej_fasta,
        "scores":         out_scores,
        "dir":            output_dir,
    }


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def calculate_scores(
    fasta_path: str | Path,
    predictor_names: list[str] | None = None,
    export_tsv: str | Path | None = None,
    jobs: int = 1,
    validate: bool = True,
    quiet: bool = False,
    ensemble_names: list[str] | None = None,
    no_individual_scores: bool = False,
    _skip_prepare: bool = False,
    _no_export: bool = False,
) -> pd.DataFrame:
    """Score all sequences in a FASTA file using available AMP predictors."""
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # ── Validation + header ────────────────────────────────────────────
    if validate:
        from pega.preprocess.validator import validate_fasta, _fasta_stats_from_path
        stats = _fasta_stats_from_path(fasta_path)
        predictor_label = ("all available" if predictor_names is None
                           else ", ".join(predictor_names))
        print(f"  Input      : {fasta_path.name}")
        print(f"  Sequences  : {stats['n']}  "
              f"(len: {stats['min_len']}–{stats['max_len']}, avg {stats['avg_len']})")
        print(f"  Predictors : {predictor_label}")
        print(f"  Jobs       : {jobs}")
        out_label = export_tsv if export_tsv else "PEGA_results_<timestamp>.tsv"
        print(f"  Output     : {out_label}")
        ok, status_msg = validate_fasta(fasta_path, verbose=False)
        print(f"  Status     : {status_msg}")
        print()
        if not ok:
            raise ValueError(
                "FASTA file contains invalid sequences. "
                "Run 'PEGA validate --fasta ...' for details."
            )

    # ── FASTA preparation ──────────────────────────────────────────────
    if _skip_prepare:
        _tmp_fasta = None
        _id_map    = {}
    else:
        _tmp_fasta, _id_map = _prepare_fasta(fasta_path)
        fasta_path = _tmp_fasta

    # ── Resolve predictor set ──────────────────────────────────────────
    # Only auto-restrict predictors when specific ensembles were requested;
    # ensemble_names == [] means "skip all ensembles", not "skip all predictors".
    if ensemble_names and predictor_names is None:
        from pega.ensemble import required_predictors
        needed_cols     = required_predictors(ensemble_names)
        predictor_names = [
            cls.name for cls in registry.list_available()
            if cls.score_column() in needed_cols
        ]
        if not quiet:
            print(f"  Auto-selected {len(predictor_names)} predictors "
                  f"for ensembles {ensemble_names}")

    if predictor_names is not None:
        predictor_classes = []
        for name in predictor_names:
            cls = registry.get(name)
            if not cls.is_available():
                print(f"  [skip] '{cls.display_name}' not available.",
                      file=sys.stderr)
                continue
            predictor_classes.append(cls)
    else:
        predictor_classes = registry.list_available()

    if not predictor_classes:
        raise ValueError("No predictors available. Run 'PEGA list'.")

    total = len(predictor_classes)

    # ── Run predictors ─────────────────────────────────────────────────
    _cpus       = os.cpu_count() or 1
    max_workers = max(1, int(_cpus * 0.75)) if jobs == -1 else max(1, jobs)
    dfs:    dict[str, pd.DataFrame] = {}
    errors: dict[str, str]          = {}

    # Say how many of these measure a liability rather than activity —
    # "12 predictors" otherwise reads as 12 activity scores. Announced for
    # both paths; only the parallel one used to print anything.
    n_liability = sum(
        1 for cls in predictor_classes
        if cls.score_column() in _LIABILITY_COLUMNS
    )
    split = (
        f" ([bold]{total - n_liability}[/bold] activity, "
        f"[bold]{n_liability}[/bold] liability)" if n_liability else ""
    )
    how = (
        "sequentially" if max_workers == 1
        else f"with [bold]{max_workers}[/bold] parallel workers"
    )
    noun = "predictor" if total == 1 else "predictors"
    _console.print(f"  Running [bold]{total}[/bold] {noun}{split} {how}...")
    print()

    if max_workers == 1:
        for i, cls in enumerate(predictor_classes, 1):
            _progress(i, total, cls.display_name)
            t0 = time.perf_counter()
            try:
                dfs[cls.name] = cls().score(fasta_path)
            except Exception as exc:
                errors[cls.display_name] = str(exc)
                print(f"  [warning] {cls.display_name} failed "
                      f"({time.perf_counter()-t0:.1f}s): {exc}", file=sys.stderr)
    else:
        import tqdm as _tqdm_module
        _orig_init = _tqdm_module.tqdm.__init__

        def _silent_init(self, *a, **kw):
            kw["disable"] = True
            _orig_init(self, *a, **kw)

        _tqdm_module.tqdm.__init__ = _silent_init
        # Prevent joblib/loky conflicts with scikit-learn inside threads
        os.environ["LOKY_MAX_CPU_COUNT"] = "1"
        completed = 0

        def _run(cls):
            t0 = time.perf_counter()
            return cls.name, cls().score(fasta_path), time.perf_counter() - t0

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run, cls): cls
                           for cls in predictor_classes}
                for future in as_completed(futures):
                    cls = futures[future]
                    completed += 1
                    try:
                        name, df, elapsed = future.result()
                        dfs[name] = df
                        _progress(completed, total,
                                  f"{cls.display_name} ✓", elapsed)
                    except Exception as exc:
                        errors[cls.display_name] = str(exc)
                        _progress(completed, total,
                                  f"{cls.display_name} ✗")
                        print(f"       failed: {exc}", file=sys.stderr)
        finally:
            _tqdm_module.tqdm.__init__ = _orig_init

    if not dfs:
        raise RuntimeError("All predictors failed.")

    # ── Merge ─────────────────────────────────────────────────────────
    ordered = [dfs[cls.name] for cls in predictor_classes if cls.name in dfs]
    for df in ordered:
        df["seq_name"] = df["seq_name"].astype(str)
    merged = ordered[0]
    for df in ordered[1:]:
        merged = pd.merge(merged, df, on="seq_name", how="outer")

    if errors:
        print(f"\n  Failed: {', '.join(errors)}")

    # ── Restore original headers (normal mode only) ────────────────────
    if not _skip_prepare and _id_map:
        merged.insert(
            1, "original_header",
            merged["seq_name"].map(
                {k: v["original_header"] for k, v in _id_map.items()}
            )
        )

    # ── Ensemble scores ────────────────────────────────────────────────
    from pega.ensemble import compute_ensembles, ENSEMBLES
    if ensemble_names is None:
        merged = compute_ensembles(merged)
    elif ensemble_names:
        unknown = [n for n in ensemble_names if n not in ENSEMBLES]
        if unknown:
            print(f"  [warning] Unknown ensembles: {unknown}", file=sys.stderr)
        valid_ens = [n for n in ensemble_names if n in ENSEMBLES]
        if valid_ens:
            merged = compute_ensembles(merged, ensemble_names=valid_ens)

    # ── Liability columns last ────────────────────────────────────────
    liability_cols = [c for c in _LIABILITY_COLUMNS if c in merged.columns]
    if liability_cols:
        merged = merged[
            [c for c in merged.columns if c not in liability_cols]
            + liability_cols
        ]

    # ── Drop individual scores if requested ───────────────────────────
    if no_individual_scores:
        ens_cols = [c for c in merged.columns if c.startswith("ensemble_")]
        id_cols  = [c for c in ("seq_name", "original_header")
                    if c in merged.columns]
        merged   = merged[id_cols + ens_cols]

    # ── Export ────────────────────────────────────────────────────────
    if not _no_export:
        if export_tsv is None:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_tsv = f"PEGA_results_{timestamp}.tsv"
        out_path = Path(export_tsv)
        merged.to_csv(out_path, sep="\t", index=False)
        # Break the column count down by kind — a bare total hides that
        # some of these columns score the opposite of what the rest do.
        n_ens  = sum(1 for c in merged.columns if c.startswith("ensemble_"))
        n_liab = sum(1 for c in merged.columns if c in _LIABILITY_COLUMNS)
        n_id   = sum(1 for c in ("seq_name", "original_header")
                     if c in merged.columns)
        n_act  = len(merged.columns) - n_ens - n_liab - n_id
        parts  = [f"{n_act} activity", f"{n_ens} ensemble"]
        if n_liab:
            parts.append(f"{n_liab} liability")
        _console.print(
            f"\n  [bold green]Saved[/bold green] → [bold]{out_path}[/bold]"
        )
        _console.print(
            f"  [dim]{len(merged)} sequences · "
            f"{' · '.join(parts)} columns[/dim]"
        )

    # ── Cleanup ───────────────────────────────────────────────────────
    if _tmp_fasta is not None and _tmp_fasta.exists():
        _tmp_fasta.unlink()

    return merged
