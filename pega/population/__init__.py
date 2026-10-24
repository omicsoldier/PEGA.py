"""
pega.population
===============
Population management for the PEGA genetic algorithm.

This module handles:

- **Generation** — create initial populations from scratch, from a seed
  peptide, or from an existing FASTA file.
- **Scoring** — score a list of sequences in a single batch call to the
  predictor pipeline (one temp FASTA, one ``calculate_scores`` call).
  This is significantly more efficient than scoring each sequence
  individually as in classic GA implementations.
- **Fitness** — compute the geometric mean across all predictor scores as
  the fitness value used by selection operators.
- **Statistics** — per-generation descriptive statistics logged to TSV.
- **I/O** — save/load populations as FASTA files.

Population representation
--------------------------
A population is a ``pandas.DataFrame`` with the following columns:

- ``seq_name`` — unique identifier (e.g. ``G_0_00001``).
- ``seq_aa``   — amino acid sequence string.
- One ``*_score`` column per predictor that ran successfully.
- ``mean_geometric`` — geometric mean across all predictor scores; used as
  the fitness column by ``pega.operators.selection``.

References
----------
Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization and
  Machine Learning. Addison-Wesley.
Porto, W.F. et al. (2012). Computational design of antimicrobial peptides.
  BMC Bioinformatics, 13, S6.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pega.operators.mutation import mutate, insert_aa, delete_aa, _CANONICAL

_CANONICAL_LIST: list[str] = list(_CANONICAL)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_random(
    n: int,
    length: int,
    allow_duplicates: bool = False,
) -> list[str]:
    """Generate ``n`` random peptides of fixed length.

    Parameters
    ----------
    n : int
        Number of peptides to generate.
    length : int
        Length of each peptide (number of residues).
    allow_duplicates : bool
        If ``False`` (default), all returned sequences are unique.

    Returns
    -------
    list[str]
        List of amino acid sequences.

    Raises
    ------
    ValueError
        If ``n`` or ``length`` ≤ 0.
    """
    if n <= 0 or length <= 0:
        raise ValueError(f"n and length must be > 0 (got n={n}, length={length}).")

    def _rand() -> str:
        return "".join(random.choices(_CANONICAL_LIST, k=length))

    if allow_duplicates:
        return [_rand() for _ in range(n)]

    seen: set[str] = set()
    while len(seen) < n:
        seen.add(_rand())
    return list(seen)


def generate_from_seed(
    seed: str | list[str],
    n: int,
    length: int,
    mutation_rate: float = 0.2,
    strategy: str = "blosum62",
    allow_duplicates: bool = False,
) -> list[str]:
    """Generate a population by mutating one or more seed peptides.

    Seeds are first adjusted to ``length`` via insertions/deletions, then
    mutated repeatedly until ``n`` unique sequences are produced.

    Parameters
    ----------
    seed : str or list[str]
        One or more starting peptide sequences.
    n : int
        Target population size.
    length : int
        Target sequence length.
    mutation_rate : float
        Fraction of positions mutated per mutation step. Default: 0.2.
    strategy : str
        Mutation strategy passed to :func:`pega.operators.mutation.mutate`.
        Default: ``"blosum62"``.
    allow_duplicates : bool
        If ``False`` (default), all returned sequences are unique.

    Returns
    -------
    list[str]
        List of ``n`` amino acid sequences of length ``length``.

    Raises
    ------
    ValueError
        If any seed contains non-canonical amino acids.
    """
    seeds = [seed] if isinstance(seed, str) else list(seed)
    if len(seeds) > n:
        raise ValueError(
            f"Number of seeds ({len(seeds)}) cannot exceed population size n ({n})."
        )

    population: list[str] = []
    seen: set[str] = set()

    def _adjust(seq: str) -> str:
        """Bring seq to exactly ``length`` via indels."""
        attempts = 0
        while len(seq) != length and attempts < 500:
            if len(seq) < length:
                seq = insert_aa(seq)
            else:
                seq = delete_aa(seq) if len(seq) > 1 else seq
            attempts += 1
        return seq

    def _add(seq: str) -> bool:
        if allow_duplicates or seq not in seen:
            population.append(seq)
            seen.add(seq)
            return True
        return False

    # Seed members first
    for s in seeds:
        adj = _adjust(s)
        _add(adj)

    # Fill remaining via mutation
    max_iter = n * 200
    iterations = 0
    while len(population) < n and iterations < max_iter:
        template = random.choice(seeds)
        candidate = _adjust(mutate(template, mutation_rate, strategy))
        _add(candidate)
        iterations += 1

    if len(population) < n:
        # Fallback: pad with random sequences
        for seq in generate_random(n - len(population), length, allow_duplicates=True):
            _add(seq)

    return population[:n]


def generate_from_fasta(
    fasta_path: str | Path,
    n: int,
    length: int,
    mutation_rate: float = 0.2,
    strategy: str = "blosum62",
) -> list[str]:
    """Generate a population using sequences from a FASTA file as seeds.

    Each FASTA sequence is adjusted to ``length`` and used as a seed.
    If there are fewer sequences than ``n``, additional members are generated
    by mutation; if there are more, ``n`` are sampled at random.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the input FASTA file.
    n : int
        Target population size.
    length : int
        Target sequence length.
    mutation_rate : float
        Mutation rate for filling remaining slots. Default: 0.2.
    strategy : str
        Mutation strategy. Default: ``"blosum62"``.

    Returns
    -------
    list[str]
        List of ``n`` amino acid sequences.
    """
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}.")

    seeds = [str(r.seq).upper() for r in records]
    if len(seeds) > n:
        seeds = random.sample(seeds, n)

    return generate_from_seed(seeds, n, length, mutation_rate, strategy)


# ---------------------------------------------------------------------------
# Scoring and fitness
# ---------------------------------------------------------------------------


def score_sequences(
    sequences: list[str],
    generation_label: str = "G_0",
    predictor_names: list[str] | None = None,
    jobs: int = 1,
) -> pd.DataFrame:
    """Score a list of sequences using the PEGA predictor pipeline.

    All sequences are written to a single temporary FASTA file and scored
    in one batch call to :func:`pega.utils.calculate_scores`.  This is
    significantly more efficient than scoring each sequence individually.

    Parameters
    ----------
    sequences : list[str]
        List of amino acid sequences to score.
    generation_label : str
        Prefix for sequence identifiers (e.g. ``"G_1"``). Default: ``"G_0"``.
    predictor_names : list[str] or None
        Predictors to use. ``None`` uses all available.
    jobs : int
        Parallel workers for scoring. Default: 1.

    Returns
    -------
    pd.DataFrame
        Population DataFrame with columns ``seq_name``, ``seq_aa``, one
        ``*_score`` column per predictor, and ``mean_geometric`` fitness.
    """
    from pega.utils import calculate_scores

    if not sequences:
        raise ValueError("sequences list is empty.")

    # Map seq_name → seq_aa for later join
    seq_map: dict[str, str] = {}
    records = []
    for i, seq in enumerate(sequences):
        name = f"{generation_label}_{i:05d}"
        seq_map[name] = seq
        records.append(SeqRecord(Seq(seq), id=name, description=""))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, prefix="pega_pop_"
    ) as tmp:
        SeqIO.write(records, tmp, "fasta")
        tmp_path = tmp.name

    try:
        scored = calculate_scores(
            fasta_path=tmp_path,
            predictor_names=predictor_names,
            jobs=jobs,
            validate=False,
            _skip_prepare=True,
        )
    finally:
        os.unlink(tmp_path)

    # Add sequence column and compute fitness
    scored["seq_aa"] = scored["seq_name"].map(seq_map)
    scored = compute_fitness(scored)

    # Reorder: identifiers → fitness → verdict → predictor scores → ensembles
    from pega.cutoffs import VERDICT_COLUMN
    predictor_cols = [c for c in scored.columns
                      if c.endswith("_score") and not c.startswith("ensemble_")]
    ensemble_cols  = [c for c in scored.columns if c.startswith("ensemble_")]

    cols = (
        ["seq_name", "seq_aa", "mean_geometric", VERDICT_COLUMN]
        + predictor_cols
        + ensemble_cols
    )
    cols = [c for c in cols if c in scored.columns]
    return scored[cols].reset_index(drop=True)


def compute_fitness(
    df: pd.DataFrame,
    fitness: str | None = None,
) -> pd.DataFrame:
    """Set the fitness the selection operators maximise.

    The result always lands in ``mean_geometric``, whatever the source,
    because that is the column ``pega.operators.selection`` reads.

    Parameters
    ----------
    df : pd.DataFrame
        Population DataFrame from ``score_sequences()``.
    fitness : str or None
        Which score drives selection. A predictor name (``"ampnet"``), its
        score column (``"AMPnet_score"``) or an ensemble column
        (``"ensemble_AMP_class_score"``). ``None`` keeps the default: the
        geometric mean across every individual predictor column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``mean_geometric`` added or updated.

    Raises
    ------
    ValueError
        If ``fitness`` names no usable column, or names a liability
        column — selection maximises, so a haemolysis or toxicity score
        would evolve the population *towards* that liability.
    """
    exclude = {"seq_name", "seq_aa", "original_header", "mean_geometric"}

    if fitness is not None:
        from pega.cutoffs import LIABILITY_COLUMNS

        column = fitness
        if column not in df.columns:
            # A predictor's column is not its name plus "_score" — ampnet
            # emits AMPnet_score — so resolve it through the registry.
            from pega.registry import registry
            try:
                column = registry.get(fitness).score_column()
            except Exception:
                column = fitness
        if column not in df.columns:
            available = [c for c in df.columns
                         if c.endswith("_score") and c not in exclude]
            raise ValueError(
                f"Fitness '{fitness}' matches no column in the population. "
                f"Available: {available}"
            )
        if column in LIABILITY_COLUMNS:
            raise ValueError(
                f"'{column}' scores a liability, where higher is worse, and "
                "the selection operators maximise fitness — using it would "
                "evolve the population towards that liability. Pick an "
                "activity or ensemble column instead."
            )
        df["mean_geometric"] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        return df

    # Default: geometric mean over the individual predictor columns.
    score_cols = [
        c for c in df.columns
        if c.endswith("_score") and not c.startswith("ensemble_")
        and c not in exclude
    ]
    if not score_cols:
        raise ValueError(
            "No individual predictor '*_score' columns found. "
            "Run score_sequences() first."
        )
    log_scores = np.log(df[score_cols].fillna(0).clip(lower=0) + 1e-9)
    df["mean_geometric"] = np.exp(log_scores.mean(axis=1))
    return df


# ---------------------------------------------------------------------------
# Statistics and logging
# ---------------------------------------------------------------------------


def calculate_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Compute descriptive statistics for a scored population.

    Parameters
    ----------
    df : pd.DataFrame
        Scored population DataFrame.

    Returns
    -------
    dict[str, float]
        Dictionary with ``{column}_{stat}`` keys for mean and min of
        each numeric column (excluding ``seq_name``).
    """
    stats: dict[str, float] = {}
    numeric = df.select_dtypes(include=np.number).columns.difference(["seq_name"])
    for col in numeric:
        stats[f"{col}_mean"] = float(df[col].mean())
        stats[f"{col}_min"]  = float(df[col].min())
        stats[f"{col}_max"]  = float(df[col].max())
        stats[f"{col}_std"]  = float(df[col].std())
    return stats


def log_generation(
    df: pd.DataFrame,
    generation_label: str,
    output_tsv: str | Path,
    generation_time: float | None = None,
    new_members: int = 0,
) -> None:
    """Append a single-line statistics summary to a TSV log file.

    Creates the file with a header row on first call; subsequent calls
    append without writing the header again.

    Parameters
    ----------
    df : pd.DataFrame
        Scored population for the current generation.
    generation_label : str
        Generation identifier (e.g. ``"G_0"``, ``"G_1"``).
    output_tsv : str or Path
        Path to the output TSV file.
    generation_time : float or None
        Wall-clock time (seconds) for this generation. Default: ``None``.
    new_members : int
        Number of new sequences added this generation. Default: 0.
    """
    path = Path(output_tsv)
    numeric = df.select_dtypes(include=np.number).columns.difference(["seq_name"])

    header = (
        ["generation", "new_members"]
        + [f"{c}_{s}" for c in numeric for s in ("mean", "min", "max", "std")]
        + ["generation_time_s"]
    )

    row: list = [generation_label, new_members]
    for col in numeric:
        row += [
            round(df[col].mean(), 6),
            round(df[col].min(), 6),
            round(df[col].max(), 6),
            round(df[col].std(), 6),
        ]
    row.append(round(generation_time, 3) if generation_time is not None else "")

    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(map(str, row)) + "\n")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def to_fasta(
    population: pd.DataFrame | list[str],
    path: str | Path,
    generation_label: str = "G_0",
) -> Path:
    """Save a population to a FASTA file.

    Parameters
    ----------
    population : pd.DataFrame or list[str]
        A scored population DataFrame (must have ``seq_name`` and ``seq_aa``
        columns) or a plain list of sequences.
    path : str or Path
        Output file path.
    generation_label : str
        Prefix used for sequence IDs when ``population`` is a list.
        Default: ``"G_0"``.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(population, pd.DataFrame):
        records = [
            SeqRecord(Seq(row["seq_aa"]), id=row["seq_name"], description="")
            for _, row in population.iterrows()
        ]
    else:
        records = [
            SeqRecord(Seq(seq), id=f"{generation_label}_{i:05d}", description="")
            for i, seq in enumerate(population)
        ]

    SeqIO.write(records, str(path), "fasta")
    return path.resolve()


def from_fasta(path: str | Path) -> list[str]:
    """Load sequences from a FASTA file.

    Parameters
    ----------
    path : str or Path
        Input FASTA file.

    Returns
    -------
    list[str]
        List of amino acid sequences (uppercase).
    """
    return [str(r.seq).upper() for r in SeqIO.parse(str(path), "fasta")]
