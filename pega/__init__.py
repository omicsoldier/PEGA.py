"""
PEGA — Peptide Evolution via Genetic Algorithm
===============================================
Multi-predictor ensemble scoring for antimicrobial peptide research.

https://github.com/omicsoldier/PEGA.py
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "omicsoldier"
__license__ = "MIT"

__all__ = [
    "score",
    "list_predictors",
    "__version__",
]


def score(
    fasta_path: str,
    predictors: list[str] | None = None,
    ensemble: bool = True,
    export_tsv: str | None = None,
):
    """Score sequences in a FASTA file using available AMP predictors.

    Parameters
    ----------
    fasta_path:
        Path to the input FASTA file.
    predictors:
        List of predictor names to use (e.g. ``["ampnet", "modlamp_rf"]``).
        If ``None``, all available predictors are used automatically.
    ensemble:
        If ``True`` (default), validated ensemble scores are appended
        to the output DataFrame. If ``False``, only individual predictor
        scores are returned.
    export_tsv:
        Optional file path. When provided, results are written as TSV.

    Returns
    -------
    pandas.DataFrame
        One row per sequence, one column per predictor score, plus
        ensemble score columns when ``ensemble=True``.
    """
    from pega.utils import calculate_scores

    return calculate_scores(
        fasta_path=fasta_path,
        predictor_names=predictors,
        ensemble_names=None if ensemble else [],
        export_tsv=export_tsv,
    )


def list_predictors(available_only: bool = False) -> list[dict]:
    """Return metadata for all registered predictors.

    Parameters
    ----------
    available_only:
        If ``True``, only predictors whose dependencies are currently
        satisfied are returned.

    Returns
    -------
    list of dict
        Each dict contains ``name``, ``predictor_id``, ``description``,
        ``category``, and ``available`` keys.
    """
    from pega.registry import registry

    predictors = (
        registry.list_available() if available_only else registry.list_all()
    )
    return [
        {
            "name": p.name,
            "predictor_id": p.predictor_id,
            "description": p.description,
            "category": p.category,
            "available": p.is_available(),
        }
        for p in predictors
    ]
