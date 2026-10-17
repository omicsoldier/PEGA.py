"""
pega.operators.crossover
========================
Crossover operators for the PEGA genetic algorithm.

All operators take two parent sequences (strings of amino acids) and return
two offspring strings.

Available methods
-----------------
single_point    — Holland (1975). One random cut point; equal-length offspring.
two_point       — De Jong (1975). Two random cut points; equal-length offspring.
uniform         — Syswerda (1989). Per-position coin flip; equal-length offspring.
cut_and_splice  — Koza (1992), adapted. Independent cut points; offspring may
                  differ in length from parents. Marked *experimental* —
                  originally designed for genetic programming, not validated
                  specifically for peptide GA. Included because variable-length
                  exploration is biologically relevant for AMP design.

References
----------
Holland, J.H. (1975). Adaptation in Natural and Artificial Systems. MIT Press.
De Jong, K.A. (1975). Analysis of the Behavior of a Class of Genetic Adaptive
  Systems. PhD Thesis, University of Michigan.
Syswerda, G. (1989). Uniform crossover in genetic algorithms. ICGA, 2–9.
Koza, J.R. (1992). Genetic Programming. MIT Press.
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def single_point_crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """One-point crossover (Holland, 1975).

    Selects one random cut point and swaps the tails of the two parents.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of equal length (≥ 2 residues).

    Returns
    -------
    tuple[str, str]
        Two offspring of the same length as the parents.
    """
    _validate_equal(parent1, parent2)
    cut = random.randint(1, len(parent1) - 1)
    return (parent1[:cut] + parent2[cut:],
            parent2[:cut] + parent1[cut:])


def two_point_crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """Two-point crossover (De Jong, 1975).

    Selects two random cut points and swaps the middle segment between them.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of equal length (≥ 3 residues).

    Returns
    -------
    tuple[str, str]
        Two offspring of the same length as the parents.
    """
    _validate_equal(parent1, parent2, min_len=3)
    p1 = random.randint(1, len(parent1) - 2)
    p2 = random.randint(p1 + 1, len(parent1) - 1)
    return (parent1[:p1] + parent2[p1:p2] + parent1[p2:],
            parent2[:p1] + parent1[p1:p2] + parent2[p2:])


def uniform_crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """Uniform crossover (Syswerda, 1989).

    At each position, a coin flip (p = 0.5) decides which parent contributes
    the residue.  Produces maximum recombination between parents.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences of equal length (≥ 2 residues).

    Returns
    -------
    tuple[str, str]
        Two offspring of the same length as the parents.
    """
    _validate_equal(parent1, parent2)
    c1, c2 = [], []
    for a, b in zip(parent1, parent2):
        if random.random() < 0.5:
            c1.append(a)
            c2.append(b)
        else:
            c1.append(b)
            c2.append(a)
    return "".join(c1), "".join(c2)


def cut_and_splice(parent1: str, parent2: str) -> tuple[str, str]:
    """Cut-and-splice crossover — EXPERIMENTAL (adapted from Koza, 1992).

    Each parent is cut at an independently chosen random position.
    Child 1 = head of parent1 + tail of parent2.
    Child 2 = head of parent2 + tail of parent1.

    Offspring lengths may differ from the parents, enabling length-space
    exploration.  Originally designed for genetic programming; not validated
    specifically for peptide GA.  Include with caution and set length
    constraints in the population module.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences (lengths may differ; each must be ≥ 2 residues).

    Returns
    -------
    tuple[str, str]
        Two offspring (potentially different lengths from each other and
        from the parents).
    """
    _validate_single(parent1)
    _validate_single(parent2)
    cut1 = random.randint(1, len(parent1) - 1)
    cut2 = random.randint(1, len(parent2) - 1)
    return (parent1[:cut1] + parent2[cut2:],
            parent2[:cut2] + parent1[cut1:])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METHODS: dict[str, callable] = {
    "single_point":   single_point_crossover,
    "two_point":      two_point_crossover,
    "uniform":        uniform_crossover,
    "cut_and_splice": cut_and_splice,
}

_EXPERIMENTAL = {"cut_and_splice"}


def crossover(
    parent1: str,
    parent2: str,
    method: str = "single_point",
    **kwargs,
) -> tuple[str, str]:
    """Apply a crossover operator to two parent sequences.

    Parameters
    ----------
    parent1, parent2 : str
        Amino acid sequences.
    method : str
        One of ``"single_point"``, ``"two_point"``, ``"uniform"``,
        ``"cut_and_splice"``.  Default: ``"single_point"``.
    **kwargs
        Extra keyword arguments forwarded to the operator.

    Returns
    -------
    tuple[str, str]
        Two offspring sequences.

    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown crossover method '{method}'. "
            f"Available: {list(_METHODS)}"
        )
    return _METHODS[method](parent1, parent2, **kwargs)


def available_methods() -> list[str]:
    """Return all registered crossover method names."""
    return list(_METHODS)


def experimental_methods() -> list[str]:
    """Return methods marked as experimental (not validated for peptide GA)."""
    return list(_EXPERIMENTAL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_equal(p1: str, p2: str, min_len: int = 2) -> None:
    if len(p1) != len(p2):
        raise ValueError(
            f"Parents must have equal length (got {len(p1)} and {len(p2)})."
        )
    if len(p1) < min_len:
        raise ValueError(
            f"Parents must be at least {min_len} residues long."
        )


def _validate_single(p: str, min_len: int = 2) -> None:
    if len(p) < min_len:
        raise ValueError(
            f"Sequence must be at least {min_len} residues long (got {len(p)})."
        )
