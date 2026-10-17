"""
pega.operators.selection
========================
Genetic algorithm selection operators.

All functions receive a population DataFrame with a ``mean_geometric`` column
produced by :func:`pega.utils.calculate_scores` and return a DataFrame of
the selected individuals.

Available methods
-----------------
- ``proportional`` — fitness-proportionate (roulette wheel) selection.
- ``tournament``   — tournament selection.
- ``rank``         — rank-based selection.
- ``elitism``      — deterministic top-N selection.
- ``sus``          — stochastic universal sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FITNESS_COL = "mean_geometric"


def proportional(population: pd.DataFrame, n: int) -> pd.DataFrame:
    """Fitness-proportionate (roulette wheel) selection."""
    n = min(n, len(population))
    fitness = population[_FITNESS_COL].clip(lower=0)
    total = fitness.sum()
    # Drawing without replacement needs at least n entries with non-zero
    # probability, so a population where too few individuals carry any
    # fitness falls back to uniform instead of raising "Fewer non-zero
    # entries in p than size".
    if total > 0 and (fitness > 0).sum() >= n:
        probs = fitness / total
    else:
        probs = np.ones(len(population)) / len(population)
    idx = np.random.choice(population.index, size=n, replace=False, p=probs)
    return population.loc[idx]


def tournament(population: pd.DataFrame, n: int, k: int = 3) -> pd.DataFrame:
    """Tournament selection."""
    selected = []
    remaining = population.copy()
    for _ in range(min(n, len(remaining))):
        k_actual = min(k, len(remaining))
        contestants = remaining.sample(n=k_actual)
        winner = contestants.loc[contestants[_FITNESS_COL].idxmax()]
        selected.append(winner)
        remaining = remaining.drop(winner.name)
    return pd.DataFrame(selected)


def rank_based(population: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rank-based selection."""
    sorted_pop = population.sort_values(_FITNESS_COL, ascending=False).reset_index(drop=True)
    ranks = np.arange(len(sorted_pop), 0, -1, dtype=float)
    probs = ranks / ranks.sum()
    n = min(n, len(sorted_pop))
    idx = np.random.choice(sorted_pop.index, size=n, replace=False, p=probs)
    return sorted_pop.loc[idx]


def elitism(population: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministic top-N selection."""
    return population.nlargest(min(n, len(population)), _FITNESS_COL)


def stochastic_universal_sampling(population: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stochastic universal sampling (SUS)."""
    sorted_pop = population.sort_values(_FITNESS_COL, ascending=False).reset_index(drop=True)
    total = sorted_pop[_FITNESS_COL].sum()
    if total == 0:
        return sorted_pop.head(n)
    step = total / n
    start = np.random.uniform(0, step)
    pointers = [start + i * step for i in range(n)]
    selected_idx, cumsum, j = [], 0.0, 0
    for pointer in pointers:
        while cumsum < pointer and j < len(sorted_pop):
            cumsum += sorted_pop.iloc[j][_FITNESS_COL]
            j += 1
        if j > 0:
            selected_idx.append(j - 1)
    return sorted_pop.iloc[selected_idx].reset_index(drop=True)


def selection(
    population: pd.DataFrame,
    n: int,
    method: str = "proportional",
    k: int = 3,
) -> pd.DataFrame:
    """Unified selection interface.

    Parameters
    ----------
    population:
        Scored population DataFrame with a ``mean_geometric`` column.
    n:
        Number of individuals to select.
    method:
        One of ``"proportional"``, ``"tournament"``, ``"rank"``,
        ``"elitism"``, ``"sus"``.
    k:
        Tournament size (only used when ``method="tournament"``).
    """
    if len(population) < n:
        n = len(population)

    methods = {
        "proportional": lambda: proportional(population, n),
        "tournament": lambda: tournament(population, n, k=k),
        "rank": lambda: rank_based(population, n),
        "elitism": lambda: elitism(population, n),
        "sus": lambda: stochastic_universal_sampling(population, n),
    }

    if method not in methods:
        raise ValueError(
            f"Unknown selection method '{method}'. "
            f"Valid options: {', '.join(sorted(methods))}."
        )
    return methods[method]().reset_index(drop=True)
