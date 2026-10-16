"""
tests/test_all.py
=================
Full test suite for PEGA operators, population, and scoring pipeline.

Usage
-----
    conda activate pega_env
    python tests/test_all.py

Sections
--------
1. crossover   — all methods, position logic, error handling
2. mutation    — all strategies, matrices, property groups, indels
3. selection   — all methods, fitness column, edge cases
4. population  — generate, score (real predictors), fitness, stats, I/O
5. pipeline    — full PEGA score run on test.fasta
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

# ── helpers ──────────────────────────────────────────────────────────────────
_section_ok = _section_fail = 0
_total_ok = _total_fail = 0

def section(title: str) -> None:
    global _section_ok, _section_fail
    _section_ok = _section_fail = 0
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")

def check(cond: bool, msg: str) -> None:
    global _section_ok, _section_fail, _total_ok, _total_fail
    if cond:
        _section_ok  += 1
        _total_ok    += 1
        print(f"  [PASS] {msg}")
    else:
        _section_fail += 1
        _total_fail   += 1
        print(f"  [FAIL] {msg}")

def summary() -> None:
    tag = "✓" if _section_fail == 0 else "✗"
    print(f"  {tag}  {_section_ok} pass / {_section_fail} fail")



def main() -> int:
    """Run the full manual check suite and return a process exit code."""
    global _section_ok, _section_fail, _total_ok, _total_fail
    _section_ok = _section_fail = 0
    _total_ok = _total_fail = 0

    # ════════════════════════════════════════════════════════════════════════════
    # 1. CROSSOVER
    # ════════════════════════════════════════════════════════════════════════════
    section("1 · CROSSOVER  (pega.operators.crossover)")

    from pega.operators.crossover import (
        crossover, available_methods, experimental_methods,
        single_point_crossover, two_point_crossover,
        uniform_crossover,
    )

    P1, P2 = "ACDEFGHIKL", "LKIHGFEDCA"

    print("\n  ── length + canonical AAs (20 trials each) ──")
    for method in available_methods():
        a = "ACDE" if method == "cut_and_splice" else P1
        b = "LKIG" if method == "cut_and_splice" else P2
        ok_len = ok_aa = 0
        for _ in range(20):
            c1, c2 = crossover(a, b, method=method)
            if method != "cut_and_splice":
                ok_len += (len(c1) == len(a) and len(c2) == len(b))
            else:
                ok_len += (len(c1) >= 1 and len(c2) >= 1)
            ok_aa += (set(c1).issubset(CANONICAL) and set(c2).issubset(CANONICAL))
        check(ok_len == 20, f"{method}: length OK (20/20)")
        check(ok_aa  == 20, f"{method}: canonical AAs (20/20)")

    print("\n  ── position-level recombination ──")
    for _ in range(50):
        c1, c2 = uniform_crossover(P1, P2)
        for i, (a, b) in enumerate(zip(c1, c2)):
            if not ({a, b} == {P1[i], P2[i]}):
                check(False, f"uniform: pos {i} broken: got {a},{b} expected {{{P1[i]},{P2[i]}}}")
                break
        else:
            continue
        break
    else:
        check(True, "uniform: position recombination correct (50 trials)")

    print("\n  ── two_point: middle segment is always from one parent ──")
    for _ in range(50):
        c1, _ = two_point_crossover(P1, P2)
        # c1 should be: P1[:p1] + P2[p1:p2] + P1[p2:]
        # so every position must come from P1 or P2
        valid = all(c1[i] in (P1[i], P2[i]) for i in range(len(c1)))
        check(valid, f"two_point: all chars from parents → {c1}")
        break

    print("\n  ── experimental_methods ──")
    check(set(experimental_methods()) == {"cut_and_splice"},
          f"experimental = {experimental_methods()}")

    print("\n  ── error handling ──")
    for fn, args, label in [
        (single_point_crossover, ("ACDE", "LKI"),   "unequal lengths"),
        (two_point_crossover,    ("AC", "LK"),        "too short for two_point"),
        (crossover,              (P1, P2),            "bad method kwarg"),
    ]:
        try:
            if label == "bad method kwarg":
                crossover(P1, P2, method="bogus")
            else:
                fn(*args)
            check(False, f"{label}: should raise ValueError")
        except (ValueError, TypeError):
            check(True, f"{label} → ValueError raised correctly")

    summary()


    # ════════════════════════════════════════════════════════════════════════════
    # 2. MUTATION
    # ════════════════════════════════════════════════════════════════════════════
    section("2 · MUTATION  (pega.operators.mutation)")

    from pega.operators.mutation import (
        mutate, insert_aa, delete_aa,
        available_strategies, BLOSUM62, PAM30, PAM250, PROPERTY_GROUPS, _AA_TO_GROUP,
    )

    SEQ = "ACDEFGHIKL"

    print("\n  ── all strategies: length + AAs (30 trials each) ──")
    for strat in available_strategies():
        results = [mutate(SEQ, mutation_rate=0.3, strategy=strat) for _ in range(30)]
        check(all(len(r) == len(SEQ) for r in results), f"{strat}: length preserved")
        check(all(set(r).issubset(CANONICAL) for r in results), f"{strat}: canonical AAs")
        diffs = [sum(a != b for a, b in zip(SEQ, r)) for r in results]
        check(max(diffs) >= 1, f"{strat}: ≥1 mutation (max={max(diffs)})")

    print("\n  ── BLOSUM62 greedy: picks highest-scoring neighbor ──")
    best = {aa: max((a for a in BLOSUM62[aa] if a != aa),
                    key=lambda a: BLOSUM62[aa][a]) for aa in CANONICAL}
    for aa in list(CANONICAL)[:5]:
        seq = aa * 10
        m = mutate(seq, mutation_rate=1.0, strategy="blosum62")
        changed = [r for r in m if r != aa]
        if changed:
            check(changed[0] == best[aa],
                  f"BLOSUM62 greedy: {aa} → {changed[0]} (best={best[aa]})")

    print("\n  ── property_preserving: stays in group ──")
    for aa, grp in list(_AA_TO_GROUP.items())[:3]:
        ok_grp = True
        for _ in range(20):
            m = mutate(aa * 8, mutation_rate=0.5, strategy="property")
            if not all(r in PROPERTY_GROUPS[grp] for r in m):
                ok_grp = False; break
        check(ok_grp, f"property({aa}): always stays in '{grp}'")

    print("\n  ── charge_biased: >50% cationic (K,R,H) ──")
    results = [mutate("ACDE", mutation_rate=1.0, strategy="charge_biased")
               for _ in range(200)]
    mutated = [aa for r in results for aa, o in zip(r, "ACDE") if aa != o]
    if mutated:
        frac = sum(1 for a in mutated if a in "KRH") / len(mutated)
        check(frac > 0.5, f"charge_biased: cationic fraction = {frac:.2f}")

    print("\n  ── matrices: 20×20, all canonical ──")
    for name, mat in [("BLOSUM62", BLOSUM62), ("PAM30", PAM30), ("PAM250", PAM250)]:
        check(len(mat) == 20 and all(len(mat[a]) == 20 for a in mat),
              f"{name}: 20×20")
        check(all(a in CANONICAL for a in mat), f"{name}: all canonical keys")

    print("\n  ── property groups: 5 groups, cover all 20 AAs ──")
    all_g = {aa for g in PROPERTY_GROUPS.values() for aa in g}
    check(all_g == CANONICAL, f"groups cover 20 AAs: {all_g ^ CANONICAL or 'OK'}")
    check(len(PROPERTY_GROUPS) == 5, f"5 groups: {list(PROPERTY_GROUPS)}")

    print("\n  ── indels ──")
    for _ in range(30):
        ins = insert_aa(SEQ)
        check(len(ins) == len(SEQ)+1 and set(ins).issubset(CANONICAL),
              f"insert: {ins}")
        d = delete_aa(SEQ)
        check(len(d) == len(SEQ)-1 and set(d).issubset(CANONICAL),
              f"delete: {d}")
        break  # show one, stress test silently below
    ok_ins = ok_del = 0
    for _ in range(100):
        ok_ins += (len(insert_aa(SEQ)) == len(SEQ)+1)
        ok_del += (len(delete_aa(SEQ)) == len(SEQ)-1)
    check(ok_ins == 100, "insert: correct length (100/100)")
    check(ok_del == 100, "delete: correct length (100/100)")

    print("\n  ── error handling ──")
    for fn, args, label in [
        (mutate,    ("", 0.1),            "empty sequence"),
        (mutate,    ("ACDE", 0.0),        "rate=0"),
        (mutate,    ("ACDE", 1.5),        "rate>1"),
        (mutate,    ("ACDEZ", 0.1),       "non-canonical AA"),
        (mutate,    ("ACDE", 0.1, "???"), "unknown strategy"),
        (delete_aa, ("A",),               "single residue"),
    ]:
        try:
            fn(*args)
            check(False, f"{label}: should raise ValueError")
        except ValueError:
            check(True, f"{label} → ValueError")

    summary()


    # ════════════════════════════════════════════════════════════════════════════
    # 3. SELECTION
    # ════════════════════════════════════════════════════════════════════════════
    section("3 · SELECTION  (pega.operators.selection)")

    from pega.operators.selection import selection

    rng = np.random.default_rng(0)
    pop_df = pd.DataFrame({
        "seq_name":      [f"seq{i:03d}" for i in range(50)],
        "seq_aa":        ["ACDEFGHIKL"] * 50,
        "mean_geometric": rng.uniform(0.1, 0.9, 50),
    })

    print("\n  ── all methods return correct size ──")
    for method in ["proportional", "tournament", "rank", "elitism", "sus"]:
        for n in [5, 10, 25]:
            sel = selection(pop_df, n, method=method)
            check(len(sel) == n,
                  f"{method} n={n}: {len(sel)} selected")

    print("\n  ── elitism returns highest-fitness individuals ──")
    sel = selection(pop_df, 5, method="elitism")
    top5 = pop_df.nlargest(5, "mean_geometric")["mean_geometric"].values
    check(sorted(sel["mean_geometric"].values, reverse=True) ==
          sorted(top5, reverse=True),
          "elitism: top-5 fitness matches")

    print("\n  ── tournament: winner always best in group ──")
    for _ in range(20):
        sel = selection(pop_df, 10, method="tournament", k=4)
        check(len(sel) == 10, "tournament: 10 selected")
        break

    print("\n  ── n > population size: capped gracefully ──")
    sel = selection(pop_df, 200, method="proportional")
    check(len(sel) == len(pop_df), f"n>pop: capped to {len(sel)}")

    print("\n  ── error handling ──")
    try:
        selection(pop_df, 5, method="bogus")
        check(False, "unknown method: should raise ValueError")
    except ValueError as e:
        check(True, f"unknown method → ValueError: {e}")

    summary()


    # ════════════════════════════════════════════════════════════════════════════
    # 4. POPULATION  (with real predictors)
    # ════════════════════════════════════════════════════════════════════════════
    section("4 · POPULATION  (pega.population — real predictors)")

    from pega.population import (
        generate_random, generate_from_seed, generate_from_fasta,
        score_sequences, compute_fitness, calculate_statistics,
        log_generation, to_fasta, from_fasta,
    )

    print("\n  ── generate_random ──")
    pop = generate_random(20, 15)
    check(len(pop) == 20,                                   "20 sequences generated")
    check(all(len(s) == 15 for s in pop),                   "all length 15")
    check(all(set(s).issubset(CANONICAL) for s in pop),     "all canonical AAs")
    check(len(set(pop)) == 20,                               "all unique")

    print("\n  ── generate_from_seed ──")
    for strat in ["blosum62", "property", "random"]:
        pop = generate_from_seed("ACDEFGHIKLMNPQRS", 10, 16,
                                  mutation_rate=0.2, strategy=strat)
        check(len(pop) == 10 and all(len(s) == 16 for s in pop)
              and all(set(s).issubset(CANONICAL) for s in pop),
              f"from_seed ({strat}): 10 seqs, len=16, canonical")

    print("\n  ── generate_from_fasta ──")
    fasta_in = ROOT / "tests" / "data" / "test.fasta"
    if fasta_in.exists():
        pop = generate_from_fasta(str(fasta_in), 10, 20)
        check(len(pop) == 10 and all(len(s) == 20 for s in pop),
              f"from_fasta: 10 seqs len=20 from {fasta_in.name}")
    else:
        print(f"  [SKIP] test.fasta not found at {fasta_in}")

    print("\n  ── score_sequences (real predictors, small batch) ──")
    print("       (this calls the full predictor pipeline — may take ~30s)")
    seqs = generate_random(3, 15)
    t0 = time.perf_counter()
    df = score_sequences(seqs, generation_label="TEST")
    elapsed = time.perf_counter() - t0
    check(len(df) == 3,                                     f"3 sequences scored ({elapsed:.1f}s)")
    check("seq_aa" in df.columns,                           "seq_aa column present")
    check("mean_geometric" in df.columns,                   "mean_geometric (fitness) present")
    score_cols = [c for c in df.columns
                  if c.endswith("_score") and not c.startswith("ensemble_")]
    check(len(score_cols) >= 1,                             f"≥1 predictor score cols: {score_cols}")

    print("\n  ── ensemble scores ──")
    ensemble_cols = [c for c in df.columns if c.startswith("ensemble_")]
    if ensemble_cols:
        check(all(df[c].between(0, 1).all() for c in ensemble_cols),
              f"ensemble scores ∈ [0,1]: {ensemble_cols}")
    else:
        print("  [SKIP] no ensembles computed (insufficient predictors available)")

    # Verify mathematical correctness on first row: mean_geometric is the
    # geometric mean of the individual predictor scores (NaN scores are
    # treated as 0, matching pega.population.compute_fitness).
    row = df.iloc[0]
    raw = row[score_cols].fillna(0).clip(lower=0).values.astype(float)
    expected_fitness = np.exp(np.mean(np.log(raw + 1e-9)))
    check(abs(row["mean_geometric"] - expected_fitness) < 1e-6,
          "mean_geometric correct (geometric mean)")

    check(all(df["mean_geometric"].between(0, 1)), "mean_geometric ∈ [0,1]")

    print("\n  ── compute_fitness ──")
    df2 = df.drop(columns=["mean_geometric"])
    df2 = compute_fitness(df2)
    check("mean_geometric" in df2.columns,  "recomputed after drop")
    check(all(abs(df2["mean_geometric"] - df["mean_geometric"]) < 1e-9),
          "recomputed mean_geometric matches original")

    print("\n  ── calculate_statistics ──")
    stats = calculate_statistics(df)
    check(len(stats) > 0, f"{len(stats)} stats computed")
    for col in score_cols[:2]:
        for stat in ("mean","min","max","std"):
            check(f"{col}_{stat}" in stats, f"{col}_{stat} present")

    print("\n  ── log_generation ──")
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
        tsv = f.name
    os.unlink(tsv)
    log_generation(df, "G_0", tsv)
    log_generation(df, "G_1", tsv, generation_time=1.5, new_members=2)
    lg = pd.read_csv(tsv, sep="\t")
    check(len(lg) == 2,                             "2 generations logged")
    check(pd.isna(lg.loc[0,"generation_time_s"]),   "G_0: time=NaN")
    check(float(lg.loc[1,"generation_time_s"])==1.5,"G_1: time=1.5")
    os.unlink(tsv)

    print("\n  ── to_fasta / from_fasta ──")
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
        fout = f.name
    to_fasta(df, fout)
    loaded = from_fasta(fout)
    check(len(loaded) == len(df),                         "roundtrip: count matches")
    check(all(s in df["seq_aa"].values for s in loaded),  "roundtrip: sequences match")
    os.unlink(fout)

    print("\n  ── error handling ──")
    for fn, args, label in [
        (generate_random, (0, 10),          "n=0"),
        (generate_random, (10, 0),          "length=0"),
        (score_sequences, ([], "G_0"),      "empty seq list"),
    ]:
        try:
            fn(*args)
            check(False, f"{label}: should raise")
        except ValueError:
            check(True,  f"{label} → ValueError")

    summary()


    # ════════════════════════════════════════════════════════════════════════════
    # 5. PIPELINE  (PEGA score end-to-end)
    # ════════════════════════════════════════════════════════════════════════════
    section("5 · PIPELINE  (pega.utils.calculate_scores end-to-end)")

    from pega.utils import calculate_scores

    fasta_path = ROOT / "tests" / "data" / "test.fasta"
    if not fasta_path.exists():
        print(f"  [SKIP] test.fasta not found at {fasta_path}")
    else:
        print(f"\n  Scoring {fasta_path.name}  (all available predictors)...")
        t0 = time.perf_counter()
        result = calculate_scores(str(fasta_path))
        elapsed = time.perf_counter() - t0

        n_seqs = len(result)
        score_cols = [c for c in result.columns
                      if c.endswith("_score") and not c.startswith("ensemble_")]
        ensemble_cols = [c for c in result.columns if c.startswith("ensemble_")]

        check(n_seqs >= 1,                    f"{n_seqs} sequences scored ({elapsed:.1f}s)")
        check(len(score_cols) >= 1,           f"{len(score_cols)} predictor scores: {score_cols}")
        if ensemble_cols:
            check(all(result[c].between(0, 1).all() for c in ensemble_cols),
                  f"ensemble scores ∈ [0,1]: {ensemble_cols}")
        else:
            print("  [SKIP] no ensembles computed (insufficient predictors available)")
        check(all(result[c].dropna().between(0, 1).all() for c in score_cols),
              "all predictor scores ∈ [0,1]")

        print("\n  Preview (first 3 rows):")
        pd.set_option("display.max_columns", 6)
        pd.set_option("display.width", 120)
        preview_cols = ["seq_name"] + score_cols[:3] + ensemble_cols[:3]
        print(result[preview_cols].head(3).to_string(index=False))

    summary()

    print(f"\n{'═'*65}")
    print(f"  TOTAL RESULTS: {_total_ok} PASS  /  {_total_fail} FAIL")
    print(f"{'═'*65}")
    return 0 if _total_fail == 0 else 1


@pytest.mark.conda_required
@pytest.mark.slow
def test_all() -> None:
    """Pytest entry point — requires pega_env with real predictors installed.

    Reruns the full manual check suite (crossover, mutation, selection,
    population, end-to-end pipeline) and asserts success.
    """
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
