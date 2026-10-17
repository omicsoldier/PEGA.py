"""
tests/test_operators.py
=======================
Tests for pega.operators.crossover and pega.operators.mutation.
Run with: python tests/test_operators.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pega.operators.crossover import (
    crossover, available_methods, experimental_methods,
    single_point_crossover, two_point_crossover,
    uniform_crossover, cut_and_splice,
)
from pega.operators.mutation import (
    mutate, insert_aa, delete_aa,
    available_strategies, experimental_strategies,
    BLOSUM62, PAM30, PAM250, PROPERTY_GROUPS, _AA_TO_GROUP,
)

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
P1 = "ACDEFGHIKL"
P2 = "LKIHGFEDCA"
ok = fail = 0

def check(cond: bool, msg: str) -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {msg}")
    else:
        fail += 1
        print(f"  [FAIL] {msg}")


def main() -> int:
    """Run the full manual check suite and return a process exit code."""
    global ok, fail
    ok = fail = 0

    _run_checks()

    print("\n" + "═"*60)
    print(f"  RESULTS: {ok} PASS  /  {fail} FAIL")
    print("═"*60)
    return 0 if fail == 0 else 1


def test_operators() -> None:
    """Pytest entry point — reruns the manual suite and asserts success."""
    assert main() == 0


def _run_checks() -> None:
    # ── CROSSOVER ────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  CROSSOVER")
    print("═"*60)

    print("\n── single_point ──")
    for _ in range(20):
        c1, c2 = single_point_crossover(P1, P2)
        check(len(c1) == len(P1) and len(c2) == len(P2), f"length preserved → {c1} | {c2}")
        check(set(c1).issubset(CANONICAL), "canonical AAs")

    # Verify position-level recombination
    c1, c2 = single_point_crossover(P1, P2)
    for i, (a, b) in enumerate(zip(c1, c2)):
        check({a, b} == {P1[i], P2[i]}, f"pos {i}: {a},{b} from {P1[i]},{P2[i]}")

    print("\n── two_point ──")
    for _ in range(20):
        c1, c2 = two_point_crossover(P1, P2)
        check(len(c1) == len(P1), f"length preserved → {c1}")
        check(set(c1).issubset(CANONICAL), "canonical AAs")

    print("\n── uniform ──")
    for _ in range(20):
        c1, c2 = uniform_crossover(P1, P2)
        check(len(c1) == len(P1), f"length preserved → {c1}")
        check(set(c1).issubset(CANONICAL), "canonical AAs")
        for i, (a, b) in enumerate(zip(c1, c2)):
            check(a in (P1[i], P2[i]) and b in (P1[i], P2[i]),
                  f"pos {i}: {a},{b} ∈ {{{P1[i]},{P2[i]}}}")
        break

    print("\n── cut_and_splice (experimental) ──")
    for _ in range(20):
        c1, c2 = cut_and_splice(P1, P2)
        check(len(c1) >= 1 and len(c2) >= 1, f"non-empty → {c1} | {c2}")
        check(set(c1).issubset(CANONICAL), "canonical AAs")

    print("\n── dispatcher + experimental_methods ──")
    for m in available_methods():
        a, b = crossover("ACDE" if m == "cut_and_splice" else P1,
                         "LKIHG" if m == "cut_and_splice" else P2, method=m)
        check(len(a) >= 1, f"{m}: offspring produced → {a}")
    check(set(experimental_methods()) == {"cut_and_splice"},
          f"experimental = {experimental_methods()}")

    print("\n── error handling ──")
    for fn, args, label in [
        (single_point_crossover, ("ACDE", "LKI"),   "unequal lengths"),
        (two_point_crossover,    ("AC", "LK"),        "too short for two_point"),
        (crossover,              (P1, P2, "bogus"),   "unknown method"),
    ]:
        try:
            fn(*args)
            check(False, f"{label}: should raise ValueError")
        except (ValueError, TypeError) as e:
            check(True, f"{label} → error: {e}")


    # ── MUTATION ─────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  MUTATION")
    print("═"*60)

    SEQ = "ACDEFGHIKL"

    print("\n── all strategies: length + canonical AAs ──")
    for strat in available_strategies():
        results = [mutate(SEQ, mutation_rate=0.3, strategy=strat) for _ in range(30)]
        check(all(len(r) == len(SEQ) for r in results), f"{strat}: length preserved")
        check(all(set(r).issubset(CANONICAL) for r in results), f"{strat}: canonical AAs")
        diffs = [sum(a != b for a, b in zip(SEQ, r)) for r in results]
        check(max(diffs) >= 1, f"{strat}: ≥1 mutation in 30 trials (max={max(diffs)})")

    print("\n── BLOSUM62 greedy: selects highest-scoring neighbor ──")
    best_for = {aa: max((a for a in BLOSUM62[aa] if a != aa),
                         key=lambda a: BLOSUM62[aa][a]) for aa in CANONICAL}
    for aa, best in list(best_for.items())[:5]:
        seq = aa * 10
        m = mutate(seq, mutation_rate=1.0, strategy="blosum62")
        mutated = [r for r in m if r != aa]
        if mutated:
            check(mutated[0] == best,
                  f"BLOSUM62 greedy: {aa} → {mutated[0]} (best = {best})")

    print("\n── property_preserving: stays in group ──")
    for aa, grp in list(_AA_TO_GROUP.items())[:5]:
        seq = aa * 8
        for _ in range(10):
            m = mutate(seq, mutation_rate=0.5, strategy="property")
            for r in m:
                check(r in PROPERTY_GROUPS[grp],
                      f"property({aa}) → {r} stays in group '{grp}'")
        break

    print("\n── charge_biased: favors K, R, H ──")
    cationic = set("KRH")
    results = [mutate("ACDE", mutation_rate=1.0, strategy="charge_biased")
               for _ in range(200)]
    all_mutated = [aa for r in results for aa, orig in zip(r, "ACDE") if aa != orig]
    if all_mutated:
        frac = sum(1 for a in all_mutated if a in cationic) / len(all_mutated)
        check(frac > 0.5, f"charge_biased: cationic fraction = {frac:.2f} > 0.50")

    print("\n── mutation rates ──")
    for rate in [0.1, 0.3, 0.5, 1.0]:
        diffs = [sum(a != b for a, b in zip(SEQ, mutate(SEQ, rate)))
                 for _ in range(50)]
        avg = np.mean(diffs)
        check(avg >= rate * len(SEQ) * 0.3,
              f"rate={rate}: avg mutations = {avg:.2f}")

    print("\n── insert_aa / delete_aa ──")
    for _ in range(30):
        ins = insert_aa(SEQ)
        check(len(ins) == len(SEQ)+1 and set(ins).issubset(CANONICAL),
              f"insert: {ins}")
        d = delete_aa(SEQ)
        check(len(d) == len(SEQ)-1 and set(d).issubset(CANONICAL),
              f"delete: {d}")

    print("\n── matrices: 20x20, all canonical keys ──")
    for name, mat in [("BLOSUM62", BLOSUM62), ("PAM30", PAM30), ("PAM250", PAM250)]:
        check(len(mat) == 20, f"{name}: 20 rows")
        check(all(len(mat[aa]) == 20 for aa in mat), f"{name}: 20 cols per row")
        check(all(aa in CANONICAL for aa in mat), f"{name}: keys are canonical")

    print("\n── property groups: 5 groups, cover all 20 AAs ──")
    all_g = {aa for g in PROPERTY_GROUPS.values() for aa in g}
    check(all_g == CANONICAL, f"groups cover 20 AAs: {all_g ^ CANONICAL or 'OK'}")
    check(len(PROPERTY_GROUPS) == 5, f"5 groups: {list(PROPERTY_GROUPS)}")
    check(set(experimental_strategies()) == {"charge_biased"},
          f"experimental = {experimental_strategies()}")

    print("\n── error handling ──")
    for fn, args, label in [
        (mutate,     ("", 0.1),            "empty sequence"),
        (mutate,     ("ACDE", 0.0),        "rate = 0"),
        (mutate,     ("ACDE", 1.5),        "rate > 1"),
        (mutate,     ("ACDEZ", 0.1),       "non-canonical AA"),
        (mutate,     ("ACDE", 0.1, "???"), "unknown strategy"),
        (delete_aa,  ("A",),               "single residue delete"),
    ]:
        try:
            fn(*args)
            check(False, f"{label}: should raise ValueError")
        except ValueError as e:
            check(True, f"{label} → ValueError: {e}")


if __name__ == "__main__":
    sys.exit(main())
