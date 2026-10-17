"""
pega.operators.mutation
=======================
Mutation operators for the PEGA genetic algorithm.

Point substitution strategies
------------------------------
blosum62            — greedy BLOSUM62-guided substitution (Henikoff & Henikoff,
                      1992). Replaces each selected residue with its highest-
                      scoring non-identical BLOSUM62 neighbor. Conservative.
blosum62_weighted   — probabilistic BLOSUM62 sampling. Samples a replacement AA
                      from a softmax over BLOSUM62 scores. Less deterministic
                      than greedy; explores more of sequence space.
pam30               — greedy PAM30-guided substitution (Dayhoff et al., 1978).
                      Very conservative — models short evolutionary distances.
pam250              — greedy PAM250-guided substitution (Dayhoff et al., 1978).
                      Permissive — models long evolutionary distances.
property_preserving — replaces within the same physicochemical group (Fjell et
                      al., 2011). Maintains charge, polarity and hydrophobicity
                      properties relevant to AMP activity.
random              — completely random canonical AA replacement. Maximum
                      exploration; standard in GA literature (Goldberg, 1989).

EXPERIMENTAL (documented but validated only in adjacent fields)
---------------------------------------------------------------
charge_biased       — biases substitutions toward cationic residues (K, R, H).
                      Motivated by the cationic nature of most natural AMPs
                      (Hancock et al., 2012) but not validated as a formal GA
                      mutation operator.

Indel operators (standard GA for sequences)
--------------------------------------------
insert_aa           — insert one random canonical AA at a random position.
delete_aa           — remove one residue at a random position (min length: 2).

References
----------
Dayhoff, M.O. et al. (1978). Atlas of Protein Sequence and Structure. NBRF.
Fjell, C.D. et al. (2011). Identification of novel antibacterial peptides by
  chemoinformatics and machine learning. J. Chem. Inf. Model., 51, 2873–2881.
Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization and Machine
  Learning. Addison-Wesley.
Hancock, R.E.W. et al. (2012). Antibiofilm activity of host defence peptides:
  complexity provides opportunities. Expert Rev. Anti Infect. Ther., 10, 1–4.
Henikoff, S. & Henikoff, J.G. (1992). Amino acid substitution matrices from
  protein blocks. PNAS, 89(22), 10915–10919.
Porto, W.F. et al. (2012). Computational design of novel cationic antimicrobial
  peptides using BLOSUM-based mutation. BMC Bioinformatics, 13, S6.
"""

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# Substitution matrices
# ---------------------------------------------------------------------------

BLOSUM62: dict[str, dict[str, int]] = {
    "A": {"A": 4,"R":-1,"N":-2,"D":-2,"C": 0,"Q":-1,"E":-1,"G": 0,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 0,"W":-3,"Y":-2,"V": 0},
    "R": {"A":-1,"R": 5,"N": 0,"D":-2,"C":-3,"Q": 1,"E": 0,"G":-2,"H": 0,"I":-3,"L":-2,"K": 2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3},
    "N": {"A":-2,"R": 0,"N": 6,"D": 1,"C":-3,"Q": 0,"E": 0,"G": 0,"H": 1,"I":-3,"L":-3,"K": 0,"M":-2,"F":-3,"P":-2,"S": 1,"T": 0,"W":-4,"Y":-2,"V":-3},
    "D": {"A":-2,"R":-2,"N": 1,"D": 6,"C":-3,"Q": 0,"E": 2,"G":-1,"H":-1,"I":-3,"L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S": 0,"T":-1,"W":-4,"Y":-3,"V":-3},
    "C": {"A": 0,"R":-3,"N":-3,"D":-3,"C": 9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,"L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1},
    "Q": {"A":-1,"R": 1,"N": 0,"D": 0,"C":-3,"Q": 5,"E": 2,"G":-2,"H": 0,"I":-3,"L":-2,"K": 1,"M": 0,"F":-3,"P":-1,"S": 0,"T":-1,"W":-2,"Y":-1,"V":-2},
    "E": {"A":-1,"R": 0,"N": 0,"D": 2,"C":-4,"Q": 2,"E": 5,"G":-2,"H": 0,"I":-3,"L":-3,"K": 1,"M":-2,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "G": {"A": 0,"R":-2,"N": 0,"D":-1,"C":-3,"Q":-2,"E":-2,"G": 6,"H":-2,"I":-4,"L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S": 0,"T":-2,"W":-2,"Y":-3,"V":-3},
    "H": {"A":-2,"R": 0,"N": 1,"D":-1,"C":-3,"Q": 0,"E": 0,"G":-2,"H": 8,"I":-3,"L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y": 2,"V":-3},
    "I": {"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I": 4,"L": 2,"K":-3,"M": 1,"F": 0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V": 3},
    "L": {"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I": 2,"L": 4,"K":-2,"M": 2,"F": 0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V": 1},
    "K": {"A":-1,"R": 2,"N": 0,"D":-1,"C":-3,"Q": 1,"E": 1,"G":-2,"H":-1,"I":-3,"L":-2,"K": 5,"M":-1,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "M": {"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q": 0,"E":-2,"G":-3,"H":-2,"I": 1,"L": 2,"K":-1,"M": 5,"F": 0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V": 1},
    "F": {"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I": 0,"L": 0,"K":-3,"M": 0,"F": 6,"P":-4,"S":-2,"T":-2,"W": 1,"Y": 3,"V":-1},
    "P": {"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,"L":-3,"K":-1,"M":-2,"F":-4,"P": 7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2},
    "S": {"A": 1,"R":-1,"N": 1,"D": 0,"C":-1,"Q": 0,"E": 0,"G": 0,"H":-1,"I":-2,"L":-2,"K": 0,"M":-1,"F":-2,"P":-1,"S": 4,"T": 1,"W":-3,"Y":-2,"V":-2},
    "T": {"A": 0,"R":-1,"N": 0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 5,"W":-2,"Y":-2,"V": 0},
    "W": {"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,"L":-2,"K":-3,"M":-1,"F": 1,"P":-4,"S":-3,"T":-2,"W":11,"Y": 2,"V":-3},
    "Y": {"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H": 2,"I":-1,"L":-1,"K":-2,"M":-1,"F": 3,"P":-3,"S":-2,"T":-2,"W": 2,"Y": 7,"V":-1},
    "V": {"A": 0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I": 3,"L": 1,"K":-2,"M": 1,"F":-1,"P":-2,"S":-2,"T": 0,"W":-3,"Y":-1,"V": 4},
}

# PAM30 — very conservative (Dayhoff et al., 1978)
PAM30: dict[str, dict[str, int]] = {
    "A": {"A": 6,"R":-7,"N":-4,"D":-3,"C":-6,"Q":-4,"E":-2,"G":-2,"H":-7,"I":-5,"L":-6,"K":-7,"M":-5,"F":-8,"P":-2,"S": 0,"T":-1,"W":-13,"Y":-8,"V":-2},
    "R": {"A":-7,"R": 8,"N":-6,"D":-10,"C":-8,"Q":-2,"E":-9,"G":-9,"H":-2,"I":-5,"L":-8,"K": 0,"M":-4,"F":-9,"P":-4,"S":-3,"T":-6,"W":-2,"Y":-10,"V":-8},
    "N": {"A":-4,"R":-6,"N": 8,"D": 2,"C":-11,"Q":-3,"E":-2,"G":-3,"H": 0,"I":-5,"L":-7,"K":-1,"M":-9,"F":-9,"P":-6,"S": 0,"T":-2,"W":-8,"Y":-4,"V":-8},
    "D": {"A":-3,"R":-10,"N": 2,"D": 8,"C":-14,"Q":-2,"E": 2,"G":-3,"H":-4,"I":-7,"L":-12,"K":-4,"M":-11,"F":-15,"P":-8,"S":-4,"T":-5,"W":-15,"Y":-11,"V":-8},
    "C": {"A":-6,"R":-8,"N":-11,"D":-14,"C":10,"Q":-14,"E":-14,"G":-9,"H":-7,"I":-6,"L":-15,"K":-14,"M":-13,"F":-13,"P":-8,"S":-3,"T":-8,"W":-15,"Y":-4,"V":-6},
    "Q": {"A":-4,"R":-2,"N":-3,"D":-2,"C":-14,"Q": 8,"E": 1,"G":-7,"H": 1,"I":-8,"L":-5,"K":-3,"M":-4,"F":-13,"P":-3,"S":-5,"T":-5,"W":-13,"Y":-12,"V":-7},
    "E": {"A":-2,"R":-9,"N":-2,"D": 2,"C":-14,"Q": 1,"E": 8,"G":-4,"H":-5,"I":-5,"L":-9,"K":-4,"M":-7,"F":-14,"P":-5,"S":-4,"T":-6,"W":-17,"Y":-8,"V":-6},
    "G": {"A":-2,"R":-9,"N":-3,"D":-3,"C":-9,"Q":-7,"E":-4,"G": 6,"H":-9,"I":-11,"L":-10,"K":-7,"M":-8,"F":-9,"P":-6,"S":-2,"T":-6,"W":-15,"Y":-14,"V":-5},
    "H": {"A":-7,"R":-2,"N": 0,"D":-4,"C":-7,"Q": 1,"E":-5,"G":-9,"H": 9,"I":-9,"L":-6,"K":-6,"M":-10,"F":-6,"P":-4,"S":-6,"T":-7,"W":-7,"Y": 0,"V":-6},
    "I": {"A":-5,"R":-5,"N":-5,"D":-7,"C":-6,"Q":-8,"E":-5,"G":-11,"H":-9,"I": 8,"L":-1,"K":-6,"M":-1,"F":-2,"P":-8,"S":-7,"T":-2,"W":-14,"Y":-6,"V": 2},
    "L": {"A":-6,"R":-8,"N":-7,"D":-12,"C":-15,"Q":-5,"E":-9,"G":-10,"H":-6,"I":-1,"L": 7,"K":-8,"M": 1,"F":-3,"P":-7,"S":-8,"T":-7,"W":-6,"Y":-7,"V":-2},
    "K": {"A":-7,"R": 0,"N":-1,"D":-4,"C":-14,"Q":-3,"E":-4,"G":-7,"H":-6,"I":-6,"L":-8,"K": 7,"M":-2,"F":-14,"P":-6,"S":-4,"T":-3,"W":-12,"Y":-9,"V":-9},
    "M": {"A":-5,"R":-4,"N":-9,"D":-11,"C":-13,"Q":-4,"E":-7,"G":-8,"H":-10,"I":-1,"L": 1,"K":-2,"M": 11,"F":-4,"P":-8,"S":-5,"T":-4,"W":-5,"Y":-11,"V":-1},
    "F": {"A":-8,"R":-9,"N":-9,"D":-15,"C":-13,"Q":-13,"E":-14,"G":-9,"H":-6,"I":-2,"L":-3,"K":-14,"M":-4,"F": 9,"P":-10,"S":-6,"T":-9,"W":-4,"Y": 2,"V":-8},
    "P": {"A":-2,"R":-4,"N":-6,"D":-8,"C":-8,"Q":-3,"E":-5,"G":-6,"H":-4,"I":-8,"L":-7,"K":-6,"M":-8,"F":-10,"P": 8,"S":-2,"T":-4,"W":-14,"Y":-13,"V":-6},
    "S": {"A": 0,"R":-3,"N": 0,"D":-4,"C":-3,"Q":-5,"E":-4,"G":-2,"H":-6,"I":-7,"L":-8,"K":-4,"M":-5,"F":-6,"P":-2,"S": 6,"T": 0,"W":-5,"Y":-7,"V":-6},
    "T": {"A":-1,"R":-6,"N":-2,"D":-5,"C":-8,"Q":-5,"E":-6,"G":-6,"H":-7,"I":-2,"L":-7,"K":-3,"M":-4,"F":-9,"P":-4,"S": 0,"T": 7,"W":-13,"Y":-6,"V":-3},
    "W": {"A":-13,"R":-2,"N":-8,"D":-15,"C":-15,"Q":-13,"E":-17,"G":-15,"H":-7,"I":-14,"L":-6,"K":-12,"M":-5,"F":-4,"P":-14,"S":-5,"T":-13,"W":13,"Y":-5,"V":-15},
    "Y": {"A":-8,"R":-10,"N":-4,"D":-11,"C":-4,"Q":-12,"E":-8,"G":-14,"H": 0,"I":-6,"L":-7,"K":-9,"M":-11,"F": 2,"P":-13,"S":-7,"T":-6,"W":-5,"Y": 9,"V":-7},
    "V": {"A":-2,"R":-8,"N":-8,"D":-8,"C":-6,"Q":-7,"E":-6,"G":-5,"H":-6,"I": 2,"L":-2,"K":-9,"M":-1,"F":-8,"P":-6,"S":-6,"T":-3,"W":-15,"Y":-7,"V": 7},
}

# PAM250 — permissive (Dayhoff et al., 1978)
PAM250: dict[str, dict[str, int]] = {
    "A": {"A": 2,"R":-2,"N": 0,"D": 0,"C":-2,"Q": 0,"E": 0,"G": 1,"H":-1,"I":-1,"L":-2,"K":-1,"M":-1,"F":-3,"P": 1,"S": 1,"T": 1,"W":-6,"Y":-3,"V": 0},
    "R": {"A":-2,"R": 6,"N": 0,"D":-1,"C":-4,"Q": 1,"E":-1,"G":-3,"H": 2,"I":-2,"L":-3,"K": 3,"M": 0,"F":-4,"P": 0,"S": 0,"T":-1,"W": 2,"Y":-4,"V":-2},
    "N": {"A": 0,"R": 0,"N": 2,"D": 2,"C":-4,"Q": 1,"E": 1,"G": 0,"H": 2,"I":-2,"L":-3,"K": 1,"M":-2,"F":-3,"P": 0,"S": 1,"T": 0,"W":-4,"Y":-2,"V":-2},
    "D": {"A": 0,"R":-1,"N": 2,"D": 4,"C":-5,"Q": 2,"E": 3,"G": 1,"H": 1,"I":-2,"L":-4,"K": 0,"M":-3,"F":-6,"P":-1,"S": 0,"T": 0,"W":-7,"Y":-4,"V":-2},
    "C": {"A":-2,"R":-4,"N":-4,"D":-5,"C":12,"Q":-5,"E":-5,"G":-3,"H":-3,"I":-2,"L":-6,"K":-5,"M":-5,"F":-4,"P":-3,"S": 0,"T":-2,"W":-8,"Y": 0,"V":-2},
    "Q": {"A": 0,"R": 1,"N": 1,"D": 2,"C":-5,"Q": 4,"E": 2,"G":-1,"H": 3,"I":-2,"L":-2,"K": 1,"M":-1,"F":-5,"P": 0,"S":-1,"T":-1,"W":-5,"Y":-4,"V":-2},
    "E": {"A": 0,"R":-1,"N": 1,"D": 3,"C":-5,"Q": 2,"E": 4,"G": 0,"H": 1,"I":-2,"L":-3,"K": 0,"M":-2,"F":-5,"P":-1,"S": 0,"T": 0,"W":-7,"Y":-4,"V":-2},
    "G": {"A": 1,"R":-3,"N": 0,"D": 1,"C":-3,"Q":-1,"E": 0,"G": 5,"H":-2,"I":-3,"L":-4,"K":-2,"M":-3,"F":-5,"P": 0,"S": 1,"T": 0,"W":-7,"Y":-5,"V":-1},
    "H": {"A":-1,"R": 2,"N": 2,"D": 1,"C":-3,"Q": 3,"E": 1,"G":-2,"H": 6,"I":-2,"L":-2,"K": 0,"M":-2,"F":-2,"P": 0,"S":-1,"T":-1,"W":-3,"Y": 0,"V":-2},
    "I": {"A":-1,"R":-2,"N":-2,"D":-2,"C":-2,"Q":-2,"E":-2,"G":-3,"H":-2,"I": 5,"L": 2,"K":-2,"M": 2,"F": 1,"P":-2,"S":-1,"T": 0,"W":-5,"Y":-1,"V": 4},
    "L": {"A":-2,"R":-3,"N":-3,"D":-4,"C":-6,"Q":-2,"E":-3,"G":-4,"H":-2,"I": 2,"L": 6,"K":-3,"M": 4,"F": 2,"P":-3,"S":-3,"T":-2,"W":-2,"Y":-1,"V": 2},
    "K": {"A":-1,"R": 3,"N": 1,"D": 0,"C":-5,"Q": 1,"E": 0,"G":-2,"H": 0,"I":-2,"L":-3,"K": 5,"M": 0,"F":-5,"P":-1,"S": 0,"T": 0,"W":-3,"Y":-4,"V":-2},
    "M": {"A":-1,"R": 0,"N":-2,"D":-3,"C":-5,"Q":-1,"E":-2,"G":-3,"H":-2,"I": 2,"L": 4,"K": 0,"M": 6,"F": 0,"P":-2,"S":-2,"T":-1,"W":-4,"Y":-2,"V": 2},
    "F": {"A":-3,"R":-4,"N":-3,"D":-6,"C":-4,"Q":-5,"E":-5,"G":-5,"H":-2,"I": 1,"L": 2,"K":-5,"M": 0,"F": 9,"P":-5,"S":-3,"T":-3,"W": 0,"Y": 7,"V":-1},
    "P": {"A": 1,"R": 0,"N": 0,"D":-1,"C":-3,"Q": 0,"E":-1,"G": 0,"H": 0,"I":-2,"L":-3,"K":-1,"M":-2,"F":-5,"P": 6,"S": 1,"T": 0,"W":-6,"Y":-5,"V":-1},
    "S": {"A": 1,"R": 0,"N": 1,"D": 0,"C": 0,"Q":-1,"E": 0,"G": 1,"H":-1,"I":-1,"L":-3,"K": 0,"M":-2,"F":-3,"P": 1,"S": 2,"T": 1,"W":-2,"Y":-3,"V":-1},
    "T": {"A": 1,"R":-1,"N": 0,"D": 0,"C":-2,"Q":-1,"E": 0,"G": 0,"H":-1,"I": 0,"L":-2,"K": 0,"M":-1,"F":-3,"P": 0,"S": 1,"T": 3,"W":-5,"Y":-3,"V": 0},
    "W": {"A":-6,"R": 2,"N":-4,"D":-7,"C":-8,"Q":-5,"E":-7,"G":-7,"H":-3,"I":-5,"L":-2,"K":-3,"M":-4,"F": 0,"P":-6,"S":-2,"T":-5,"W":17,"Y": 0,"V":-6},
    "Y": {"A":-3,"R":-4,"N":-2,"D":-4,"C": 0,"Q":-4,"E":-4,"G":-5,"H": 0,"I":-1,"L":-1,"K":-4,"M":-2,"F": 7,"P":-5,"S":-3,"T":-3,"W": 0,"Y":10,"V":-2},
    "V": {"A": 0,"R":-2,"N":-2,"D":-2,"C":-2,"Q":-2,"E":-2,"G":-1,"H":-2,"I": 4,"L": 2,"K":-2,"M": 2,"F":-1,"P":-1,"S":-1,"T": 0,"W":-6,"Y":-2,"V": 4},
}

# Physicochemical property groups (Fjell et al., 2011)
# Used by property_preserving mutation
PROPERTY_GROUPS: dict[str, list[str]] = {
    "nonpolar_aliphatic": ["G", "A", "V", "L", "I", "P", "M"],
    "aromatic":           ["F", "Y", "W"],
    "polar_uncharged":    ["S", "T", "C", "N", "Q"],
    "positive":           ["K", "R", "H"],
    "negative":           ["D", "E"],
}

# Reverse lookup: AA → group name
_AA_TO_GROUP: dict[str, str] = {
    aa: grp
    for grp, members in PROPERTY_GROUPS.items()
    for aa in members
}

_CANONICAL: list[str] = sorted(BLOSUM62)
_CATIONIC:  list[str] = ["K", "R", "H"]

_MATRICES = {
    "blosum62": BLOSUM62,
    "pam30":    PAM30,
    "pam250":   PAM250,
}


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------


def mutate(
    sequence: str,
    mutation_rate: float = 0.1,
    strategy: str = "blosum62",
) -> str:
    """Apply a point-substitution mutation strategy to a peptide sequence.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence (canonical AAs only).
    mutation_rate : float
        Fraction of positions eligible for mutation (0 < rate ≤ 1).
        Exactly ``max(1, floor(len * rate))`` positions are sampled.
    strategy : str
        One of the strategies listed in the module docstring.
        Default: ``"blosum62"``.

    Returns
    -------
    str
        Mutated sequence of the same length as the input.

    Raises
    ------
    ValueError
        If the sequence is empty, contains non-canonical AAs, or
        ``mutation_rate`` / ``strategy`` are invalid.
    """
    _validate_sequence(sequence)
    if not 0 < mutation_rate <= 1:
        raise ValueError(f"mutation_rate must be in (0, 1], got {mutation_rate}.")

    if strategy not in _SUBSTITUTION_STRATEGIES:
        raise ValueError(
            f"Unknown mutation strategy '{strategy}'. "
            f"Available: {list(_SUBSTITUTION_STRATEGIES)}"
        )

    n_muts   = max(1, int(len(sequence) * mutation_rate))
    n_muts   = np.random.randint(1, n_muts + 1)
    positions = np.random.choice(len(sequence), n_muts, replace=False)

    seq = list(sequence)
    sub_fn = _SUBSTITUTION_STRATEGIES[strategy]
    for pos in positions:
        seq[pos] = sub_fn(seq[pos])
    return "".join(seq)


# ---------------------------------------------------------------------------
# Substitution functions (one per strategy)
# ---------------------------------------------------------------------------


def _sub_matrix_greedy(matrix: dict) -> callable:
    """Return a greedy substitution function for a given matrix."""
    def _sub(aa: str) -> str:
        row = matrix.get(aa, {})
        candidates = [a for a in row if a != aa]
        return max(candidates, key=lambda a: row[a]) if candidates else aa
    return _sub


def _sub_blosum62_weighted(aa: str) -> str:
    """Sample from a softmax over BLOSUM62 scores (probabilistic)."""
    row = BLOSUM62.get(aa, {})
    candidates = [a for a in row if a != aa]
    if not candidates:
        return aa
    scores = np.array([row[a] for a in candidates], dtype=float)
    # Softmax to get probabilities
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()
    return np.random.choice(candidates, p=probs)


def _sub_property(aa: str) -> str:
    """Replace within the same physicochemical group (Fjell et al., 2011)."""
    grp = _AA_TO_GROUP.get(aa)
    if grp is None:
        return aa
    members = [a for a in PROPERTY_GROUPS[grp] if a != aa]
    return random.choice(members) if members else aa


def _sub_charge_biased(aa: str) -> str:
    """Charge-biased substitution — EXPERIMENTAL.

    With probability 0.7 substitutes toward a cationic residue (K, R, H),
    otherwise falls back to random substitution.
    """
    if random.random() < 0.7:
        return random.choice(_CATIONIC)
    candidates = [a for a in _CANONICAL if a != aa]
    return random.choice(candidates) if candidates else aa


def _sub_random(aa: str) -> str:
    """Completely random canonical AA replacement (Goldberg, 1989)."""
    candidates = [a for a in _CANONICAL if a != aa]
    return random.choice(candidates) if candidates else aa


_SUBSTITUTION_STRATEGIES: dict[str, callable] = {
    "blosum62":          _sub_matrix_greedy(BLOSUM62),
    "blosum62_weighted": _sub_blosum62_weighted,
    "pam30":             _sub_matrix_greedy(PAM30),
    "pam250":            _sub_matrix_greedy(PAM250),
    "property":          _sub_property,
    "charge_biased":     _sub_charge_biased,   # EXPERIMENTAL
    "random":            _sub_random,
}

_EXPERIMENTAL_STRATEGIES = {"charge_biased"}


# ---------------------------------------------------------------------------
# Indel operators
# ---------------------------------------------------------------------------


def insert_aa(sequence: str) -> str:
    """Insert one random canonical AA at a random position.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence.

    Returns
    -------
    str
        Sequence one residue longer than the input.
    """
    _validate_sequence(sequence)
    aa  = random.choice(_CANONICAL)
    pos = random.randint(0, len(sequence))
    return sequence[:pos] + aa + sequence[pos:]


def delete_aa(sequence: str) -> str:
    """Remove one residue at a random position.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence (must be ≥ 2 residues).

    Returns
    -------
    str
        Sequence one residue shorter than the input.

    Raises
    ------
    ValueError
        If the sequence has fewer than 2 residues.
    """
    if len(sequence) < 2:
        raise ValueError("Sequence must be at least 2 residues to delete one.")
    _validate_sequence(sequence)
    pos = random.randint(0, len(sequence) - 1)
    return sequence[:pos] + sequence[pos + 1:]


def available_strategies() -> list[str]:
    """Return all registered substitution strategy names."""
    return list(_SUBSTITUTION_STRATEGIES)


def experimental_strategies() -> list[str]:
    """Return strategies marked as experimental."""
    return list(_EXPERIMENTAL_STRATEGIES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("Sequence must not be empty.")
    invalid = set(sequence.upper()) - set(_CANONICAL)
    if invalid:
        raise ValueError(f"Non-canonical amino acids found: {sorted(invalid)}")
