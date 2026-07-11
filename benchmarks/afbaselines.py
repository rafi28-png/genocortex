"""
Standard alignment-free baselines a reviewer expects.
======================================================

Faithful, dependency-free re-implementations of the *published algorithms*
(not wrappers around the C tools), so the comparison is against what the field
actually uses -- and still runs offline on Windows:

* ``mash_distance``  -- Mash (Ondov et al. 2016): bottom-s MinHash sketch of
                        canonical k-mers -> Jaccard estimate -> Mash distance
                        ``D = -(1/k) ln( 2J / (1+J) )``.
* ``ffp_distance``   -- Feature Frequency Profiles (Sims et al. 2009):
                        k-mer frequency profiles compared by the Jensen-Shannon
                        distance (sqrt of JSD; a true metric).

Both return the same kind of pairwise distance FPC's cross-entropy distance
does, so they slot straight into the phylogeny / clustering benchmarks. We are
explicit in the paper that these are re-implementations of the methods, matched
on k, not the original binaries.
"""

from __future__ import annotations

import numpy as np

from fpc.alphabet import Alphabet, DNA

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _fnv1a64(s: str) -> int:
    """Deterministic 64-bit FNV-1a hash (Python's builtin hash is salted)."""
    h = 0xCBF29CE484222325
    for ch in s.encode("ascii", "ignore"):
        h ^= ch
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _canonical(kmer: str) -> str:
    """Canonical k-mer for DNA = min(kmer, reverse-complement)."""
    rc = kmer.translate(_COMPLEMENT)[::-1]
    return kmer if kmer <= rc else rc


# --------------------------------------------------------------------- #
#  Mash (MinHash)                                                       #
# --------------------------------------------------------------------- #
def mash_sketch(seq: str, k: int = 14, sketch_size: int = 400,
                canonical: bool = True) -> set:
    """Bottom-``sketch_size`` MinHash sketch of a sequence's k-mers."""
    seq = seq.upper()
    hashes = set()
    n = len(seq)
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        if any(c not in "ACGTU" for c in kmer) and canonical:
            # allow non-DNA (e.g. protein) only when canonical is off
            if canonical:
                continue
        if canonical:
            kmer = _canonical(kmer)
        hashes.add(_fnv1a64(kmer))
    return set(sorted(hashes)[:sketch_size])


def mash_distance_from_sketches(a: set, b: set, k: int, sketch_size: int) -> float:
    """Mash distance between two bottom-s sketches."""
    if not a or not b:
        return 1.0
    union_bottom = sorted(a | b)[:sketch_size]
    shared = sum(1 for h in union_bottom if h in a and h in b)
    j = shared / len(union_bottom)
    if j <= 0.0:
        return 1.0
    if j >= 1.0:
        return 0.0
    return -np.log(2.0 * j / (1.0 + j)) / k


def mash_distance_matrix(sequences, k: int = 14, sketch_size: int = 400,
                         canonical: bool = True) -> np.ndarray:
    sketches = [mash_sketch(s, k, sketch_size, canonical) for s in sequences]
    n = len(sequences)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = mash_distance_from_sketches(sketches[i], sketches[j], k, sketch_size)
            D[i, j] = D[j, i] = d
    return D


# --------------------------------------------------------------------- #
#  FFP (Feature Frequency Profiles + Jensen-Shannon distance)           #
# --------------------------------------------------------------------- #
def _kmer_profile(seq: str, alphabet: Alphabet, k: int) -> np.ndarray:
    a = alphabet.size
    ids = alphabet.encode(seq)
    vec = np.zeros(a ** k, dtype=np.float64)
    for t in range(len(ids) - k + 1):
        window = ids[t:t + k]
        if (window >= a).any():
            continue
        code = 0
        for w in window:
            code = code * a + int(w)
        vec[code] += 1.0
    tot = vec.sum()
    return vec / tot if tot > 0 else vec


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))
    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(0.0, jsd)))


def ffp_distance_matrix(sequences, alphabet: Alphabet = DNA, k: int = 6) -> np.ndarray:
    profiles = [_kmer_profile(s, alphabet, k) for s in sequences]
    n = len(sequences)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _js_distance(profiles[i], profiles[j])
            D[i, j] = D[j, i] = d
    return D
