"""
Honest baselines to compare the Fractal Predictive Cortex against.
==================================================================

A methods paper is only as convincing as what it is measured *against*. These
are the standard, respected alternatives:

* ``KmerMarkov``    -- a fixed-order (k-mer) Markov model. The canonical
                       alignment-free sequence model; FPC's direct competitor
                       for prediction / compression.
* ``gzip_bits``     -- a general-purpose compressor as a model-free code-length
                       reference (the "can you beat off-the-shelf zip" bar).
* ``kmer_vector``   -- k-mer frequency features for classical ML classifiers /
                       distance-based clustering baselines.

All pure-Python / NumPy; ``scikit-learn`` is used only in ``bench.py`` for the
classical-ML baselines.
"""

from __future__ import annotations

import gzip
import numpy as np

from fpc.alphabet import Alphabet, DNA


class KmerMarkov:
    """A fixed-order Markov model over an alphabet -- the standard baseline.

    Unlike the cortex (which *chooses* its effective order per region), this
    uses one global order ``k`` for every context, with Laplace smoothing.
    """

    def __init__(self, alphabet: Alphabet = DNA, k: int = 4, alpha: float = 0.5):
        self.alphabet = alphabet
        self.k = int(k)
        self.a = alphabet.size
        self.alpha = float(alpha)
        self.counts = np.zeros((self.a ** self.k, self.a), dtype=np.float64)
        self.global_counts = np.full(self.a, alpha, dtype=np.float64)
        self._pk = self.a ** self.k

    def _contexts(self, ids: np.ndarray):
        """Yield (context_code_or_None, symbol) for scorable positions."""
        a, k = self.a, self.k
        n = len(ids)
        for t in range(n):
            s = int(ids[t])
            if s >= a:
                continue
            if t < k:
                yield None, s
                continue
            window = ids[t - k:t]
            if (window >= a).any():          # wildcard in context -> unknown
                yield None, s
                continue
            code = 0
            for w in window:
                code = code * a + int(w)
            yield code, s

    def fit(self, ids: np.ndarray) -> "KmerMarkov":
        for code, s in self._contexts(np.asarray(ids)):
            self.global_counts[s] += 1.0
            if code is not None:
                self.counts[code, s] += 1.0
        return self

    def _dist(self, code):
        if code is None:
            g = self.global_counts
            return g / g.sum()
        row = self.counts[code]
        return (row + self.alpha) / (row.sum() + self.a * self.alpha)

    def cross_entropy(self, ids: np.ndarray) -> float:
        tot, n = 0.0, 0
        for code, s in self._contexts(np.asarray(ids)):
            p = self._dist(code)[s]
            tot += -np.log2(p) if p > 0 else 60.0
            n += 1
        return tot / n if n else float("nan")

    def accuracy(self, ids: np.ndarray) -> float:
        hit, n = 0, 0
        for code, s in self._contexts(np.asarray(ids)):
            if int(np.argmax(self._dist(code))) == s:
                hit += 1
            n += 1
        return hit / n if n else float("nan")


def nrc_nccd_distance_matrices(sequences, alphabet: Alphabet = DNA, k: int = 9):
    """Prior-art cross-entropy distances: NRC and NCCD (finite-context substrate).

    Faithful single-order finite-context-model realisations of the two published
    distances the FPC distance must be compared against:

      * NRC  (Normalized Relative Compression; Pratas, Silva & Pinho 2018):
        train a Markov model on y, freeze it, measure bits/symbol of x under it,
        normalised by log2|A|. Symmetrised for a distance.
      * NCCD (Normalized Conditional Compression Distance; Pratas & Pinho 2014):
        NCCD(x,y) = max{C(x|y), C(y|x)} / max{C(x), C(y)}, where C(x|y) is the
        code length of x under y's model and C(x) is x under its own model.

    (The published versions use a *mixture* of context models up to k~20 with
    substitution tolerance; here we use one order-k dense FCM — a faithful,
    memory-bounded approximation, matched on k. Returns (D_nrc, D_nccd).)
    """
    import math
    ids = [alphabet.encode(s) for s in sequences]
    n = len(sequences)
    models = [KmerMarkov(alphabet, k).fit(x) for x in ids]
    ce = np.zeros((n, n))                       # ce[i,j] = bits/symbol of seq_i under model_j
    for i in range(n):
        for j in range(n):
            ce[i, j] = models[j].cross_entropy(ids[i])
    lengths = np.array([len(x) for x in ids], dtype=np.float64)
    log_a = math.log2(alphabet.size)

    nrc = ce / log_a
    c_self = np.array([ce[i, i] * lengths[i] for i in range(n)])   # C(x) under own model
    d_nrc = np.zeros((n, n))
    d_nccd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_nrc[i, j] = d_nrc[j, i] = 0.5 * (nrc[i, j] + nrc[j, i])
            c_xy = ce[i, j] * lengths[i]        # C(x|y)
            c_yx = ce[j, i] * lengths[j]        # C(y|x)
            d_nccd[i, j] = d_nccd[j, i] = max(c_xy, c_yx) / max(c_self[i], c_self[j])
    return d_nrc, d_nccd


def gzip_bits(seq: str) -> float:
    """Bits per symbol achieved by gzip -- a model-free code-length reference."""
    raw = seq.encode("ascii", "ignore")
    if not raw:
        return float("nan")
    comp = gzip.compress(raw, compresslevel=9)
    return 8.0 * len(comp) / len(raw)


def kmer_vector(seq: str, alphabet: Alphabet = DNA, k: int = 4) -> np.ndarray:
    """Normalised k-mer frequency vector (features for classical ML / distance)."""
    a = alphabet.size
    ids = alphabet.encode(seq)
    vec = np.zeros(a ** k, dtype=np.float64)
    n = len(ids)
    for t in range(n - k + 1):
        window = ids[t:t + k]
        if (window >= a).any():
            continue
        code = 0
        for w in window:
            code = code * a + int(w)
        vec[code] += 1.0
    tot = vec.sum()
    return vec / tot if tot > 0 else vec
