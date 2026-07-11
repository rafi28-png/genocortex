"""
The four abilities -- all from the one brain.
=============================================

This module is the friendly front door. Every function here is a thin wrapper
that turns the single mechanism of the :class:`FractalPredictiveCortex`
(minimise surprise) into a concrete capability:

    predict   -> ``complete`` / ``evaluate_prediction``
    detect    -> ``detect_anomalies`` / ``detect_against_reference``
    classify  -> ``FPCClassifier``
    discover  -> ``distance_matrix`` / ``mds_embedding`` / ``cluster``

Nothing here re-implements intelligence; it re-expresses the same surprise
signal for four different questions. That unification is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .alphabet import Alphabet, DNA
from .cortex import FractalPredictiveCortex
from .surprise import SurpriseTrack, Anomaly

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    _HAVE_SCIPY_CLUSTER = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY_CLUSTER = False


# ===================================================================== #
#  1. PREDICT                                                           #
# ===================================================================== #
def complete(cortex: FractalPredictiveCortex, prefix: str, length: int,
             sample: bool = False, temperature: float = 1.0) -> str:
    """Continue ``prefix`` for ``length`` more symbols using the trained brain."""
    ids = cortex.alphabet.encode(prefix)
    out = cortex.generate(ids, length, sample=sample, temperature=temperature)
    return cortex.alphabet.decode(out)


def evaluate_prediction(cortex: FractalPredictiveCortex, sequence: str) -> dict:
    """Next-symbol prediction quality on a held-out sequence (no learning).

    Returns top-1 accuracy (argmax hits), cross-entropy (bits/symbol) and
    perplexity. Top-1 accuracy is what an ML reviewer will ask for; perplexity
    is the information-theoretic twin.
    """
    ids = cortex.alphabet.encode(sequence)
    a = cortex.a
    C = cortex.embedding.context_points(ids)
    hits = 0
    n = 0
    track = np.full(len(ids), np.nan)
    for t in range(len(ids)):
        s = int(ids[t])
        if s >= a:
            continue
        p, _, _ = cortex.predict_dist(C[t])
        ps = p[s]
        track[t] = -np.log2(ps) if ps > 0 else 60.0
        if int(np.argmax(p)) == s:
            hits += 1
        n += 1
    st = SurpriseTrack(track)
    return {
        "accuracy": hits / n if n else float("nan"),
        "cross_entropy": st.mean_bits,
        "perplexity": st.perplexity,
        "n": n,
    }


# ===================================================================== #
#  2. DETECT                                                            #
# ===================================================================== #
@dataclass
class DetectionResult:
    track: SurpriseTrack
    anomalies: list          # list[Anomaly]
    cortex: FractalPredictiveCortex


def detect_anomalies(sequence: str, alphabet: Alphabet = DNA, order: int = 8,
                     window: int = 25, z_threshold: float = 4.0,
                     alpha: float = 0.05, epochs: int = 1, **cortex_kwargs) -> DetectionResult:
    """Two-pass self-anomaly detection on a single sequence.

    Pass 1 learns the sequence's own statistics; pass 2 scores it *frozen* so
    the surprise trace is free of cold-start bias. Regions the model still finds
    surprising after learning the rest are the candidate mutations / inserts.
    """
    cortex = FractalPredictiveCortex(alphabet, order=order, **cortex_kwargs)
    ids = alphabet.encode(sequence)
    cortex.fit(ids, epochs=epochs)
    track = cortex.score(ids)                 # frozen second pass
    anoms = track.anomalies(window=window, z_threshold=z_threshold, alpha=alpha)
    return DetectionResult(track=track, anomalies=anoms, cortex=cortex)


def detect_against_reference(reference: str, query: str, alphabet: Alphabet = DNA,
                             order: int = 8, window: int = 25,
                             z_threshold: float = 4.0, alpha: float = 0.05,
                             epochs: int = 1, **cortex_kwargs) -> DetectionResult:
    """Learn a reference genome, then flag stretches of a query it cannot explain.

    This is the "foreign DNA / horizontal transfer / contamination" detector:
    the reference brain is surprised precisely where the query departs from the
    reference's statistics.
    """
    cortex = FractalPredictiveCortex(alphabet, order=order, **cortex_kwargs)
    cortex.fit(alphabet.encode(reference), epochs=epochs)
    track = cortex.score(alphabet.encode(query))
    anoms = track.anomalies(window=window, z_threshold=z_threshold, alpha=alpha)
    return DetectionResult(track=track, anomalies=anoms, cortex=cortex)


# ===================================================================== #
#  3. CLASSIFY                                                          #
# ===================================================================== #
class FPCClassifier:
    """A generative, minimum-description-length classifier.

    Train one cortex per class; classify a query by *which brain is least
    surprised* by it (lowest cross-entropy). With equal priors this is the
    Bayes-optimal decision rule for the learned generative models, and it needs
    **no negative examples and no feature engineering**.
    """

    def __init__(self, alphabet: Alphabet = DNA, order: int = 8, epochs: int = 1,
                 **cortex_kwargs):
        self.alphabet = alphabet
        self.order = order
        self.epochs = epochs
        self.cortex_kwargs = cortex_kwargs
        self.models: dict = {}
        self.classes_: list = []

    def fit(self, sequences, labels) -> "FPCClassifier":
        by_label: dict = {}
        for seq, lab in zip(sequences, labels):
            by_label.setdefault(lab, []).append(seq)
        self.classes_ = sorted(by_label)
        for lab in self.classes_:
            cortex = FractalPredictiveCortex(self.alphabet, order=self.order,
                                             **self.cortex_kwargs)
            for seq in by_label[lab]:
                cortex.fit(self.alphabet.encode(seq), epochs=self.epochs)
            self.models[lab] = cortex
        return self

    def score_matrix(self, sequences) -> np.ndarray:
        """``(n_sequences, n_classes)`` matrix of cross-entropies (lower=better)."""
        M = np.zeros((len(sequences), len(self.classes_)))
        for i, seq in enumerate(sequences):
            ids = self.alphabet.encode(seq)
            for j, lab in enumerate(self.classes_):
                M[i, j] = self.models[lab].cross_entropy(ids)
        return M

    def predict(self, sequences) -> list:
        M = self.score_matrix(sequences)
        idx = np.argmin(M, axis=1)
        return [self.classes_[j] for j in idx]

    def predict_proba(self, sequences) -> np.ndarray:
        """Turn cross-entropies into class posteriors via a softmax over -bits.

        (Assumes equal priors; the temperature is 1 bit.) Useful for confidence,
        ROC curves, and rejecting low-confidence calls.
        """
        M = self.score_matrix(sequences)
        logits = -M
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return p


def classification_accuracy(pred, true) -> float:
    pred = list(pred); true = list(true)
    return float(np.mean([p == t for p, t in zip(pred, true)]))


# ===================================================================== #
#  4. DISCOVER (unsupervised structure)                                 #
# ===================================================================== #
def distance_matrix(sequences, alphabet: Alphabet = DNA, order: int = 6,
                    epochs: int = 1, **cortex_kwargs) -> np.ndarray:
    """Symmetric, alignment-free cross-entropy distance between sequences.

    For each sequence we train its own cortex, then measure the *excess* bits
    each model spends on every other sequence relative to that sequence's own
    model::

        d(A, B) = 1/2 [ (CE_A(B) - CE_B(B)) + (CE_B(A) - CE_A(A)) ]

    This approximates a symmetrised relative entropy (a Kullback-Leibler-style
    divergence) between the sequences' generative statistics: 0 for identical
    statistics, growing with divergence. It is a drop-in distance for
    phylogenetics and clustering, with no alignment.
    """
    n = len(sequences)
    ids = [alphabet.encode(s) for s in sequences]
    cortices = []
    for x in ids:
        c = FractalPredictiveCortex(alphabet, order=order, **cortex_kwargs)
        c.fit(x, epochs=epochs)
        cortices.append(c)
    # ce[i][j] = cross-entropy of sequence j under model i.
    ce = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ce[i, j] = cortices[i].cross_entropy(ids[j])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.5 * ((ce[i, j] - ce[j, j]) + (ce[j, i] - ce[i, i]))
            D[i, j] = D[j, i] = max(0.0, d)
    return D


def mds_embedding(D: np.ndarray, dims: int = 3) -> np.ndarray:
    """Classical multidimensional scaling of a distance matrix to ``dims``-D.

    Pure NumPy. Returns coordinates ``(n, dims)`` suitable for the 3-D viewer.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = 0.5 * (B + B.T)                      # symmetrise against round-off
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order][:dims]
    vecs = vecs[:, order][:, :dims]
    vals = np.clip(vals, 0, None)
    return vecs * np.sqrt(vals)[None, :]


def cluster(D: np.ndarray, n_clusters: int, method: str = "average") -> np.ndarray:
    """Hierarchical clustering of a distance matrix into ``n_clusters`` groups."""
    if not _HAVE_SCIPY_CLUSTER:  # pragma: no cover
        raise RuntimeError("scipy is required for cluster()")
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    condensed = D[iu]
    Z = linkage(condensed, method=method)
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def clustering_purity(pred, true) -> float:
    """Fraction of items in the majority true-class of their predicted cluster."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    total = 0
    for c in np.unique(pred):
        members = true[pred == c]
        if len(members):
            vals, counts = np.unique(members, return_counts=True)
            total += counts.max()
    return float(total / len(true))
