"""
The four abilities -- all from the one brain.
=============================================

This module is the friendly front door. Every function here is a thin wrapper
that turns the single mechanism of the :class:`FractalPredictiveCortex`
(minimise surprise) into a concrete capability:

    predict   -> ``complete`` / ``evaluate_prediction``
    detect    -> ``detect_anomalies`` / ``detect_against_reference``
    classify  -> ``FPCClassifier`` / ``ClassificationCall``
    discover  -> ``distance_matrix`` / ``mds_embedding`` / ``cluster``

Nothing here re-implements intelligence; it re-expresses the same surprise
signal for four different questions. That unification is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np

from .alphabet import Alphabet, DNA
from .cortex import FractalPredictiveCortex, MODEL_FORMAT_VERSION
from .io_fasta import Record
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
    C = cortex.embedding.context_points(ids, use_fast=cortex.use_fast)
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
    anomalies: list[Anomaly]
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
                             epochs: int = 1, double_strand: bool = False,
                             **cortex_kwargs) -> DetectionResult:
    """Learn a reference genome, then flag stretches of a query it cannot explain.

    This is the "foreign DNA / horizontal transfer / contamination" detector:
    the reference brain is surprised precisely where the query departs from the
    reference's statistics.
    """
    cortex = FractalPredictiveCortex(alphabet, order=order, **cortex_kwargs)
    cortex.fit(alphabet.encode(reference), epochs=epochs, double_strand=double_strand)
    track = cortex.score(alphabet.encode(query))
    anoms = track.anomalies(window=window, z_threshold=z_threshold, alpha=alpha)
    return DetectionResult(track=track, anomalies=anoms, cortex=cortex)


# ===================================================================== #
#  3. CLASSIFY                                                          #
# ===================================================================== #
@dataclass
class ClassificationCall:
    """Detailed minimum-description-length classification call for one query."""
    id: str
    predicted_class: str
    strand: str                 # '+' or '-'
    bits_per_base: float        # cross-entropy of winning class
    posterior: float            # softmax posterior over -bits_per_base
    runner_up: str | None       # second-best class (or None if 1 class)
    delta_bits: float           # runner_up_bits - bits_per_base (bits/symbol margin)


class FPCClassifier:
    """A generative, minimum-description-length classifier.

    Train one cortex per class; classify a query by *which brain is least
    surprised* by it (lowest cross-entropy). With equal priors this is the
    Bayes-optimal decision rule for the learned generative models, and it needs
    **no negative examples and no feature engineering**.

    When ``strand_aware=True`` (default for DNA/RNA), each query sequence is
    evaluated on both ``+`` and ``-`` (reverse-complement) strands so unoriented
    sequencing reads and contigs classify accurately and receive an explicit
    strand call.
    """

    def __init__(
        self,
        alphabet: Alphabet = DNA,
        order: int = 8,
        epochs: int = 1,
        strand_aware: bool | None = None,
        double_strand_train: bool = False,
        **cortex_kwargs,
    ):
        self.alphabet = alphabet
        self.order = order
        self.epochs = epochs
        self.strand_aware = (
            bool(alphabet.is_complementary) if strand_aware is None else bool(strand_aware)
        )
        self.double_strand_train = bool(double_strand_train)
        self.cortex_kwargs = cortex_kwargs
        self.models: dict[str, FractalPredictiveCortex] = {}
        self.classes_: list = []

    def fit(self, sequences, labels) -> "FPCClassifier":
        by_label: dict = {}
        for seq, lab in zip(sequences, labels):
            by_label.setdefault(lab, []).append(seq)
        self.classes_ = sorted(by_label)
        self.models = {}
        for lab in self.classes_:
            cortex = FractalPredictiveCortex(
                self.alphabet, order=self.order, **self.cortex_kwargs
            )
            for seq in by_label[lab]:
                cortex.fit(
                    self.alphabet.encode(seq),
                    epochs=self.epochs,
                    double_strand=self.double_strand_train,
                )
            self.models[lab] = cortex
        return self

    def score_matrix_with_strands(self, sequences) -> tuple[np.ndarray, list[list[str]]]:
        """Return ``(M, strands)`` where ``M[i, j]`` is best bits/base and ``strands[i][j]`` is ``'+'``/``'-'``."""
        if not self.classes_:
            raise RuntimeError("FPCClassifier must be fitted before scoring")
        n_seq = len(sequences)
        n_cls = len(self.classes_)
        M = np.zeros((n_seq, n_cls), dtype=np.float64)
        strands: list[list[str]] = [["+"] * n_cls for _ in range(n_seq)]
        check_rc = self.strand_aware and self.alphabet.is_complementary

        for i, seq in enumerate(sequences):
            ids = self.alphabet.encode(seq)
            rc_ids = (
                self.alphabet.reverse_complement_ids(ids)
                if (check_rc and len(ids) > 0)
                else None
            )
            for j, lab in enumerate(self.classes_):
                model = self.models[lab]
                ce_fwd = model.score(ids).mean_bits
                if rc_ids is not None:
                    ce_rev = model.score(rc_ids).mean_bits
                    if np.isfinite(ce_rev) and (not np.isfinite(ce_fwd) or ce_rev < ce_fwd):
                        M[i, j] = ce_rev
                        strands[i][j] = "-"
                        continue
                M[i, j] = ce_fwd
                strands[i][j] = "+"
        return M, strands

    def score_matrix(self, sequences) -> np.ndarray:
        """``(n_sequences, n_classes)`` matrix of cross-entropies (lower=better)."""
        M, _ = self.score_matrix_with_strands(sequences)
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
        return self._posteriors_from_scores(M)

    @staticmethod
    def _posteriors_from_scores(M: np.ndarray) -> np.ndarray:
        logits = np.where(np.isfinite(M), -M, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        denom = p.sum(axis=1, keepdims=True)
        return np.where(denom > 0, p / denom, 1.0 / max(1, M.shape[1]))

    def classify_records(self, records) -> list[ClassificationCall]:
        """Classify FASTA/FASTQ records and return rich :class:`ClassificationCall` objects."""
        ids_list: list[str] = []
        seqs_list: list[str] = []
        for idx, rec in enumerate(records):
            if isinstance(rec, Record):
                ids_list.append(rec.id)
                seqs_list.append(rec.seq)
            elif isinstance(rec, tuple) and len(rec) >= 2:
                ids_list.append(str(rec[0]))
                seqs_list.append(str(rec[1]))
            elif isinstance(rec, str):
                ids_list.append(f"seq_{idx + 1}")
                seqs_list.append(rec)
            else:
                raise TypeError(f"Unsupported record type: {type(rec)!r}")

        if not seqs_list:
            return []

        M, strands = self.score_matrix_with_strands(seqs_list)
        probs = self._posteriors_from_scores(M)
        n_cls = len(self.classes_)
        calls: list[ClassificationCall] = []

        for i, rid in enumerate(ids_list):
            row = M[i]
            order = np.argsort(row)
            best_j = int(order[0])
            best_cls = str(self.classes_[best_j])
            best_bits = float(row[best_j])
            best_strand = strands[i][best_j]
            post = float(probs[i, best_j])
            if n_cls >= 2:
                sec_j = int(order[1])
                runner_up = str(self.classes_[sec_j])
                delta = float(row[sec_j] - best_bits)
            else:
                runner_up = None
                delta = 0.0
            calls.append(
                ClassificationCall(
                    id=rid,
                    predicted_class=best_cls,
                    strand=best_strand,
                    bits_per_base=best_bits,
                    posterior=post,
                    runner_up=runner_up,
                    delta_bits=delta,
                )
            )
        return calls

    def save(self, path: str) -> str:
        """Save multi-class classifier to a single zero-pickle ``.npz`` archive."""
        if not self.classes_:
            raise RuntimeError("Cannot save an unfitted FPCClassifier")
        str_classes = [str(c) for c in self.classes_]
        meta = {
            "format_version": MODEL_FORMAT_VERSION,
            "model_type": "FPCClassifier",
            "alphabet": {
                "name": self.alphabet.name,
                "symbols": self.alphabet.symbols,
                "wildcard": self.alphabet.wildcard,
                "complements": self.alphabet.complements,
            },
            "order": self.order,
            "epochs": self.epochs,
            "strand_aware": self.strand_aware,
            "double_strand_train": self.double_strand_train,
            "cortex_kwargs": self.cortex_kwargs,
            "classes": str_classes,
        }
        archive_arrays: dict[str, np.ndarray] = {
            "meta_json": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        }
        for idx, lab in enumerate(self.classes_):
            c_state = self.models[lab].to_state_dict()
            for key, arr in c_state.items():
                archive_arrays[f"c{idx}_{key}"] = arr
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **archive_arrays)
        return path

    @classmethod
    def load(cls, path: str, use_fast: bool = True) -> "FPCClassifier":
        """Load a saved :class:`FPCClassifier` with ``allow_pickle=False``."""
        with np.load(path, allow_pickle=False) as archive:
            raw = {k: archive[k] for k in archive.files}
        meta = json.loads(bytes(np.asarray(raw["meta_json"], dtype=np.uint8)).decode("utf-8"))
        if meta.get("model_type") != "FPCClassifier":
            raise ValueError(f"Expected FPCClassifier archive, got {meta.get('model_type')!r}")
        a_cfg = meta["alphabet"]
        alphabet = Alphabet(
            name=a_cfg["name"],
            symbols=a_cfg["symbols"],
            wildcard=a_cfg["wildcard"],
            complements=a_cfg.get("complements"),
        )
        kw = dict(meta.get("cortex_kwargs", {}))
        kw["use_fast"] = use_fast
        clf = cls(
            alphabet=alphabet,
            order=int(meta["order"]),
            epochs=int(meta["epochs"]),
            strand_aware=bool(meta.get("strand_aware", alphabet.is_complementary)),
            double_strand_train=bool(meta.get("double_strand_train", False)),
            **kw,
        )
        clf.classes_ = list(meta["classes"])
        clf.models = {}
        for idx, lab in enumerate(clf.classes_):
            prefix = f"c{idx}_"
            sub = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
            clf.models[lab] = FractalPredictiveCortex.from_state_dict(sub, use_fast=use_fast)
        return clf


def classification_accuracy(pred, true) -> float:
    pred = list(pred); true = list(true)
    return float(np.mean([p == t for p, t in zip(pred, true)]))


# ===================================================================== #
#  4. DISCOVER (unsupervised structure)                                 #
# ===================================================================== #
def distance_matrix(
    sequences,
    alphabet: Alphabet = DNA,
    order: int = 6,
    epochs: int = 1,
    strand_invariant: bool = False,
    **cortex_kwargs,
) -> np.ndarray:
    """Symmetric, alignment-free cross-entropy distance between sequences.

    For each sequence we train its own cortex, then measure the *excess* bits
    each model spends on every other sequence relative to that sequence's own
    model::

        d(A, B) = 1/2 [ (CE_A(B) - CE_B(B)) + (CE_B(A) - CE_A(A)) ]

    When ``strand_invariant=True`` and the alphabet is complementary (DNA/RNA),
    each cortex is trained on the canonicalised ``{seq, revcomp(seq)}`` pair
    and evaluated symmetrically over both strands so ``D(A, B) == D(A, rc(B))``
    identically.
    """
    n = len(sequences)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    use_sym_strand = bool(strand_invariant and alphabet.is_complementary)
    fwd_ids = [alphabet.encode(s) for s in sequences]
    rc_ids = (
        [alphabet.reverse_complement_ids(x) for x in fwd_ids]
        if use_sym_strand
        else None
    )

    cortices = []
    for idx, x in enumerate(fwd_ids):
        c = FractalPredictiveCortex(alphabet, order=order, **cortex_kwargs)
        if use_sym_strand and rc_ids is not None:
            xr = rc_ids[idx]
            # Canonicalise strand training order so fit(B) == fit(rc(B)) bit-for-bit
            x_bytes = x.tobytes()
            xr_bytes = xr.tobytes()
            first, second = (x, xr) if x_bytes <= xr_bytes else (xr, x)
            for _ in range(max(1, epochs)):
                c.fit(first, epochs=1)
                c.fit(second, epochs=1)
        else:
            c.fit(x, epochs=epochs)
        cortices.append(c)

    # ce[i][j] = cross-entropy of sequence j under model i.
    ce = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            c_fwd = cortices[i].cross_entropy(fwd_ids[j])
            if use_sym_strand and rc_ids is not None:
                c_rev = cortices[i].cross_entropy(rc_ids[j])
                ce[i, j] = 0.5 * (c_fwd + c_rev)
            else:
                ce[i, j] = c_fwd

    D = np.zeros((n, n), dtype=np.float64)
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
