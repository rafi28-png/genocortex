"""
The Fractal Predictive Cortex (FPC) -- the brain.
=================================================

A self-organising, online, generative model of a symbolic sequence that learns
with **no training labels, no alignment, and no pretrained weights**. It reads
a sequence once, left to right, and does four things at the same time, all from
one mechanism:

    predict   -- give a probability distribution over the next symbol,
    detect    -- expose mutations / anomalies as spikes of surprise,
    classify  -- score how well it explains *another* sequence (cross-entropy),
    discover  -- expose structure through the geometry of its prototypes.

How it works (the whole idea in one paragraph)
----------------------------------------------
The sequence is streamed through the Chaos Game embedding, so at every step the
"context" (everything read so far, exponentially discounted) is a single point
in the plane. The cortex keeps a growing cloud of **prototypes** -- little
memory cells, each sitting at a location in that plane and holding a tally of
"what symbol tended to come next when the context landed near me". To predict,
it blends the nearest prototypes (Gaussian-weighted) into a distribution over
the next symbol. It then sees the truth, pays a *surprise* cost of
``-log2 P(truth)``, reinforces the winning prototype, and -- crucially -- if it
was in unfamiliar territory (the nearest prototype was too far, or the surprise
was abnormally high) it **grows a brand-new prototype right there**. So the
model literally decides *when and where to allocate memory*, guided by its own
surprise. Its sole objective, cumulative surprise, equals the code length of
the sequence: minimising it is exactly the Minimum Description Length principle.

The spatial resolution at which prototypes are allowed to split, ``r ** order``,
is the one knob that sets the model's effective memory depth ("Markov order"):
because CGR distance halves with each additional shared suffix symbol, a
distance threshold of ``r ** order`` corresponds to resolving ``order`` symbols
of context.

Pure NumPy/SciPy (with an optional PEP 3118 C spatial-grid accelerator
``fpc._fast`` for megabase-scale chromosomes). No API. No internet.
"""

from __future__ import annotations

import json
import numpy as np

from .alphabet import Alphabet
from .cgr import ChaosGameEmbedding
from .surprise import SurpriseTrack

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:  # pragma: no cover
    _HAVE_KDTREE = False

try:
    from . import _fast
    HAS_FAST_KERNEL = True
except Exception:  # pragma: no cover - optional compiled accelerator
    _fast = None
    HAS_FAST_KERNEL = False

MODEL_FORMAT_VERSION = 1


class FractalPredictiveCortex:
    """A growing, self-organising predictive model over the fractal mind-space.

    Parameters
    ----------
    alphabet:
        The symbol alphabet (DNA / RNA / PROTEIN / custom).
    order:
        Target effective memory depth (in symbols). Sets the prototype split
        resolution ``dist_threshold = r ** order``. Larger = longer context,
        more prototypes, sharper predictions, more memory.
    contraction:
        Optional CGR contraction override (see :class:`ChaosGameEmbedding`).
    n_neighbors:
        How many nearest prototypes to blend when predicting.
    alpha:
        Laplace (add-alpha) smoothing on prototype counts. Guarantees every
        symbol keeps non-zero probability, so surprise is always finite.
    background_weight:
        Mixture weight given to the global symbol frequencies. A small safety
        net that keeps predictions sane in unseen regions.
    learn_rate:
        Step size for nudging a winning prototype toward the current context
        (self-organising-map style). ``0`` freezes prototype positions.
    surprise_growth:
        If True, also spawn prototypes where surprise is abnormally high (not
        only where space is empty) -- "grow where you are shocked".
    max_prototypes:
        Hard cap on memory. When exceeded, least-used prototypes are pruned.
    prune_interval:
        How often (in symbols) to run usage decay + pruning.
    usage_decay:
        Per-step multiplicative decay of prototype usage, so stale cells fade.
    seed:
        RNG seed for reproducible sampling / tie-breaks.
    use_fast:
        Use the compiled C spatial-grid kernel ``fpc._fast`` when available
        (default True). Set ``False`` to force the reference NumPy loop.
    """

    def __init__(
        self,
        alphabet: Alphabet,
        order: int = 8,
        contraction: float | None = None,
        n_neighbors: int = 8,
        alpha: float = 0.5,
        background_weight: float = 0.02,
        learn_rate: float = 0.02,
        surprise_growth: bool = True,
        max_prototypes: int = 6000,
        prune_interval: int = 4000,
        usage_decay: float = 0.9997,
        seed: int = 0,
        use_fast: bool = True,
    ):
        self.alphabet = alphabet
        self.use_fast = bool(use_fast)
        self.embedding = ChaosGameEmbedding(alphabet, contraction, use_fast=self.use_fast)
        self.r = self.embedding.r
        self.a = alphabet.size
        self.order = int(order)

        self.dist_threshold = float(self.r ** self.order)
        self.sigma = max(self.dist_threshold, 1e-9)
        self.n_neighbors = int(n_neighbors)
        self.alpha = float(alpha)
        self.background_weight = float(background_weight)
        self.learn_rate = float(learn_rate)
        self.surprise_growth = bool(surprise_growth)
        if int(max_prototypes) < 1:
            raise ValueError(f"max_prototypes must be >= 1, got {max_prototypes}")
        self.max_prototypes = int(max_prototypes)
        self.prune_interval = int(prune_interval)
        self.usage_decay = float(usage_decay)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        # -- prototype store (dynamic, capacity-doubling arrays) ---------
        self._cap = 0
        self._m = 0
        self.X = np.empty((0, 2), dtype=np.float64)        # positions
        self.N = np.empty((0, self.a), dtype=np.float64)   # next-symbol counts
        self.usage = np.empty((0,), dtype=np.float64)      # decayed hit counts

        # Global background (Laplace-primed) symbol counts.
        self.global_counts = np.full(self.a, self.alpha, dtype=np.float64)

        # Running surprise statistics (Welford) for the "surprise growth" rule.
        self._steps = 0
        self._surp_n = 0
        self._surp_mean = 0.0
        self._surp_m2 = 0.0

    # ------------------------------------------------------------------ #
    #  prototype storage                                                 #
    # ------------------------------------------------------------------ #
    def _ensure_capacity(self, need: int) -> None:
        if need <= self._cap:
            return
        new_cap = max(64, self._cap * 2, need)
        X = np.zeros((new_cap, 2), dtype=np.float64)
        N = np.zeros((new_cap, self.a), dtype=np.float64)
        U = np.zeros((new_cap,), dtype=np.float64)
        X[: self._m] = self.X[: self._m]
        N[: self._m] = self.N[: self._m]
        U[: self._m] = self.usage[: self._m]
        self.X, self.N, self.usage = X, N, U
        self._cap = new_cap

    def _add_prototype(self, x: np.ndarray, symbol: int | None) -> int:
        self._ensure_capacity(self._m + 1)
        idx = self._m
        self.X[idx] = x
        self.N[idx] = 0.0
        if symbol is not None:
            self.N[idx, symbol] += 1.0
        self.usage[idx] = 1.0
        self._m += 1
        return idx

    @property
    def n_prototypes(self) -> int:
        return self._m

    def _prune(self) -> None:
        """Keep the ``max_prototypes`` most-used cells; drop the rest."""
        if self._m <= self.max_prototypes:
            return
        keep = np.argsort(self.usage[: self._m], kind="mergesort")[::-1][: self.max_prototypes]
        keep.sort()
        k = len(keep)
        self.X[:k] = self.X[keep]
        self.N[:k] = self.N[keep]
        self.usage[:k] = self.usage[keep]
        self._m = k

    # ------------------------------------------------------------------ #
    #  prediction                                                        #
    # ------------------------------------------------------------------ #
    def _background(self) -> np.ndarray:
        return self.global_counts / self.global_counts.sum()

    def predict_dist(self, context_point: np.ndarray):
        """Predict the next-symbol distribution at a context point.

        Returns ``(p, nearest_idx, nearest_dist)`` where ``p`` is a length-``a``
        probability vector over the canonical symbols.
        """
        bg = self._background()
        if self._m == 0:
            return bg.copy(), -1, np.inf

        c = np.asarray(context_point, dtype=np.float64)
        diff = self.X[: self._m] - c
        d2 = np.einsum("ij,ij->i", diff, diff)          # squared distances

        k = min(self.n_neighbors, self._m)
        if k < self._m:
            nn = np.argpartition(d2, k - 1)[:k]
        else:
            nn = np.arange(self._m)
        nearest_local = nn[np.argmin(d2[nn])]
        nearest_dist = float(np.sqrt(d2[nearest_local]))

        w = np.exp(-d2[nn] / (2.0 * self.sigma * self.sigma))
        wsum = w.sum()
        if wsum <= 0:                                    # numerical underflow
            return bg.copy(), int(nearest_local), nearest_dist

        counts = self.N[nn]                              # (k, a)
        weighted = (w[:, None] * counts).sum(axis=0)     # (a,)
        denom = weighted.sum()
        p_proto = (weighted + self.alpha) / (denom + self.a * self.alpha)

        lam = self.background_weight
        p = (1.0 - lam) * p_proto + lam * bg
        p /= p.sum()
        return p, int(nearest_local), nearest_dist

    def predict_next(self, prefix_ids: np.ndarray) -> np.ndarray:
        """Distribution over the symbol that follows ``prefix_ids``."""
        prefix_ids = np.asarray(prefix_ids)
        if len(prefix_ids) == 0:
            c = self.embedding.centre
        else:
            c = self.embedding.embed(prefix_ids, use_fast=self.use_fast)[-1]
        p, _, _ = self.predict_dist(c)
        return p

    # ------------------------------------------------------------------ #
    #  learning                                                          #
    # ------------------------------------------------------------------ #
    def _update_surprise_stats(self, bits: float) -> None:
        self._surp_n += 1
        d = bits - self._surp_mean
        self._surp_mean += d / self._surp_n
        self._surp_m2 += d * (bits - self._surp_mean)

    def _surprise_is_high(self, bits: float) -> bool:
        if self._surp_n < 50:
            return False
        var = self._surp_m2 / max(1, self._surp_n - 1)
        std = np.sqrt(var)
        return bits > self._surp_mean + 2.0 * std

    def fit(self, ids: np.ndarray, epochs: int = 1, update: bool = True,
            double_strand: bool = False) -> SurpriseTrack:
        """Stream a sequence through the cortex, learning as it goes.

        Parameters
        ----------
        ids:
            Encoded symbol-id array.
        epochs:
            Number of passes over the sequence.
        update:
            If False, scores without updating prototypes.
        double_strand:
            If True and the alphabet is complementary (DNA/RNA), also trains on
            the reverse-complement strand during each epoch so the model learns
            both chromosomal strands symmetrically.

        Returns the :class:`SurpriseTrack` from the **final** forward pass.
        """
        ids = np.asarray(ids, dtype=np.int64)
        rc_ids = (
            self.alphabet.reverse_complement_ids(ids).astype(np.int64)
            if (double_strand and update and self.alphabet.is_complementary and len(ids) > 0)
            else None
        )
        track = None
        for _ in range(max(1, epochs)):
            track = self._stream(ids, update=update)
            if rc_ids is not None:
                self._stream(rc_ids, update=True)
        return track

    def score(self, ids: np.ndarray) -> SurpriseTrack:
        """Score a sequence **without learning** -- a frozen forward pass.

        This is what classification and clean anomaly detection use: how many
        bits does *this already-trained brain* spend to explain the sequence?
        """
        ids_c = np.ascontiguousarray(ids, dtype=np.int64)
        if len(ids_c) == 0:
            return SurpriseTrack(np.empty(0, dtype=np.float64))

        if self.use_fast and HAS_FAST_KERNEL and self._m > 0:
            C = np.ascontiguousarray(
                self.embedding.context_points(ids_c, use_fast=True), dtype=np.float64
            )
            bits = np.full(len(ids_c), np.nan, dtype=np.float64)
            _fast.cortex_score_batch(
                ids_c,
                C,
                np.ascontiguousarray(self.X[: self._m], dtype=np.float64),
                np.ascontiguousarray(self.N[: self._m], dtype=np.float64),
                np.ascontiguousarray(self.global_counts, dtype=np.float64),
                bits,
                int(self.a),
                int(self._m),
                int(self.n_neighbors),
                float(self.sigma),
                float(self.alpha),
                float(self.background_weight),
            )
            return SurpriseTrack(bits)

        if _HAVE_KDTREE and self._m > 0:
            return self._score_fast(ids_c)
        return self._stream(ids_c, update=False)

    def _score_fast(self, ids: np.ndarray) -> SurpriseTrack:
        a = self.a
        C = self.embedding.context_points(ids, use_fast=self.use_fast)
        tree = cKDTree(self.X[: self._m])
        k = min(self.n_neighbors, self._m)
        try:
            dist, idx = tree.query(C, k=k, workers=-1)
        except TypeError:  # pragma: no cover - older SciPy fallback
            dist, idx = tree.query(C, k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = np.exp(-(dist ** 2) / (2.0 * self.sigma * self.sigma))   # (T, k)
        wsum = w.sum(axis=1, keepdims=True)              # (T, 1)
        counts = self.N[idx]                             # (T, k, a)
        weighted = np.einsum("tk,tka->ta", w, counts)    # (T, a)
        denom = weighted.sum(axis=1, keepdims=True)
        p_proto = (weighted + self.alpha) / (denom + a * self.alpha)
        bg = self._background()
        lam = self.background_weight
        P = (1.0 - lam) * p_proto + lam * bg[None, :]
        P /= P.sum(axis=1, keepdims=True)
        P = np.where(wsum > 0.0, P, bg[None, :])

        bits = np.full(len(ids), np.nan, dtype=np.float64)
        canonical = (ids >= 0) & (ids < a)
        tvals = np.arange(len(ids))[canonical]
        probs = P[tvals, ids[canonical]]
        probs = np.clip(probs, 1e-18, 1.0)
        bits[canonical] = -np.log2(probs)
        return SurpriseTrack(bits)

    def _growth_cap(self) -> int:
        if self.prune_interval > 0:
            return self.max_prototypes + max(16, self.max_prototypes // 4)
        return self.max_prototypes

    def _stream(self, ids: np.ndarray, update: bool) -> SurpriseTrack:
        ids_c = np.ascontiguousarray(ids, dtype=np.int64)
        if len(ids_c) == 0:
            return SurpriseTrack(np.empty(0, dtype=np.float64))

        if self.use_fast and HAS_FAST_KERNEL:
            need_cap = max(64, self._growth_cap() + 1, self._m + 1)
            self._ensure_capacity(need_cap)
            C = np.ascontiguousarray(
                self.embedding.context_points(ids_c, use_fast=True), dtype=np.float64
            )
            bits = np.full(len(ids_c), np.nan, dtype=np.float64)
            m_new, steps_new, surp_n_new, surp_mean_new, surp_m2_new = _fast.cortex_stream(
                ids_c,
                C,
                self.X,
                self.N,
                self.usage,
                self.global_counts,
                bits,
                int(self.a),
                int(self._m),
                int(self._cap),
                int(self.n_neighbors),
                float(self.dist_threshold),
                float(self.sigma),
                float(self.alpha),
                float(self.background_weight),
                float(self.learn_rate),
                bool(self.surprise_growth),
                int(self.max_prototypes),
                int(self.prune_interval),
                float(self.usage_decay),
                int(self._steps),
                int(self._surp_n),
                float(self._surp_mean),
                float(self._surp_m2),
                int(1 if update else 0),
            )
            self._m = int(m_new)
            self._steps = int(steps_new)
            self._surp_n = int(surp_n_new)
            self._surp_mean = float(surp_mean_new)
            self._surp_m2 = float(surp_m2_new)
            return SurpriseTrack(bits)

        a = self.a
        C = self.embedding.context_points(ids_c, use_fast=False)
        bits = np.full(len(ids_c), np.nan, dtype=np.float64)

        for t in range(len(ids_c)):
            s = int(ids_c[t])
            c = C[t]
            p, nn, dmin = self.predict_dist(c)

            if 0 <= s < a:                                # canonical symbol
                ps = p[s]
                b = -np.log2(ps) if ps > 0 else 60.0
                bits[t] = b
                if update:
                    self._update_surprise_stats(b)
                    self.global_counts[s] += 1.0
                    self._learn(c, s, nn, dmin, b)
            # wildcard symbols advance the trajectory but are not scored/learned.

            if update:
                self._steps += 1
                if self.prune_interval > 0 and self._steps % self.prune_interval == 0:
                    self.usage[: self._m] *= (self.usage_decay ** self.prune_interval)
                    self._prune()

        if update and self._m > self.max_prototypes:
            self._prune()

        return SurpriseTrack(bits)

    def _learn(self, c: np.ndarray, symbol: int, nearest: int,
               nearest_dist: float, bits: float) -> None:
        """Reinforce, adapt, and (maybe) grow a prototype."""
        empty = self._m == 0
        far = nearest_dist > self.dist_threshold
        shocked = self.surprise_growth and self._surprise_is_high(bits) \
            and nearest_dist > 0.5 * self.dist_threshold
        can_grow = self._m < self._growth_cap()

        if empty or ((far or shocked) and can_grow):
            self._add_prototype(c, symbol)
            return

        # Otherwise reinforce and gently move the winning prototype.
        self.N[nearest, symbol] += 1.0
        self.usage[nearest] += 1.0
        if self.learn_rate > 0.0:
            self.X[nearest] += self.learn_rate * (c - self.X[nearest])

    # ------------------------------------------------------------------ #
    #  read-outs used by the four abilities                              #
    # ------------------------------------------------------------------ #
    def cross_entropy(self, ids: np.ndarray, strand_aware: bool = False) -> float:
        """Average surprise (bits/symbol) this brain spends on ``ids``.

        If ``strand_aware=True`` and the alphabet is complementary (DNA/RNA),
        returns ``min(CE(ids), CE(rc(ids)))``.
        """
        ce_fwd = self.score(ids).mean_bits
        if strand_aware and self.alphabet.is_complementary and len(ids) > 0:
            ce_rev = self.score(self.alphabet.reverse_complement_ids(ids)).mean_bits
            if np.isfinite(ce_rev) and (not np.isfinite(ce_fwd) or ce_rev < ce_fwd):
                return ce_rev
        return ce_fwd

    def cross_entropy_with_strand(self, ids: np.ndarray) -> tuple[float, str]:
        """Return ``(best_cross_entropy, strand)`` where strand is ``'+'`` or ``'-'``."""
        ce_fwd = self.score(ids).mean_bits
        if not self.alphabet.is_complementary or len(ids) == 0:
            return ce_fwd, "+"
        ce_rev = self.score(self.alphabet.reverse_complement_ids(ids)).mean_bits
        if np.isfinite(ce_rev) and (not np.isfinite(ce_fwd) or ce_rev < ce_fwd):
            return ce_rev, "-"
        return ce_fwd, "+"

    def generate(self, prefix_ids: np.ndarray, length: int,
                 sample: bool = True, temperature: float = 1.0) -> np.ndarray:
        """Continue a sequence for ``length`` symbols using the learned model.

        ``sample=False`` takes the arg-max (most likely) symbol each step;
        otherwise it samples with the given ``temperature``.
        """
        out = list(np.asarray(prefix_ids, dtype=np.int64))
        for _ in range(length):
            p = self.predict_next(np.array(out, dtype=np.int64))
            if sample:
                if temperature != 1.0:
                    logp = np.log(np.clip(p, 1e-12, 1.0)) / max(1e-6, temperature)
                    p = np.exp(logp - logp.max())
                    p /= p.sum()
                nxt = int(self._rng.choice(self.a, p=p))
            else:
                nxt = int(np.argmax(p))
            out.append(nxt)
        return np.array(out, dtype=np.int64)

    def prototype_frame(self):
        """Snapshot of the learned prototypes for visualisation / analysis.

        Returns a dict of arrays: ``x``, ``y`` (positions), ``usage``,
        ``dominant`` (id of the most-predicted next symbol), ``entropy`` (bits
        of the local next-symbol distribution -- low = decisive prototype).
        """
        m = self._m
        X = self.X[:m]
        N = self.N[:m] + self.alpha
        P = N / N.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(P * np.log2(P), axis=1)
        return {
            "x": X[:, 0].copy(),
            "y": X[:, 1].copy(),
            "usage": self.usage[:m].copy(),
            "dominant": np.argmax(N, axis=1) if m > 0 else np.empty(0, dtype=np.int64),
            "entropy": ent,
        }

    # ------------------------------------------------------------------ #
    #  safe zero-pickle model persistence                                #
    # ------------------------------------------------------------------ #
    def to_state_dict(self) -> dict:
        """Export model arrays and JSON metadata dictionary for safe persistence."""
        meta = {
            "format_version": MODEL_FORMAT_VERSION,
            "model_type": "FractalPredictiveCortex",
            "alphabet": {
                "name": self.alphabet.name,
                "symbols": self.alphabet.symbols,
                "wildcard": self.alphabet.wildcard,
                "complements": self.alphabet.complements,
            },
            "params": {
                "order": self.order,
                "contraction": self.r,
                "n_neighbors": self.n_neighbors,
                "alpha": self.alpha,
                "background_weight": self.background_weight,
                "learn_rate": self.learn_rate,
                "surprise_growth": self.surprise_growth,
                "max_prototypes": self.max_prototypes,
                "prune_interval": self.prune_interval,
                "usage_decay": self.usage_decay,
                "seed": self.seed,
            },
            "stats": {
                "m": self._m,
                "steps": self._steps,
                "surp_n": self._surp_n,
                "surp_mean": self._surp_mean,
                "surp_m2": self._surp_m2,
            },
        }
        return {
            "meta_json": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            "X": self.X[: self._m].copy(),
            "N": self.N[: self._m].copy(),
            "usage": self.usage[: self._m].copy(),
            "global_counts": self.global_counts.copy(),
        }

    @classmethod
    def from_state_dict(cls, arrays: dict, use_fast: bool = True) -> "FractalPredictiveCortex":
        """Reconstruct a :class:`FractalPredictiveCortex` from :meth:`to_state_dict`."""
        meta = json.loads(bytes(np.asarray(arrays["meta_json"], dtype=np.uint8)).decode("utf-8"))
        if meta.get("model_type") != "FractalPredictiveCortex":
            raise ValueError(f"Expected FractalPredictiveCortex archive, got {meta.get('model_type')!r}")
        a_cfg = meta["alphabet"]
        alphabet = Alphabet(
            name=a_cfg["name"],
            symbols=a_cfg["symbols"],
            wildcard=a_cfg["wildcard"],
            complements=a_cfg.get("complements"),
        )
        p = meta["params"]
        cortex = cls(
            alphabet=alphabet,
            order=p["order"],
            contraction=p["contraction"],
            n_neighbors=p["n_neighbors"],
            alpha=p["alpha"],
            background_weight=p["background_weight"],
            learn_rate=p["learn_rate"],
            surprise_growth=p["surprise_growth"],
            max_prototypes=p["max_prototypes"],
            prune_interval=p["prune_interval"],
            usage_decay=p["usage_decay"],
            seed=p.get("seed", 0),
            use_fast=use_fast,
        )
        X = np.asarray(arrays["X"], dtype=np.float64)
        N = np.asarray(arrays["N"], dtype=np.float64)
        usage = np.asarray(arrays["usage"], dtype=np.float64)
        m = len(X)
        cortex._ensure_capacity(max(64, m))
        cortex._m = m
        if m > 0:
            cortex.X[:m] = X
            cortex.N[:m] = N
            cortex.usage[:m] = usage
        cortex.global_counts = np.asarray(arrays["global_counts"], dtype=np.float64).copy()
        st = meta["stats"]
        cortex._steps = int(st["steps"])
        cortex._surp_n = int(st["surp_n"])
        cortex._surp_mean = float(st["surp_mean"])
        cortex._surp_m2 = float(st["surp_m2"])
        return cortex

    def save(self, path: str) -> str:
        """Save trained cortex to a compressed, zero-pickle ``.npz`` archive."""
        state = self.to_state_dict()
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **state)
        return path

    @classmethod
    def load(cls, path: str, use_fast: bool = True) -> "FractalPredictiveCortex":
        """Load a cortex saved with :meth:`save` (``allow_pickle=False``)."""
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        return cls.from_state_dict(arrays, use_fast=use_fast)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        backend = "C" if (self.use_fast and HAS_FAST_KERNEL) else "NumPy"
        return (f"FractalPredictiveCortex(alphabet={self.alphabet.name}, "
                f"order={self.order}, prototypes={self._m}, "
                f"dist_threshold={self.dist_threshold:.2e}, backend={backend})")
