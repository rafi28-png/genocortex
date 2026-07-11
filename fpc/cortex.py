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

Pure NumPy/SciPy. No API. No internet.
"""

from __future__ import annotations

import numpy as np

from .alphabet import Alphabet
from .cgr import ChaosGameEmbedding
from .surprise import SurpriseTrack

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:  # pragma: no cover
    _HAVE_KDTREE = False


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
    ):
        self.alphabet = alphabet
        self.embedding = ChaosGameEmbedding(alphabet, contraction)
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
        self.max_prototypes = int(max_prototypes)
        self.prune_interval = int(prune_interval)
        self.usage_decay = float(usage_decay)
        self._rng = np.random.default_rng(seed)

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
        keep = np.argsort(self.usage[: self._m])[::-1][: self.max_prototypes]
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
            c = self.embedding.embed(prefix_ids)[-1]
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

    def fit(self, ids: np.ndarray, epochs: int = 1, update: bool = True) -> SurpriseTrack:
        """Stream a sequence through the cortex, learning as it goes.

        Returns the :class:`SurpriseTrack` from the **final** pass. Multiple
        epochs let the model consolidate (later passes have lower surprise).
        """
        ids = np.asarray(ids, dtype=np.int64)
        track = None
        for _ in range(max(1, epochs)):
            track = self._stream(ids, update=update)
        return track

    def score(self, ids: np.ndarray) -> SurpriseTrack:
        """Score a sequence **without learning** -- a frozen forward pass.

        This is what classification and clean anomaly detection use: how many
        bits does *this already-trained brain* spend to explain the sequence?

        Because the prototypes are frozen, we can vectorise the whole pass with a
        KD-tree batch query instead of the per-symbol Python loop used during
        learning -- typically 1-2 orders of magnitude faster, which is what makes
        the all-vs-all cross-entropy distance matrices (classification,
        phylogeny) practical on real genomes.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if _HAVE_KDTREE and self._m > 0 and len(ids) > 0:
            return self._score_fast(ids)
        return self._stream(ids, update=False)

    def _score_fast(self, ids: np.ndarray) -> SurpriseTrack:
        a = self.a
        C = self.embedding.context_points(ids)          # (T, 2)
        tree = cKDTree(self.X[: self._m])
        k = min(self.n_neighbors, self._m)
        dist, idx = tree.query(C, k=k)                   # (T,) or (T, k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = np.exp(-(dist ** 2) / (2.0 * self.sigma * self.sigma))   # (T, k)
        counts = self.N[idx]                             # (T, k, a)
        weighted = np.einsum("tk,tka->ta", w, counts)    # (T, a)
        denom = weighted.sum(axis=1, keepdims=True)
        p_proto = (weighted + self.alpha) / (denom + a * self.alpha)
        bg = self._background()
        lam = self.background_weight
        P = (1.0 - lam) * p_proto + lam * bg[None, :]
        P /= P.sum(axis=1, keepdims=True)

        bits = np.full(len(ids), np.nan, dtype=np.float64)
        canonical = ids < a
        tvals = np.arange(len(ids))[canonical]
        probs = P[tvals, ids[canonical]]
        probs = np.clip(probs, 1e-18, 1.0)
        bits[canonical] = -np.log2(probs)
        return SurpriseTrack(bits)

    def _stream(self, ids: np.ndarray, update: bool) -> SurpriseTrack:
        a = self.a
        C = self.embedding.context_points(ids)           # context before each pos
        bits = np.full(len(ids), np.nan, dtype=np.float64)

        for t in range(len(ids)):
            s = int(ids[t])
            c = C[t]
            p, nn, dmin = self.predict_dist(c)

            if s < a:                                     # canonical symbol
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
                if self._steps % self.prune_interval == 0:
                    self.usage[: self._m] *= (self.usage_decay ** self.prune_interval)
                    self._prune()

        return SurpriseTrack(bits)

    def _learn(self, c: np.ndarray, symbol: int, nearest: int,
               nearest_dist: float, bits: float) -> None:
        """Reinforce, adapt, and (maybe) grow a prototype."""
        empty = self._m == 0
        far = nearest_dist > self.dist_threshold
        shocked = self.surprise_growth and self._surprise_is_high(bits) \
            and nearest_dist > 0.5 * self.dist_threshold
        can_grow = self._m < self.max_prototypes

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
    def cross_entropy(self, ids: np.ndarray) -> float:
        """Average surprise (bits/symbol) this brain spends on ``ids``.

        The classification primitive: the class whose cortex has the *lowest*
        cross-entropy on a query is the maximum-likelihood / minimum-description
        -length label for that query.
        """
        return self.score(ids).mean_bits

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
            "dominant": np.argmax(N, axis=1),
            "entropy": ent,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"FractalPredictiveCortex(alphabet={self.alphabet.name}, "
                f"order={self.order}, prototypes={self._m}, "
                f"dist_threshold={self.dist_threshold:.2e})")
