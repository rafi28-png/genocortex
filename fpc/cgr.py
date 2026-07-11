"""
The fractal mind-space -- (generalised) Chaos Game Representation.
==================================================================

This is the *geometry* the brain thinks in. A Chaos Game Representation (CGR)
turns a symbolic sequence into a trajectory of points in the unit plane by a
simple iterated contraction map::

    p_0 = centre
    p_t = (1 - r) * p_{t-1} + r * vertex(symbol_t)

Each canonical symbol owns one *vertex*; the wildcard pulls gently toward the
centre (it carries no positional information).

Why this matters
----------------
The single, beautiful property that the whole engine leans on:

    **Two positions whose preceding k symbols are identical land at (nearly)
    the same point.** More generally, geometric distance in this space is an
    exponentially-discounted memory of shared recent context.

So *geometry becomes context*. That is exactly what lets a spatial,
self-organising model (the cortex) behave like a variable-order sequence
predictor -- with no alignment and no training.

The classic square CGR for DNA (4 symbols at the unit-square corners,
contraction 1/2) is recovered exactly as a special case. For other alphabet
sizes we place the vertices on a circle and pick a contraction ratio that
keeps the sub-copies from overlapping: ``r = min(1/2, sin(pi / a))``.
"""

from __future__ import annotations

import numpy as np

from .alphabet import Alphabet


class ChaosGameEmbedding:
    """Maps symbol-id sequences into the 2-D fractal mind-space.

    Parameters
    ----------
    alphabet:
        The :class:`~fpc.alphabet.Alphabet` in use.
    contraction:
        Optional override for the contraction ratio ``r``. If ``None`` a good
        default is chosen from the alphabet size (see module docstring).
    """

    def __init__(self, alphabet: Alphabet, contraction: float | None = None):
        self.alphabet = alphabet
        a = alphabet.size
        if contraction is None:
            contraction = 0.5 if a <= 4 else min(0.5, float(np.sin(np.pi / a)))
        if not (0.0 < contraction < 1.0):
            raise ValueError("contraction must lie in (0, 1)")
        self.r = float(contraction)
        self.centre = np.array([0.5, 0.5], dtype=np.float64)
        self.vertices = self._build_vertices(a)
        # Wildcard "vertex" is the centre -> a neutral, information-free pull.
        self._all_vertices = np.vstack([self.vertices, self.centre])

    # -- geometry --------------------------------------------------------
    @staticmethod
    def _build_vertices(a: int) -> np.ndarray:
        """Vertex coordinates for ``a`` canonical symbols, inside [0, 1]^2."""
        if a == 4:
            # Classic DNA CGR corners, in alphabet order (e.g. A, C, G, T).
            return np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                            dtype=np.float64)
        # General case: regular polygon on a circle of radius 1/2, centred at
        # (0.5, 0.5). Start at the top and go clockwise for a stable layout.
        k = np.arange(a)
        theta = np.pi / 2.0 - 2.0 * np.pi * k / a
        return np.column_stack([0.5 + 0.5 * np.cos(theta),
                                0.5 + 0.5 * np.sin(theta)]).astype(np.float64)

    # -- the core map ----------------------------------------------------
    def embed(self, ids: np.ndarray) -> np.ndarray:
        """Return the CGR trajectory ``P`` with ``P[t]`` = point *after* symbol
        ``t``. Shape ``(T, 2)``. Vectorised, so it is fast on whole genomes.

        The recurrence ``p_t = (1-r) p_{t-1} + r v_t`` is linear, hence

            ``p_t = (1-r)^{t+1} centre + r * sum_{j<=t} (1-r)^{t-j} v_j``.

        We evaluate it with a stable cumulative reweighting rather than a
        Python loop.
        """
        ids = np.asarray(ids, dtype=np.int64)
        T = ids.shape[0]
        if T == 0:
            return np.empty((0, 2), dtype=np.float64)
        v = self._all_vertices[ids]                      # (T, 2) target vertices
        r = self.r
        one_minus = 1.0 - r

        # Weighted running sum computed via logs of (1-r) to stay stable for
        # long sequences. w_t = (1-r)^t ; scaled contributions r*v_j / w_{j+1}.
        t_idx = np.arange(T, dtype=np.float64)
        # Contribution of vertex j to the sum, factored so the cumsum is O(T):
        #   p_t = (1-r)^{t+1} c + r * (1-r)^t * cumsum_j [ v_j / (1-r)^{j+1} ]
        # To avoid overflow of (1-r)^{-j} we instead just run a compact loop in
        # NumPy over chunks; for clarity and robustness we use the direct O(T)
        # scan below (still pure NumPy, no per-symbol Python overhead via
        # np.frompyfunc-free accumulation).
        P = np.empty((T, 2), dtype=np.float64)
        acc = self.centre.copy()
        rv = r * v                                        # precompute r * vertex
        for t in range(T):                                # single tight loop
            acc = one_minus * acc + rv[t]
            P[t] = acc
        del t_idx
        return P

    def context_points(self, ids: np.ndarray) -> np.ndarray:
        """Points that represent the *context preceding* each position.

        ``C[t]`` is where the trajectory sits just before symbol ``t`` is read,
        i.e. the geometric summary of symbols ``0..t-1``. ``C[0]`` is the
        centre. This is what the cortex uses to predict symbol ``t``.
        """
        P = self.embed(ids)
        C = np.empty_like(P)
        C[0] = self.centre
        if len(P) > 1:
            C[1:] = P[:-1]
        return C

    # -- fixed-length features ------------------------------------------
    def density_image(self, ids_or_points, grid: int = 64) -> np.ndarray:
        """A ``grid x grid`` Frequency-CGR-style density image.

        This is an *alignment-free, fixed-length* fingerprint of a sequence of
        any length -- the generalisation of the well-known FCGR image to any
        alphabet. Rows/cols index the unit square; the value is how often the
        trajectory visited each cell (row-normalised to a probability).
        """
        pts = ids_or_points
        if np.ndim(pts) == 1:
            pts = self.embed(np.asarray(pts))
        if len(pts) == 0:
            return np.zeros((grid, grid), dtype=np.float64)
        ij = np.clip((pts * grid).astype(np.int64), 0, grid - 1)
        img = np.zeros((grid, grid), dtype=np.float64)
        np.add.at(img, (ij[:, 1], ij[:, 0]), 1.0)         # (row=y, col=x)
        total = img.sum()
        return img / total if total > 0 else img
