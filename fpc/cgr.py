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

from .alphabet import Alphabet, DNA

try:
    from . import _fast
    _HAVE_FAST_CGR = True
except Exception:  # pragma: no cover - optional compiled accelerator
    _fast = None
    _HAVE_FAST_CGR = False


class ChaosGameEmbedding:
    """Maps symbol-id sequences into the 2-D fractal mind-space.

    Parameters
    ----------
    alphabet:
        The :class:`~fpc.alphabet.Alphabet` in use (default :data:`~fpc.alphabet.DNA`).
    contraction:
        Optional override for the contraction ratio ``r``. If ``None`` a good
        default is chosen from the alphabet size (see module docstring).
    use_fast:
        Use the compiled C kernel ``fpc._fast`` when available (default True).
    """

    def __init__(self, alphabet: Alphabet = DNA, contraction: float | None = None,
                 use_fast: bool = True):
        self.alphabet = alphabet
        a = alphabet.size
        if contraction is None:
            contraction = 0.5 if a <= 4 else min(0.5, float(np.sin(np.pi / a)))
        if not (0.0 < contraction < 1.0):
            raise ValueError("contraction must lie in (0, 1)")
        self.r = float(contraction)
        self.use_fast = bool(use_fast)
        self.centre = np.array([0.5, 0.5], dtype=np.float64)
        self.vertices = self._build_vertices(a)
        # Wildcard "vertex" is the centre -> a neutral, information-free pull.
        self._all_vertices = np.ascontiguousarray(
            np.vstack([self.vertices, self.centre]), dtype=np.float64
        )

    def _coerce_ids(self, ids: np.ndarray | str | bytes) -> np.ndarray:
        if isinstance(ids, (str, bytes)):
            s = ids if isinstance(ids, str) else ids.decode("ascii", "replace")
            return np.ascontiguousarray(self.alphabet.encode(s), dtype=np.int64)
        return np.ascontiguousarray(ids, dtype=np.int64)

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
    def embed(self, ids: np.ndarray | str | bytes, use_fast: bool | None = None) -> np.ndarray:
        """Return the CGR trajectory ``P`` with ``P[t]`` = point *after* symbol
        ``t``. Shape ``(T, 2)``. Accelerated in C when ``fpc._fast`` is built.

        The recurrence ``p_t = (1-r) p_{t-1} + r v_t`` is linear, hence

            ``p_t = (1-r)^{t+1} centre + r * sum_{j<=t} (1-r)^{t-j} v_j``.
        """
        ids_c = self._coerce_ids(ids)
        T = ids_c.shape[0]
        if T == 0:
            return np.empty((0, 2), dtype=np.float64)

        fast = self.use_fast if use_fast is None else bool(use_fast)
        P = np.empty((T, 2), dtype=np.float64)

        if fast and _HAVE_FAST_CGR:
            _fast.cgr_embed(
                ids_c,
                self._all_vertices,
                float(self.centre[0]),
                float(self.centre[1]),
                self.r,
                P,
            )
            return P

        n_verts = len(self._all_vertices)
        safe_ids = np.where((ids_c >= 0) & (ids_c < n_verts), ids_c, n_verts - 1)
        v = self._all_vertices[safe_ids]                  # (T, 2) target vertices
        r = self.r
        one_minus = 1.0 - r
        acc = self.centre.copy()
        rv = r * v                                        # precompute r * vertex
        for t in range(T):
            acc = one_minus * acc + rv[t]
            P[t] = acc
        return P

    def context_points(self, ids: np.ndarray | str | bytes, use_fast: bool | None = None) -> np.ndarray:
        """Points that represent the *context preceding* each position.

        ``C[t]`` is where the trajectory sits just before symbol ``t`` is read,
        i.e. the geometric summary of symbols ``0..t-1``. ``C[0]`` is the
        centre. This is what the cortex uses to predict symbol ``t``.
        """
        P = self.embed(ids, use_fast=use_fast)
        if len(P) == 0:
            return P
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
        if isinstance(pts, (str, bytes)) or np.ndim(pts) == 1:
            pts = self.embed(pts)
        if len(pts) == 0:
            return np.zeros((grid, grid), dtype=np.float64)
        ij = np.clip((pts * grid).astype(np.int64), 0, grid - 1)
        img = np.zeros((grid, grid), dtype=np.float64)
        np.add.at(img, (ij[:, 1], ij[:, 0]), 1.0)         # (row=y, col=x)
        total = img.sum()
        return img / total if total > 0 else img
