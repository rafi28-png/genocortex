"""
The currency of the brain -- *surprise*.
========================================

Everything the Fractal Predictive Cortex does is measured in one unit: the
number of bits it takes to encode the symbol that actually happened, given the
symbol it expected. That is ``surprise_t = -log2 P(s_t | context)``.

* Low surprise  -> "I saw this coming" (the model understands this region).
* High surprise -> "that shocked me" (novelty, a mutation, a foreign insert).

This module holds the per-position surprise trace and turns it into
interpretable, statistically-grounded read-outs:

* ``total_bits`` / ``mean_bits`` -- the sequence's description length and its
  per-symbol cross-entropy (lower = the model fits better; this *is* the object
  the whole engine minimises).
* ``perplexity`` -- the effective branching factor, an intuitive quality gauge.
* ``anomalies(...)`` -- a robust scan statistic that flags mutation / anomaly
  regions with a per-region p-value and a multiple-testing correction, so a
  "hit" is defensible rather than eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy.stats import norm as _norm
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a declared dependency
    _HAVE_SCIPY = False


def benjamini_hochberg(p_values: np.ndarray, total_tests: int | None = None) -> np.ndarray:
    """Compute step-up Benjamini-Hochberg FDR q-values.

    Parameters
    ----------
    p_values:
        1-D array of raw upper-tail p-values for candidate regions.
    total_tests:
        Effective number of tests ``M`` across the scanned genome/sequence
        (defaults to ``len(p_values)``, coerced to ``>= len(p_values)``).
    """
    p = np.asarray(p_values, dtype=np.float64)
    k = len(p)
    if k == 0:
        return np.empty(0, dtype=np.float64)
    m = max(k, int(total_tests) if total_tests is not None else k)
    order = np.argsort(p, kind="mergesort")
    ranked_p = p[order]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    raw_q = ranked_p * (m / ranks)
    # Enforce monotonicity from largest rank down to 1, capped at 1.0
    mono_q = np.minimum(1.0, np.minimum.accumulate(raw_q[::-1])[::-1])
    out = np.empty(k, dtype=np.float64)
    out[order] = mono_q
    return out


@dataclass
class Anomaly:
    """One flagged region of unusually high surprise."""
    start: int                  # inclusive, 0-based position in the sequence
    end: int                    # inclusive
    peak_pos: int               # position of maximum smoothed surprise
    peak_z: float               # robust z-score at the peak
    mean_bits: float            # average surprise (bits) inside the region
    p_value: float              # per-region significance (upper tail)
    significant: bool           # survives Bonferroni family-wise error control
    q_value: float = 1.0        # Benjamini-Hochberg FDR-adjusted q-value
    fdr_significant: bool = False  # survives Benjamini-Hochberg FDR <= alpha

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        flag = "*" if self.significant else ("+" if self.fdr_significant else " ")
        return (f"Anomaly{flag}[{self.start}-{self.end}] "
                f"z={self.peak_z:.1f} bits={self.mean_bits:.2f} "
                f"p={self.p_value:.1e} q={self.q_value:.1e}")


class SurpriseTrack:
    """A per-position record of the brain's surprise (in bits).

    ``bits[t]`` is ``NaN`` at positions that could not be scored (wildcard /
    ambiguous symbols); every summary statistic ignores those positions.
    """

    def __init__(self, bits: np.ndarray):
        self.bits = np.asarray(bits, dtype=np.float64)
        self.valid = ~np.isnan(self.bits)

    # -- global summaries ------------------------------------------------
    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    @property
    def total_bits(self) -> float:
        """Description length of the sequence under the model (bits)."""
        return float(np.nansum(self.bits))

    @property
    def mean_bits(self) -> float:
        """Cross-entropy per symbol (bits) -- the core quality number."""
        n = self.n_valid
        return float(self.total_bits / n) if n else float("nan")

    @property
    def perplexity(self) -> float:
        """Effective branching factor = 2 ** mean_bits."""
        m = self.mean_bits
        return float(2.0 ** m) if np.isfinite(m) else float("nan")

    # -- local anomaly detection ----------------------------------------
    def smoothed(self, window: int = 25) -> np.ndarray:
        """Surprise smoothed by a centred moving average over valid positions.

        NaNs are handled by averaging only the valid samples inside each window
        (so a run of wildcards does not create a fake dip).
        """
        b = np.where(self.valid, self.bits, 0.0)
        w = self.valid.astype(np.float64)
        if window < 1:
            window = 1
        kernel = np.ones(window, dtype=np.float64)
        num = np.convolve(b, kernel, mode="same")
        den = np.convolve(w, kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return out

    def anomalies(self, window: int = 25, z_threshold: float = 4.0,
                  alpha: float = 0.05) -> list[Anomaly]:
        """Flag regions of unusually high surprise (candidate mutations / inserts).

        Method: smooth the surprise, build a *robust* baseline from the median
        and MAD (so a few big spikes do not inflate the baseline), z-score every
        position, then merge contiguous runs above ``z_threshold`` into regions.
        Each region gets an upper-tail Gaussian p-value at its peak, a
        Bonferroni ``significant`` flag, and a Benjamini-Hochberg FDR ``q_value``
        and ``fdr_significant`` flag against the number of independent windows.

        Parameters
        ----------
        window:       smoothing / scan width, in symbols.
        z_threshold:  robust-z cutoff for calling a position "surprising".
        alpha:        error-rate cutoff for ``significant`` (FWER) and
                      ``fdr_significant`` (Benjamini-Hochberg FDR).
        """
        s = self.smoothed(window)
        finite = np.isfinite(s)
        if finite.sum() < 3:
            return []
        vals = s[finite]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad if mad > 0 else (np.std(vals) or 1.0)
        z = np.full_like(s, np.nan)
        z[finite] = (s[finite] - med) / scale

        above = np.zeros(s.shape, dtype=bool)
        above[finite] = z[finite] >= z_threshold

        # Number of roughly-independent tests (for Bonferroni & BH FDR).
        n_tests = max(1, int(finite.sum() // max(1, window)))
        p_cut = alpha / n_tests

        anomalies: list[Anomaly] = []
        i = 0
        n = len(above)
        while i < n:
            if not above[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            seg = slice(i, j + 1)
            zseg = z[seg]
            peak_local = int(np.nanargmax(zseg))
            peak_pos = i + peak_local
            peak_z = float(zseg[peak_local])
            with np.errstate(invalid="ignore"):
                region_bits = float(np.nanmean(self.bits[seg]))
            if _HAVE_SCIPY:
                p = float(_norm.sf(peak_z))
            else:  # pragma: no cover - fallback tail bound
                p = float(np.exp(-0.5 * peak_z * peak_z))
            anomalies.append(Anomaly(
                start=i, end=j, peak_pos=peak_pos, peak_z=peak_z,
                mean_bits=region_bits, p_value=p, significant=(p <= p_cut),
            ))
            i = j + 1

        if anomalies:
            q_vals = benjamini_hochberg(
                np.array([a.p_value for a in anomalies], dtype=np.float64),
                total_tests=max(n_tests, len(anomalies)),
            )
            for anom, qv in zip(anomalies, q_vals):
                anom.q_value = float(qv)
                anom.fdr_significant = bool(qv <= alpha)
        return anomalies

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.bits)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"SurpriseTrack(n={len(self.bits)}, valid={self.n_valid}, "
                f"mean_bits={self.mean_bits:.3f}, perplexity={self.perplexity:.2f})")
