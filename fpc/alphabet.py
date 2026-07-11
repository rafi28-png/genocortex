"""
Alphabet handling -- the engine is *alphabet-agnostic*.
=======================================================

Everything downstream (the fractal embedding and the cortex) speaks integer
symbol ids, never raw characters. This one module is the only place that knows
about letters, so the *same* brain works on DNA, RNA, protein, or any custom
symbolic alphabet you invent.

Design notes
------------
* Symbols are mapped to contiguous integer ids ``0 .. size-1``.
* Unknown / ambiguous characters (e.g. IUPAC 'N', or a gap '-') are folded to a
  dedicated *wildcard* id so the pipeline never crashes on messy real data.
* Case is normalised to upper-case for nucleic acids and proteins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Alphabet:
    """An immutable symbol alphabet.

    Parameters
    ----------
    name:
        Human readable name, e.g. ``"DNA"``.
    symbols:
        Ordered string of canonical symbols, e.g. ``"ACGT"``. Order matters --
        it fixes the vertex order used by the Chaos Game embedding, so keep it
        stable across a project.
    wildcard:
        A single character used for anything not in ``symbols`` (ambiguity
        codes, gaps, whitespace...). It receives its own id at the end so a
        wildcard is *never* silently confused with a real symbol.
    """

    name: str
    symbols: str
    wildcard: str = "*"
    _index: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"duplicate symbols in alphabet {self.name!r}")
        if self.wildcard in self.symbols:
            raise ValueError("wildcard must not be one of the canonical symbols")
        idx = {ch: i for i, ch in enumerate(self.symbols)}
        idx[self.wildcard] = len(self.symbols)
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "_index", idx)

    # -- sizes -----------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of *canonical* symbols (excludes the wildcard)."""
        return len(self.symbols)

    @property
    def full_size(self) -> int:
        """Number of ids including the wildcard."""
        return len(self.symbols) + 1

    @property
    def wildcard_id(self) -> int:
        return len(self.symbols)

    # -- encoding / decoding --------------------------------------------
    def encode(self, seq: str) -> np.ndarray:
        """Turn a string into an int32 array of symbol ids.

        Characters not in the alphabet map to :pyattr:`wildcard_id`. This is
        intentionally forgiving so genuine genomic files (with 'N', '-', line
        breaks that were stripped upstream, etc.) never raise.
        """
        up = seq.upper()
        wid = self.wildcard_id
        idx = self._index
        return np.fromiter((idx.get(ch, wid) for ch in up), dtype=np.int32, count=len(up))

    def decode(self, ids) -> str:
        """Inverse of :pymeth:`encode` (wildcards render as ``self.wildcard``)."""
        table = self.symbols + self.wildcard
        return "".join(table[int(i)] for i in ids)

    def clean_fraction(self, ids: np.ndarray) -> float:
        """Fraction of symbols that are *not* wildcards -- a data-quality gauge."""
        if len(ids) == 0:
            return 1.0
        return float(np.mean(ids != self.wildcard_id))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.size


# -- ready-made alphabets -----------------------------------------------
DNA = Alphabet("DNA", "ACGT", wildcard="N")
RNA = Alphabet("RNA", "ACGU", wildcard="N")
# The 20 standard amino acids (single-letter), plus '*' wildcard for X/B/Z/gaps.
PROTEIN = Alphabet("PROTEIN", "ACDEFGHIKLMNPQRSTVWY", wildcard="X")
