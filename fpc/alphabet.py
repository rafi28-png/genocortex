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
* Encoding, decoding, and reverse-complementation use O(N) vectorised 256-byte
  lookup tables so multi-megabase chromosomes process in milliseconds.
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
    complements:
        Optional string of complementary symbols aligned position-by-position
        with ``symbols`` (e.g. ``"TGCA"`` for DNA ``"ACGT"``). Enables
        double-strand / reverse-complement operations for nucleic acids.
    """

    name: str
    symbols: str
    wildcard: str = "*"
    complements: str | None = None
    _index: dict = field(default_factory=dict, compare=False, repr=False)
    _encode_lut: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        compare=False,
        repr=False,
    )
    _decode_bytes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8),
        compare=False,
        repr=False,
    )
    _rc_lut: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.symbols) == 0:
            raise ValueError(f"alphabet {self.name!r} must contain at least 1 symbol")
        if len(self.symbols) > 255:
            raise ValueError(f"alphabet {self.name!r} exceeds 255 symbols ({len(self.symbols)})")
        if not self.symbols.isascii():
            raise ValueError(f"alphabet {self.name!r} symbols must be ASCII")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"duplicate symbols in alphabet {self.name!r}")
        if len(self.wildcard) != 1 or not self.wildcard.isascii():
            raise ValueError("wildcard must be a single ASCII character")
        if self.wildcard in self.symbols:
            raise ValueError("wildcard must not be one of the canonical symbols")

        idx = {ch: i for i, ch in enumerate(self.symbols)}
        idx[self.wildcard] = len(self.symbols)
        object.__setattr__(self, "_index", idx)

        wid = len(self.symbols)
        lut = np.full(256, wid, dtype=np.int32)
        for ch, i in idx.items():
            o_up = ord(ch.upper())
            o_lo = ord(ch.lower())
            if o_up < 128:
                lut[o_up] = i
            if o_lo < 128:
                lut[o_lo] = i
        object.__setattr__(self, "_encode_lut", lut)

        table_str = self.symbols + self.wildcard
        dec = np.frombuffer(table_str.encode("ascii"), dtype=np.uint8).copy()
        object.__setattr__(self, "_decode_bytes", dec)

        if self.complements is not None:
            if len(self.complements) != len(self.symbols):
                raise ValueError("complements must have the same length as symbols")
            if set(self.complements) != set(self.symbols):
                raise ValueError("complements must be a permutation of symbols")
            rc = np.empty(len(self.symbols) + 1, dtype=np.int32)
            for i, comp_ch in enumerate(self.complements):
                rc[i] = idx[comp_ch]
            rc[wid] = wid
            object.__setattr__(self, "_rc_lut", rc)

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

    @property
    def is_complementary(self) -> bool:
        """True if this alphabet defines base-pairing complements (DNA/RNA)."""
        return self._rc_lut is not None

    # -- encoding / decoding --------------------------------------------
    def encode(self, seq: str | bytes) -> np.ndarray:
        """Turn a string (or ASCII bytes) into an int32 array of symbol ids.

        Characters not in the alphabet map to :pyattr:`wildcard_id`. This is
        intentionally forgiving so genuine genomic files (with 'N', '-', line
        breaks that were stripped upstream, etc.) never raise.
        """
        if isinstance(seq, bytes):
            raw = np.frombuffer(seq, dtype=np.uint8)
            return self._encode_lut[raw].copy()
        if not seq:
            return np.empty(0, dtype=np.int32)
        if seq.isascii():
            raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            return self._encode_lut[raw].copy()
        lut = self._encode_lut
        wid = self.wildcard_id
        return np.array(
            [int(lut[ord(ch)]) if ord(ch) < 128 else wid for ch in seq],
            dtype=np.int32,
        )

    def decode(self, ids) -> str:
        """Inverse of :pymeth:`encode` (wildcards render as ``self.wildcard``)."""
        arr = np.asarray(ids, dtype=np.int32)
        if arr.size == 0:
            return ""
        clipped = np.where((arr >= 0) & (arr < self.full_size), arr, self.wildcard_id)
        return self._decode_bytes[clipped].tobytes().decode("ascii")

    # -- strand operations ----------------------------------------------
    def reverse_complement_ids(self, ids: np.ndarray) -> np.ndarray:
        """Return reverse-complemented symbol-id array (O(N) vectorised)."""
        if self._rc_lut is None:
            raise ValueError(f"alphabet {self.name!r} does not define complements")
        arr = np.asarray(ids, dtype=np.int32)
        if arr.size == 0:
            return np.empty(0, dtype=np.int32)
        clipped = np.where((arr >= 0) & (arr < self.full_size), arr, self.wildcard_id)
        return self._rc_lut[clipped][::-1].copy()

    def reverse_complement(self, seq: str) -> str:
        """Return the reverse-complement string (non-canonical chars -> wildcard)."""
        return self.decode(self.reverse_complement_ids(self.encode(seq)))

    def clean_fraction(self, ids: np.ndarray) -> float:
        """Fraction of symbols that are *not* wildcards -- a data-quality gauge."""
        if len(ids) == 0:
            return 1.0
        return float(np.mean(np.asarray(ids) != self.wildcard_id))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.size


# -- ready-made alphabets -----------------------------------------------
DNA = Alphabet("DNA", "ACGT", wildcard="N", complements="TGCA")
RNA = Alphabet("RNA", "ACGU", wildcard="N", complements="UGCA")
# The 20 standard amino acids (single-letter), plus 'X' wildcard for X/B/Z/gaps.
PROTEIN = Alphabet("PROTEIN", "ACDEFGHIKLMNPQRSTVWY", wildcard="X")

BUILTIN_ALPHABETS = {
    "DNA": DNA,
    "RNA": RNA,
    "PROTEIN": PROTEIN,
}


def get_alphabet(name: str) -> Alphabet:
    """Resolve a built-in alphabet by case-insensitive name."""
    key = name.strip().upper()
    if key not in BUILTIN_ALPHABETS:
        raise ValueError(f"unknown alphabet {name!r}; choose from {sorted(BUILTIN_ALPHABETS)}")
    return BUILTIN_ALPHABETS[key]
