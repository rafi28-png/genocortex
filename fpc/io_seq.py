"""
Streaming FASTA & FASTQ sequence I/O with gzip and Phred quality masking.
=========================================================================

Self-contained (zero BioPython dependency) sequence readers and writers for real
bioinformatics pipelines:

* ``FASTA`` (``.fa``, ``.fna``, ``.ffn``, ``.faa``, ``.fasta``, plus ``.gz``)
* ``FASTQ`` (``.fq``, ``.fastq``, plus ``.gz``) — handles both standard 4-line
  and multi-line wrapped FASTQ records safely by matching quality length to
  sequence length (so ``@`` Phred+33 Q=31 bytes never break record boundaries).
* ``stdin`` streaming via ``path="-"``.
* Vectorised NumPy Phred quality masking: bases with ``Q < min_phred`` are
  replaced with the alphabet wildcard (default ``"N"``) so sequencing errors do
  not inject artificial surprise spikes or corrupt prototype counts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import io
import os
import sys
from typing import Iterable, Iterator
import numpy as np

from .io_fasta import Record, write_fasta

SEQ_EXTENSIONS = (
    ".fasta", ".fa", ".fna", ".ffn", ".faa",
    ".fastq", ".fq",
    ".fasta.gz", ".fa.gz", ".fna.gz", ".ffn.gz", ".faa.gz",
    ".fastq.gz", ".fq.gz",
)


@dataclass
class SeqRecord(Record):
    """A biological sequence record from FASTA or FASTQ.

    Subclasses :class:`fpc.io_fasta.Record` so existing code expecting
    ``Record(id, description, seq)`` works unchanged.
    """
    qual: str | None = None

    @property
    def mean_phred(self) -> float:
        """Mean Phred+33 quality score, or ``NaN`` if no quality string is present."""
        if not self.qual:
            return float("nan")
        q = np.frombuffer(self.qual.encode("ascii", "replace"), dtype=np.uint8)
        return float(np.mean(q.astype(np.float64) - 33.0))


@contextmanager
def _open_text_stream(path: str | os.PathLike[str]):
    """Open plain text, ``.gz``, or ``"-"`` (stdin) as a UTF-8 text stream."""
    path_str = os.fspath(path)
    if path_str == "-":
        yield sys.stdin
        return
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Sequence file not found: {path_str!r}")
    if path_str.lower().endswith(".gz"):
        gz = gzip.open(path_str, "rb")
        wrapper = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
        try:
            yield wrapper
        finally:
            wrapper.close()
    else:
        fh = open(path_str, "r", encoding="utf-8", errors="replace")
        try:
            yield fh
        finally:
            fh.close()


def mask_low_quality(
    seq: str,
    qual: str,
    min_phred: int = 0,
    wildcard: str = "N",
    phred_offset: int = 33,
) -> str:
    """Replace bases whose Phred quality is ``< min_phred`` with ``wildcard``."""
    if len(seq) != len(qual):
        raise ValueError(
            f"FASTQ sequence length ({len(seq)}) != quality length ({len(qual)})"
        )
    if min_phred <= 0 or not seq:
        return seq
    wild_byte = ord(wildcard[0])
    if seq.isascii():
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).copy()
    else:
        seq_bytes = np.array(
            [ord(ch) if ord(ch) < 128 else wild_byte for ch in seq], dtype=np.uint8
        )
    if qual.isascii():
        qual_bytes = np.frombuffer(qual.encode("ascii"), dtype=np.uint8)
    else:
        floor_byte = max(0, min(255, int(phred_offset)))
        qual_bytes = np.array(
            [ord(ch) if 33 <= ord(ch) <= 126 else floor_byte for ch in qual],
            dtype=np.uint8,
        )
    cutoff = int(phred_offset) + int(min_phred)
    seq_bytes[qual_bytes < cutoff] = wild_byte
    return seq_bytes.tobytes().decode("ascii")


def _iter_fasta_handle(fh: Iterable[str]) -> Iterator[SeqRecord]:
    header: str | None = None
    chunks: list[str] = []
    saw_any_content = False

    for raw in fh:
        line = raw.strip()
        if not line:
            continue
        saw_any_content = True
        if line.startswith(">"):
            if header is not None:
                rid = header.split()[0] if header.split() else header
                yield SeqRecord(id=rid, description=header, seq="".join(chunks).upper(), qual=None)
            header = line[1:].strip()
            if not header:
                raise ValueError("Malformed FASTA record: empty header after '>'")
            chunks = []
        else:
            if header is None:
                raise ValueError(
                    f"Malformed FASTA input: expected '>' header, got {line[:40]!r}"
                )
            chunks.append(line)

    if header is not None:
        rid = header.split()[0] if header.split() else header
        yield SeqRecord(id=rid, description=header, seq="".join(chunks).upper(), qual=None)
    elif saw_any_content:
        raise ValueError("Malformed FASTA input: no '>' records found")


def _iter_fastq_handle(
    fh: Iterable[str],
    min_phred: int = 0,
    wildcard: str = "N",
    phred_offset: int = 33,
) -> Iterator[SeqRecord]:
    it = iter(fh)
    while True:
        # 1. Find next non-empty header line starting with '@'
        header_line = None
        for raw in it:
            s = raw.strip()
            if s:
                header_line = s
                break
        if header_line is None:
            break
        if not header_line.startswith("@"):
            raise ValueError(
                f"Malformed FASTQ record: expected '@' header, got {header_line[:40]!r}"
            )
        desc = header_line[1:].strip()
        if not desc:
            raise ValueError("Malformed FASTQ record: empty header after '@'")
        rid = desc.split()[0]

        # 2. Read sequence lines until '+' separator
        seq_chunks: list[str] = []
        saw_plus = False
        for raw in it:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("+"):
                saw_plus = True
                break
            seq_chunks.append(s)
        if not saw_plus:
            raise ValueError(f"Malformed FASTQ record {rid!r}: missing '+' separator line")

        seq = "".join(seq_chunks).upper()
        target_len = len(seq)

        # 3. Read quality lines until exact target_len characters collected
        qual_chunks: list[str] = []
        qual_len = 0
        while qual_len < target_len:
            try:
                raw_q = next(it)
            except StopIteration as exc:
                raise ValueError(
                    f"Malformed FASTQ record {rid!r}: truncated quality string "
                    f"({qual_len} < {target_len})"
                ) from exc
            sq = raw_q.strip()
            if not sq:
                continue
            qual_chunks.append(sq)
            qual_len += len(sq)

        qual = "".join(qual_chunks)
        if len(qual) != target_len:
            raise ValueError(
                f"Malformed FASTQ record {rid!r}: quality length ({len(qual)}) "
                f"does not match sequence length ({target_len})"
            )

        if min_phred > 0:
            seq = mask_low_quality(
                seq, qual, min_phred=min_phred, wildcard=wildcard, phred_offset=phred_offset
            )
        yield SeqRecord(id=rid, description=desc, seq=seq, qual=qual)


def iter_fasta(path: str) -> Iterator[SeqRecord]:
    """Stream :class:`SeqRecord` entries from a FASTA (or ``.fasta.gz``) file."""
    with _open_text_stream(path) as fh:
        yield from _iter_fasta_handle(fh)


def read_fasta_records(path: str) -> list[SeqRecord]:
    """Read all FASTA records into memory."""
    return list(iter_fasta(path))


def iter_fastq(
    path: str,
    min_phred: int = 0,
    wildcard: str = "N",
    phred_offset: int = 33,
) -> Iterator[SeqRecord]:
    """Stream :class:`SeqRecord` entries from a FASTQ (or ``.fastq.gz``) file."""
    with _open_text_stream(path) as fh:
        yield from _iter_fastq_handle(
            fh, min_phred=min_phred, wildcard=wildcard, phred_offset=phred_offset
        )


def read_fastq(
    path: str,
    min_phred: int = 0,
    wildcard: str = "N",
    phred_offset: int = 33,
) -> list[SeqRecord]:
    """Read all FASTQ records into memory with optional Phred quality masking."""
    return list(
        iter_fastq(path, min_phred=min_phred, wildcard=wildcard, phred_offset=phred_offset)
    )


def iter_sequences(
    path: str,
    min_phred: int = 0,
    wildcard: str = "N",
    phred_offset: int = 33,
) -> Iterator[SeqRecord]:
    """Auto-detect FASTA vs FASTQ (including ``.gz``, ``"-"`` stdin, or directory) and stream."""
    if path != "-" and os.path.isdir(path):
        found = False
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(SEQ_EXTENSIONS):
                found = True
                yield from iter_sequences(
                    os.path.join(path, name),
                    min_phred=min_phred,
                    wildcard=wildcard,
                    phred_offset=phred_offset,
                )
        if not found:
            raise ValueError(f"No FASTA/FASTQ sequence files found in directory {path!r}")
        return

    with _open_text_stream(path) as fh:
        first_nonempty = None
        for raw in fh:
            if raw.strip():
                first_nonempty = raw
                break
        if first_nonempty is None:
            raise ValueError(f"Sequence input {path!r} is empty")

        lead = first_nonempty.lstrip()[0]
        chained = _chain_first_line(first_nonempty, fh)
        if lead == ">":
            yield from _iter_fasta_handle(chained)
        elif lead == "@":
            yield from _iter_fastq_handle(
                chained, min_phred=min_phred, wildcard=wildcard, phred_offset=phred_offset
            )
        else:
            raise ValueError(
                f"Unrecognised sequence format in {path!r}: expected '>' (FASTA) or '@' (FASTQ), "
                f"got {first_nonempty.strip()[:40]!r}"
            )


def _chain_first_line(first: str, rest: Iterable[str]) -> Iterator[str]:
    yield first
    yield from rest


def read_sequences(
    path: str,
    min_phred: int = 0,
    wildcard: str = "N",
    phred_offset: int = 33,
) -> list[SeqRecord]:
    """Read all sequences from a FASTA/FASTQ file, ``.gz`` archive, ``"-"``, or directory."""
    records = list(
        iter_sequences(path, min_phred=min_phred, wildcard=wildcard, phred_offset=phred_offset)
    )
    if not records:
        raise ValueError(f"No sequence records found in {path!r}")
    return records


__all__ = [
    "SeqRecord",
    "SEQ_EXTENSIONS",
    "mask_low_quality",
    "iter_fasta",
    "read_fasta_records",
    "iter_fastq",
    "read_fastq",
    "iter_sequences",
    "read_sequences",
    "write_fasta",
]
