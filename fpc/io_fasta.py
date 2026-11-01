"""
A tiny, dependency-free FASTA reader/writer.
=============================================

Deliberately self-contained (no BioPython needed) so the whole engine keeps
running on nothing but NumPy/SciPy. Handles multi-record files, wrapped or
unwrapped sequence lines, blank lines, Windows/Unix newlines, and transparently
reads gzip-compressed ``.fasta.gz`` files.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
import os


@dataclass
class Record:
    """One FASTA record."""
    id: str          # the token right after '>'
    description: str  # the full header line (without '>')
    seq: str          # the sequence, concatenated and upper-cased

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.seq)


def _open_text(path: str | os.PathLike[str]):
    path_str = os.fspath(path)
    if path_str.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path_str, "rb"), encoding="utf-8")
    return open(path_str, "r", encoding="utf-8")


def read_fasta(path: str | os.PathLike[str]) -> list[Record]:
    """Read every record from a (optionally gzipped) FASTA file."""
    records: list[Record] = []
    header = None
    chunks: list[str] = []

    def flush():
        if header is not None:
            desc = header
            rid = header.split()[0] if header.split() else header
            records.append(Record(id=rid, description=desc, seq="".join(chunks).upper()))

    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
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
        flush()
    return records


def read_fasta_dir(directory: str | os.PathLike[str], exts=(".fasta", ".fa", ".fna", ".ffn", ".fasta.gz", ".fa.gz")) -> list[Record]:
    """Read every FASTA file in a directory (non-recursive)."""
    dir_str = os.fspath(directory)
    out: list[Record] = []
    for name in sorted(os.listdir(dir_str)):
        if name.lower().endswith(exts):
            out.extend(read_fasta(os.path.join(dir_str, name)))
    return out


def write_fasta(path: str | os.PathLike[str], records, width: int = 70) -> None:
    """Write records (``Record`` objects or ``(id, seq)`` tuples) to FASTA."""
    with open(os.fspath(path), "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            if isinstance(rec, Record):
                rid, desc, seq = rec.id, rec.description, rec.seq
            else:
                rid, seq = rec
                desc = rid
            fh.write(f">{desc}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")
