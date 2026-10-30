"""
Megabase Genomic Island & Horizontal Gene Transfer (HGT) Scanner.
=================================================================

Scans complete chromosomes or multi-contig assemblies for compositionally
anomalous islands (pathogenicity islands, prophages, integrative conjugative
elements, viral insertions, or sample contamination) using the Fractal
Predictive Cortex:

1. **Reference-guided mode** (:func:`scan_against_reference`):
   Trains a double-strand cortex on a core/reference genome (e.g., commensal
   *Escherichia coli* K-12 MG1655) and scans a query genome (e.g., pathogenic
   *E. coli* O157:H7) in linear time. Inverted core blocks are recognised
   automatically because both chromosomal strands are learned symmetrically,
   isolating genuine foreign acquisitions.

2. **Reference-free single-genome mode** (:func:`scan_self_islands`):
   Uses buffered *K*-fold leave-one-segment-out cross-scoring across the genome
   so 10–100 kb prophages and pathogenicity islands are excluded (with a guard
   band) from the training set when their locus is scored, preventing large
   islands from memorising themselves away.

Exports standard 0-based half-open ``BED6`` (:func:`write_bed`), ``BedGraph``
(:func:`write_bedgraph`), and rich tabular ``TSV`` (:func:`write_islands_tsv`)
files ready for IGV, UCSC Genome Browser, or downstream `bedtools` workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

try:
    from scipy.stats import norm as _norm
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from .alphabet import Alphabet, DNA
from .cortex import FractalPredictiveCortex
from .io_fasta import Record
from .surprise import benjamini_hochberg


@dataclass
class GenomicIsland:
    """A genomic island called from elevated predictive surprise.

    Coordinates follow the standard 0-based half-open ``[start, end)`` BED
    convention.
    """
    chrom: str
    start: int          # 0-based inclusive
    end: int            # 0-based exclusive
    peak_pos: int       # 0-based coordinate of peak window midpoint
    peak_z: float       # robust z-score of peak window
    mean_bits: float    # mean predictive surprise (bits/symbol) across [start, end)
    delta_bits: float   # excess surprise over genome median window baseline
    gc_content: float   # GC fraction in [0, 1] (NaN for non-nucleotide alphabets)
    delta_gc: float     # island GC minus chromosome background GC
    p_value: float      # upper-tail Gaussian p-value at peak_z
    q_value: float      # genome-wide Benjamini-Hochberg FDR q-value
    significant: bool   # q_value <= alpha

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class WindowTrack:
    """Sliding-window surprise and z-score profile along one chromosome/contig."""
    chrom: str
    starts: np.ndarray      # (W,) int64, 0-based inclusive
    ends: np.ndarray        # (W,) int64, 0-based exclusive
    mean_bits: np.ndarray   # (W,) float64
    z_scores: np.ndarray    # (W,) float64


def default_max_prototypes(alphabet: Alphabet, order: int) -> int:
    """Scale prototype budget to prevent prototype starvation on megabase genomes."""
    approx_contexts = float(alphabet.size) ** min(int(order), 10)
    return max(8192, min(262144, int(1.25 * approx_contexts)))


def _coerce_records(records) -> list[Record]:
    if isinstance(records, str):
        return [Record(id="seq1", description="seq1", seq=records.upper())]
    if isinstance(records, Record):
        return [records]
    out: list[Record] = []
    for idx, item in enumerate(records):
        if isinstance(item, Record):
            out.append(item)
        elif isinstance(item, tuple) and len(item) >= 2:
            rid, seq = str(item[0]), str(item[1]).upper()
            out.append(Record(id=rid, description=rid, seq=seq))
        elif isinstance(item, str):
            rid = f"seq{idx + 1}"
            out.append(Record(id=rid, description=rid, seq=item.upper()))
        else:
            raise TypeError(f"Unsupported sequence record type: {type(item)!r}")
    if not out:
        raise ValueError("Expected at least one sequence record")
    return out


def _build_gc_cumsums(ids: np.ndarray, alphabet: Alphabet) -> tuple[np.ndarray, np.ndarray] | None:
    """Return O(1) prefix sums (csum_gc, csum_canonical) for DNA/RNA alphabets."""
    if not ("G" in alphabet.symbols and "C" in alphabet.symbols and alphabet.size == 4):
        return None
    g_id = alphabet.symbols.index("G")
    c_id = alphabet.symbols.index("C")
    is_gc = ((ids == g_id) | (ids == c_id)).astype(np.int64)
    is_canon = ((ids >= 0) & (ids < alphabet.size)).astype(np.int64)
    csum_gc = np.empty(len(ids) + 1, dtype=np.int64)
    csum_can = np.empty(len(ids) + 1, dtype=np.int64)
    csum_gc[0] = 0
    csum_can[0] = 0
    np.cumsum(is_gc, out=csum_gc[1:])
    np.cumsum(is_canon, out=csum_can[1:])
    return csum_gc, csum_can


def _window_shannon_h0(ids: np.ndarray, alphabet: Alphabet, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Compute exact 0th-order empirical Shannon entropy H0(W) per window in O(N + a*W)."""
    n = len(ids)
    w_count = len(starts)
    if w_count == 0 or n == 0:
        return np.zeros(w_count, dtype=np.float64)
    a = alphabet.size
    counts = np.zeros((w_count, a), dtype=np.float64)
    cs = np.empty(n + 1, dtype=np.int64)
    cs[0] = 0
    for sym in range(a):
        np.cumsum((ids == sym).astype(np.int64), out=cs[1:])
        counts[:, sym] = cs[ends] - cs[starts]
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(totals > 0, counts / totals, 0.0)
        terms = np.where(probs > 0, -probs * np.log2(np.where(probs > 0, probs, 1.0)), 0.0)
    return terms.sum(axis=1)


def _call_islands_from_bits(
    records: Sequence[Record],
    encoded_list: Sequence[np.ndarray],
    bits_list: Sequence[np.ndarray],
    alphabet: Alphabet,
    window: int = 1000,
    step: int = 250,
    z_threshold: float = 3.5,
    alpha: float = 0.05,
    merge_gap: int = 2000,
    min_length: int | None = None,
    normalize_entropy: bool = True,
) -> tuple[list[GenomicIsland], list[WindowTrack]]:
    """Compute O(N) sliding-window excess-surprise z-scores and merge FDR-controlled islands.

    When ``normalize_entropy=True`` (default), z-scores are computed on the
    excess predictive description length ``CE_model(W) - H_0(W)`` (the empirical
    Kullback-Leibler divergence beyond local 0th-order composition), preventing
    AT-rich or GC-rich compositional skew from masking foreign pathogenicity
    islands whose intrinsic 0th-order Shannon entropy ``H_0(W)`` is < 2 bits.
    """
    window = max(10, int(window))
    step = max(1, int(step))
    eff_min_len = int(min_length) if min_length is not None else window

    raw_windows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    all_valid_scores: list[np.ndarray] = []
    all_valid_win_bits: list[np.ndarray] = []
    total_valid_bases = 0

    bits_csums: list[tuple[np.ndarray, np.ndarray]] = []
    gc_csums: list[tuple[np.ndarray, np.ndarray] | None] = []

    for rec, ids, bits in zip(records, encoded_list, bits_list):
        n = len(bits)
        valid = np.isfinite(bits)
        total_valid_bases += int(valid.sum())

        cs_b = np.empty(n + 1, dtype=np.float64)
        cs_v = np.empty(n + 1, dtype=np.int64)
        cs_b[0] = 0.0
        cs_v[0] = 0
        np.cumsum(np.where(valid, bits, 0.0), out=cs_b[1:])
        np.cumsum(valid.astype(np.int64), out=cs_v[1:])
        bits_csums.append((cs_b, cs_v))
        gc_csums.append(_build_gc_cumsums(ids, alphabet))

        if n <= window:
            starts = np.array([0], dtype=np.int64) if n > 0 else np.empty(0, dtype=np.int64)
            ends = np.array([n], dtype=np.int64) if n > 0 else np.empty(0, dtype=np.int64)
        else:
            starts = np.arange(0, n - window + 1, step, dtype=np.int64)
            if starts[-1] + window < n:
                starts = np.append(starts, max(0, n - window))
            ends = np.minimum(n, starts + window)

        if len(starts) == 0:
            empty_f = np.empty(0, dtype=np.float64)
            raw_windows.append((rec.id, starts, ends, empty_f, empty_f))
            continue

        v_counts = cs_v[ends] - cs_v[starts]
        b_sums = cs_b[ends] - cs_b[starts]
        min_v = np.maximum(1, ((ends - starts) // 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            w_bits = np.where(v_counts >= min_v, b_sums / v_counts, np.nan)

        if normalize_entropy:
            h0 = _window_shannon_h0(ids, alphabet, starts, ends)
            w_score = np.where(np.isfinite(w_bits), w_bits - h0, np.nan)
        else:
            w_score = w_bits

        raw_windows.append((rec.id, starts, ends, w_bits, w_score))
        fin_s = w_score[np.isfinite(w_score)]
        fin_b = w_bits[np.isfinite(w_bits)]
        if len(fin_s) > 0:
            all_valid_scores.append(fin_s)
        if len(fin_b) > 0:
            all_valid_win_bits.append(fin_b)

    if not all_valid_scores:
        empty_tracks = [
            WindowTrack(chrom=cid, starts=st, ends=en, mean_bits=wb, z_scores=np.full_like(wb, np.nan))
            for cid, st, en, wb, _ in raw_windows
        ]
        return [], empty_tracks

    pooled_scores = np.concatenate(all_valid_scores)
    med_score = float(np.median(pooled_scores))
    mad = float(np.median(np.abs(pooled_scores - med_score)))
    scale = 1.4826 * mad if mad > 0 else (float(np.std(pooled_scores)) or 1.0)
    med_bits = float(np.median(np.concatenate(all_valid_win_bits)))

    window_tracks: list[WindowTrack] = []
    islands: list[GenomicIsland] = []

    for idx_rec, (cid, starts, ends, w_bits, w_score) in enumerate(raw_windows):
        z = np.where(np.isfinite(w_score), (w_score - med_score) / scale, np.nan)
        window_tracks.append(
            WindowTrack(chrom=cid, starts=starts, ends=ends, mean_bits=w_bits, z_scores=z)
        )

        flagged = np.flatnonzero(np.isfinite(z) & (z >= z_threshold))
        if len(flagged) == 0:
            continue

        cs_b, cs_v = bits_csums[idx_rec]
        gc_pair = gc_csums[idx_rec]
        if gc_pair is not None:
            cs_gc, cs_can = gc_pair
            tot_can = int(cs_can[-1])
            chrom_gc = float(cs_gc[-1] / tot_can) if tot_can > 0 else float("nan")
        else:
            chrom_gc = float("nan")

        # Group flagged windows separated by <= merge_gap
        grp_start_idx = int(flagged[0])
        grp_end_idx = int(flagged[0])
        groups: list[tuple[int, int]] = []
        for fi in flagged[1:]:
            fi_int = int(fi)
            if int(starts[fi_int]) - int(ends[grp_end_idx]) <= merge_gap:
                grp_end_idx = fi_int
            else:
                groups.append((grp_start_idx, grp_end_idx))
                grp_start_idx = fi_int
                grp_end_idx = fi_int
        groups.append((grp_start_idx, grp_end_idx))

        for g0, g1 in groups:
            istart = int(starts[g0])
            iend = int(ends[g1])
            if (iend - istart) < eff_min_len:
                continue

            sub_z = z[g0 : g1 + 1]
            local_best = int(np.nanargmax(sub_z))
            best_w = g0 + local_best
            peak_z = float(z[best_w])
            peak_pos = int((starts[best_w] + ends[best_w]) // 2)

            v_cnt = int(cs_v[iend] - cs_v[istart])
            isl_bits = float((cs_b[iend] - cs_b[istart]) / v_cnt) if v_cnt > 0 else float("nan")
            delta_bits = isl_bits - med_bits if np.isfinite(isl_bits) else float("nan")

            if gc_pair is not None:
                cs_gc, cs_can = gc_pair
                can_cnt = int(cs_can[iend] - cs_can[istart])
                isl_gc = float((cs_gc[iend] - cs_gc[istart]) / can_cnt) if can_cnt > 0 else float("nan")
                d_gc = isl_gc - chrom_gc if np.isfinite(isl_gc) and np.isfinite(chrom_gc) else float("nan")
            else:
                isl_gc = float("nan")
                d_gc = float("nan")

            if _HAVE_SCIPY:
                pval = float(_norm.sf(peak_z))
            else:  # pragma: no cover
                pval = float(np.exp(-0.5 * peak_z * peak_z))

            islands.append(
                GenomicIsland(
                    chrom=cid,
                    start=istart,
                    end=iend,
                    peak_pos=peak_pos,
                    peak_z=peak_z,
                    mean_bits=isl_bits,
                    delta_bits=delta_bits,
                    gc_content=isl_gc,
                    delta_gc=d_gc,
                    p_value=pval,
                    q_value=1.0,
                    significant=False,
                )
            )

    if islands:
        n_indep_windows = max(1, total_valid_bases // window)
        q_vals = benjamini_hochberg(
            np.array([isl.p_value for isl in islands], dtype=np.float64),
            total_tests=max(n_indep_windows, len(islands)),
        )
        for isl, qv in zip(islands, q_vals):
            isl.q_value = float(qv)
            isl.significant = bool(qv <= alpha)

    return islands, window_tracks


def scan_against_reference(
    ref_records,
    query_records,
    alphabet: Alphabet = DNA,
    order: int = 7,
    max_prototypes: int | None = None,
    double_strand: bool = True,
    epochs: int = 1,
    window: int = 1000,
    step: int = 250,
    z_threshold: float = 3.5,
    alpha: float = 0.05,
    merge_gap: int = 2000,
    min_length: int | None = None,
    cortex: FractalPredictiveCortex | None = None,
    **cortex_kwargs,
) -> tuple[list[GenomicIsland], list[WindowTrack], FractalPredictiveCortex]:
    """Train on ``ref_records`` (unless ``cortex`` is pre-trained) and scan ``query_records``."""
    q_recs = _coerce_records(query_records)
    if cortex is None:
        r_recs = _coerce_records(ref_records)
        eff_max_proto = (
            int(max_prototypes)
            if max_prototypes is not None
            else default_max_prototypes(alphabet, order)
        )
        cortex = FractalPredictiveCortex(
            alphabet, order=order, max_prototypes=eff_max_proto, **cortex_kwargs
        )
        for r in r_recs:
            cortex.fit(alphabet.encode(r.seq), epochs=epochs, double_strand=double_strand)

    q_encoded = [alphabet.encode(r.seq) for r in q_recs]
    q_bits = [cortex.score(ids).bits for ids in q_encoded]
    islands, tracks = _call_islands_from_bits(
        q_recs,
        q_encoded,
        q_bits,
        alphabet=alphabet,
        window=window,
        step=step,
        z_threshold=z_threshold,
        alpha=alpha,
        merge_gap=merge_gap,
        min_length=min_length,
    )
    return islands, tracks, cortex


def scan_self_islands(
    query_records,
    alphabet: Alphabet = DNA,
    order: int = 7,
    max_prototypes: int | None = None,
    double_strand: bool = True,
    n_folds: int = 8,
    guard_bp: int = 25000,
    window: int = 1000,
    step: int = 250,
    z_threshold: float = 3.5,
    alpha: float = 0.05,
    merge_gap: int = 2000,
    min_length: int | None = None,
    **cortex_kwargs,
) -> tuple[list[GenomicIsland], list[WindowTrack]]:
    """Reference-free single-genome island discovery via buffered K-fold cross-scoring.

    Partitions the concatenated assembly into ``n_folds`` contiguous segments.
    Each segment is scored by a cortex trained exclusively on the rest of the
    genome outside a ``guard_bp`` flanking buffer, ensuring 10-100 kb islands
    cannot memorise themselves into the background model.
    """
    q_recs = _coerce_records(query_records)
    q_encoded = [alphabet.encode(r.seq) for r in q_recs]
    eff_max_proto = (
        int(max_prototypes)
        if max_prototypes is not None
        else default_max_prototypes(alphabet, order)
    )

    lengths = [len(ids) for ids in q_encoded]
    total_bp = sum(lengths)
    if total_bp == 0:
        raise ValueError("Query sequence(s) contain 0 bases")

    # Concatenate with wildcard spacer so contexts do not bleed across contigs
    spacer = np.full(max(1, order), alphabet.wildcard_id, dtype=np.int64)
    pieces: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for idx, ids in enumerate(q_encoded):
        if idx > 0:
            pieces.append(spacer)
            cursor += len(spacer)
        start_c = cursor
        pieces.append(ids)
        cursor += len(ids)
        offsets.append((start_c, cursor))
    concat_ids = np.concatenate(pieces)
    concat_len = len(concat_ids)

    eff_folds = max(2, min(int(n_folds), max(2, concat_len // max(200, window * 2))))
    eff_guard = min(int(guard_bp), concat_len // (2 * eff_folds))
    fold_edges = np.linspace(0, concat_len, eff_folds + 1, dtype=np.int64)

    oof_bits = np.full(concat_len, np.nan, dtype=np.float64)
    ctx_pad = max(1, order)

    for f in range(eff_folds):
        f_start = int(fold_edges[f])
        f_end = int(fold_edges[f + 1])
        if f_end <= f_start:
            continue

        excl_start = max(0, f_start - eff_guard)
        excl_end = min(concat_len, f_end + eff_guard)

        cortex = FractalPredictiveCortex(
            alphabet, order=order, max_prototypes=eff_max_proto, **cortex_kwargs
        )
        if excl_start > 0:
            cortex.fit(concat_ids[:excl_start], epochs=1, double_strand=double_strand)
        if excl_end < concat_len:
            cortex.fit(concat_ids[excl_end:], epochs=1, double_strand=double_strand)

        # Include a small left prefix so CGR context is warmed up at f_start
        score_start = max(0, f_start - ctx_pad)
        scored = cortex.score(concat_ids[score_start:f_end]).bits
        oof_bits[f_start:f_end] = scored[f_start - score_start :]

    q_bits = [oof_bits[s:e].copy() for s, e in offsets]
    return _call_islands_from_bits(
        q_recs,
        q_encoded,
        q_bits,
        alphabet=alphabet,
        window=window,
        step=step,
        z_threshold=z_threshold,
        alpha=alpha,
        merge_gap=merge_gap,
        min_length=min_length,
    )


def write_bed(
    islands: Sequence[GenomicIsland],
    path: str,
    only_significant: bool = False,
) -> int:
    """Write islands in standard 6-column BED format (0-based half-open)."""
    written = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for idx, isl in enumerate(islands, start=1):
            if only_significant and not isl.significant:
                continue
            dgc_tag = f"_dGC{isl.delta_gc:+.3f}" if np.isfinite(isl.delta_gc) else ""
            sig_tag = "_FDR" if isl.significant else "_cand"
            name = f"island_{idx}_z{isl.peak_z:.2f}{dgc_tag}{sig_tag}"
            score = min(1000, max(0, int(round(isl.peak_z * 100.0))))
            fh.write(f"{isl.chrom}\t{isl.start}\t{isl.end}\t{name}\t{score}\t.\n")
            written += 1
    return written


def write_bedgraph(
    window_tracks: Sequence[WindowTrack],
    path: str,
    metric: str = "z_score",
    track_name: str = "GenoCortex_Surprise",
) -> int:
    """Write sliding-window surprise or z-score tracks in UCSC/IGV BedGraph format.

    Because BedGraph requires non-overlapping intervals, each window is emitted
    over ``[start, min(end, next_start))`` (tiling the chromosome cleanly).
    """
    if metric not in ("z_score", "bits"):
        raise ValueError(f"metric must be 'z_score' or 'bits', got {metric!r}")
    desc = "GenoCortex robust surprise z-score" if metric == "z_score" else "GenoCortex surprise (bits/base)"
    written = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f'track type=bedGraph name="{track_name}" description="{desc}"\n')
        for wt in window_tracks:
            vals = wt.z_scores if metric == "z_score" else wt.mean_bits
            m = len(wt.starts)
            for i in range(m):
                v = float(vals[i])
                if not np.isfinite(v):
                    continue
                s = int(wt.starts[i])
                e = int(wt.starts[i + 1]) if (i + 1 < m) else int(wt.ends[i])
                if e <= s:
                    continue
                fh.write(f"{wt.chrom}\t{s}\t{e}\t{v:.4f}\n")
                written += 1
    return written


def write_islands_tsv(islands: Sequence[GenomicIsland], path: str) -> None:
    """Write full statistical island report as tab-separated values."""
    header = (
        "chrom\tstart\tend\tlength\tpeak_pos\tpeak_z\tmean_bits\tdelta_bits\t"
        "gc_content\tdelta_gc\tp_value\tq_value\tsignificant\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        for isl in islands:
            gc_s = f"{isl.gc_content:.4f}" if np.isfinite(isl.gc_content) else "NA"
            dgc_s = f"{isl.delta_gc:+.4f}" if np.isfinite(isl.delta_gc) else "NA"
            fh.write(
                f"{isl.chrom}\t{isl.start}\t{isl.end}\t{isl.length}\t{isl.peak_pos}\t"
                f"{isl.peak_z:.3f}\t{isl.mean_bits:.4f}\t{isl.delta_bits:+.4f}\t"
                f"{gc_s}\t{dgc_s}\t{isl.p_value:.3e}\t{isl.q_value:.3e}\t"
                f"{str(isl.significant).lower()}\n"
            )


__all__ = [
    "GenomicIsland",
    "WindowTrack",
    "default_max_prototypes",
    "scan_against_reference",
    "scan_self_islands",
    "write_bed",
    "write_bedgraph",
    "write_islands_tsv",
]
