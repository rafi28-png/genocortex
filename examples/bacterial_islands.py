"""
Megabase Pathogenicity-Island & Prophage Discovery on Complete Bacterial Genomes
================================================================================

Real 10.14 Mb comparative genomics validation of GenoCortex v1.0:

* **Reference (commensal core):** *Escherichia coli* str. K-12 substr. MG1655
  (RefSeq ``NC_000913.3``, 4,641,652 bp)
* **Query (enterohemorrhagic pathogen):** *Escherichia coli* O157:H7 str. Sakai
  (RefSeq ``NC_002695.2``, 5,498,450 bp)

Trains a double-strand Fractal Predictive Cortex on both chromosomal strands of
*E. coli* K-12 MG1655 (making the brain invariant to core chromosomal inversions)
and scans the complete 5.50 Mb *E. coli* O157:H7 Sakai chromosome in seconds,
testing recovery against published Sakai pathogenicity islands and Shiga-toxin
prophages (Hayashi et al., *DNA Research* 8:11-22, 2001):

* **SpLE1 (Tellurite resistance & adhesin island)**
* **Sp5 (Stx2 Shiga-toxin-converting prophage)**
* **Sp15 (Stx1 Shiga-toxin-converting prophage)**
* **SpLE3 (efa1/lifA lymphostatin pathogenicity island)**
* **LEE / SpLE4 (Locus of Enterocyte Effacement — Type III secretion system)**

Run:
    python examples/bacterial_islands.py
"""

from __future__ import annotations

import gzip
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import DNA, HAS_FAST_KERNEL, fetch, io_seq, islands
from viz import cognitive_map_html

EMAIL = "rafiur2810@gmail.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "bacteria")
OUT_DIR = os.path.join(ROOT, "output")

GENOMES = {
    "E_coli_K12_MG1655": ("NC_000913.3", "Escherichia coli str. K-12 substr. MG1655"),
    "E_coli_O157_H7_Sakai": ("NC_002695.2", "Escherichia coli O157:H7 str. Sakai"),
}

# Published ground-truth loci on RefSeq NC_002695.2 (Hayashi et al. 2001, DNA Res. 8:11-22)
KNOWN_SAKAI_LOCI = [
    ("Sp5 (Stx2 Shiga-toxin prophage)", 1_245_000, 1_310_000),
    ("SpLE1 (Tellurite/iha PAI)", 1_368_000, 1_456_000),
    ("Sp15 (Stx1 Shiga-toxin prophage)", 2_920_000, 2_975_000),
    ("SpLE3 (efa1/lifA PAI)", 3_850_000, 3_876_000),
    ("LEE / SpLE4 (Type III secretion PAI)", 4_580_000, 4_625_000),
]


def load_or_fetch_genome(tag: str, accession: str) -> io_seq.SeqRecord:
    """Load cached .fna.gz genome or fetch once from NCBI RefSeq."""
    os.makedirs(DATA_DIR, exist_ok=True)
    gz_path = os.path.join(DATA_DIR, f"{tag}.fna.gz")
    if os.path.exists(gz_path):
        return io_seq.read_sequences(gz_path)[0]

    print(f"  Downloading {tag} ({accession}) from NCBI RefSeq ...")
    rec = fetch.ncbi_fasta(accession, db="nucleotide", email=EMAIL, timeout=120.0, retries=4)
    with gzip.open(gz_path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write(f">{accession} {rec.description}\n")
        for i in range(0, len(rec.seq), 70):
            fh.write(rec.seq[i : i + 70] + "\n")
    return io_seq.read_sequences(gz_path)[0]


def overlap_bp(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"GenoCortex v1.0 — Megabase Bacterial Pathogenicity-Island Study (C kernel={HAS_FAST_KERNEL})")

    k12 = load_or_fetch_genome("E_coli_K12_MG1655", GENOMES["E_coli_K12_MG1655"][0])
    sakai = load_or_fetch_genome("E_coli_O157_H7_Sakai", GENOMES["E_coli_O157_H7_Sakai"][0])
    total_bp = len(k12.seq) + len(sakai.seq)
    print(f"  Reference : {k12.id} ({len(k12.seq):,} bp)")
    print(f"  Query     : {sakai.id} ({len(sakai.seq):,} bp)")
    print(f"  Total DNA : {total_bp:,} bp ({total_bp / 1e6:.2f} Mb)")

    t0 = time.perf_counter()
    called_islands, win_tracks, cortex = islands.scan_against_reference(
        ref_records=[k12],
        query_records=[sakai],
        alphabet=DNA,
        order=8,
        double_strand=True,
        window=2000,
        step=500,
        z_threshold=3.5,
        alpha=0.05,
        merge_gap=3500,
        min_length=2000,  # Captures 2-3 kb toxin moron cassettes (stx1AB/stx2AB) as well as multi-kb PAIs
    )
    elapsed = time.perf_counter() - t0
    sig_islands = [isl for isl in called_islands if isl.significant]
    total_island_bp = sum(isl.length for isl in sig_islands)

    print(
        f"\n[Scan Summary]\n"
        f"  Wall time          : {elapsed:.2f} s ({total_bp / elapsed / 1e6:.2f} Mb/s combined)\n"
        f"  Learned prototypes : {cortex.n_prototypes:,}\n"
        f"  FDR islands (q<=5%): {len(sig_islands)} islands spanning {total_island_bp:,} bp "
        f"({100.0 * total_island_bp / len(sakai.seq):.1f}% of Sakai chromosome)"
    )

    bed_path = os.path.join(OUT_DIR, "o157_sakai_islands.bed")
    bg_path = os.path.join(OUT_DIR, "o157_sakai_surprise.bedgraph")
    tsv_path = os.path.join(OUT_DIR, "o157_sakai_islands.tsv")
    islands.write_bed(sig_islands, bed_path, only_significant=True)
    islands.write_bedgraph(win_tracks, bg_path, metric="z_score")
    islands.write_islands_tsv(called_islands, tsv_path)

    print("\n[Top 10 Highest-Surprise Pathogenicity Islands / Prophages]")
    top10 = sorted(sig_islands, key=lambda x: x.peak_z, reverse=True)[:10]
    for rank, isl in enumerate(top10, start=1):
        print(
            f"  {rank:2d}. {isl.chrom}:{isl.start:,}-{isl.end:,} ({isl.length / 1000:.1f} kb) | "
            f"z={isl.peak_z:5.2f} | bits={isl.mean_bits:.3f} ({isl.delta_bits:+.3f}) | "
            f"GC={100*isl.gc_content:4.1f}% ({100*isl.delta_gc:+5.1f}%) | q={isl.q_value:.2e}"
        )

    print("\n[Ground-Truth Sakai Virulence Island Recovery]")
    recovered = 0
    for label, g0, g1 in KNOWN_SAKAI_LOCI:
        hits = [isl for isl in sig_islands if overlap_bp(isl.start, isl.end, g0, g1) > 0]
        if hits:
            recovered += 1
            best = max(hits, key=lambda x: x.peak_z)
            ov = sum(overlap_bp(isl.start, isl.end, g0, g1) for isl in hits)
            print(
                f"  [HIT]  {label:38s} ({g0:,}-{g1:,}) -> "
                f"overlap={ov:,} bp, peak_z={best.peak_z:.2f}, GC={100*best.gc_content:.1f}%, q={best.q_value:.2e}"
            )
        else:
            print(f"  [MISS] {label:38s} ({g0:,}-{g1:,})")

    print(f"\n  Recovered {recovered}/{len(KNOWN_SAKAI_LOCI)} canonical O157:H7 virulence loci.")

    # Render 3-D cognitive map around the LEE pathogenicity island (4.55 - 4.66 Mb)
    lee_slice = DNA.encode(sakai.seq[4_550_000:4_660_000])
    html_path = os.path.join(OUT_DIR, "o157_lee_cognitive_map.html")
    cognitive_map_html(
        cortex,
        lee_slice,
        html_path,
        title="E. coli K-12 Cortex Scanning O157:H7 LEE Pathogenicity Island (4.55-4.66 Mb)",
    )
    print(f"  Exported: {bed_path}\n            {bg_path}\n            {tsv_path}\n            {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
