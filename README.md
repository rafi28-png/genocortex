# GenoCortex v1.0 — Fractal Predictive Cortex (FPC)

**Live Client-Side Browser Workbench & Paper:** https://rafi28-png.github.io/genocortex/

GenoCortex is a training-free, alignment-free, self-organising predictive genomics engine, command-line toolkit (`genocortex`), and live Web Worker browser workbench. A single unsupervised continuous-space model performs **next-symbol prediction**, **megabase pathogenicity-island / horizontal-gene-transfer (HGT) scanning**, **strand-aware sequence & read classification**, and **alignment-free phylogenetic tree inference** — without training labels, multiple sequence alignment, pretrained weights, or fixed $k$-mer tables.

---

## What's New in v1.0

1. **ANSI C PEP 3118 Spatial-Grid Kernel (`fpc/_fast.c`):**
   - Exact adaptive power-of-2 spatial-grid $k$-NN index (`32×32`, `128×128`, `256×256` cells over $[0,1]^2$) accelerating CGR embedding, online prototype growth/drift/pruning, and frozen batch scoring by **~80–120×** over pure Python/NumPy (`~1 Mb/s` single-thread throughput, bit-exact with the NumPy reference to machine epsilon $\le 8.88 \times 10^{-16}$).
   - Pure NumPy/SciPy fallback remains automatic if no C compiler is present.
2. **Double-Strand Symmetry & Local-Entropy-Normalised Excess Surprise:**
   - Reverse-complement-aware DNA/RNA training (`double_strand=True`), strand-aware read classification (`strand_aware=True`), and strand-invariant symmetric cross-entropy distance matrices (`strand_invariant=True`).
   - Sliding-window genomic island caller scores **excess predictive description length** $\text{CE}_{\text{ref}}(W) - H_0(W)$ (empirical Kullback–Leibler divergence above local 0th-order Shannon base entropy) with **Benjamini–Hochberg FDR** ($q$-values), preventing 0th-order AT/GC compositional entropy skew from masking AT-rich pathogenicity islands or GC-rich toxin moron cassettes.
3. **Production Bioinformatics Formats & `genocortex` CLI:**
   - Streaming FASTA & FASTQ (`.fa`, `.fna`, `.fq`, `.fastq`, `.gz`, `stdin`) with vectorized Phred+33 quality masking (`--min-phred`), **BED6** & **UCSC BedGraph** island tracks, **Saitou–Nei Neighbor-Joining** **Newick (`.nwk`)** trees, and **zero-pickle `.fpc.npz`** model archives (`allow_pickle=False`).
4. **Live Client-Side Browser Workbench (`docs/`):**
   - Full ES2022 `Float64Array`/`Int32Array` port of the FPC engine (`docs/fpc-engine.js`, parity-tested against Python/C in `pytest` via Node.js to $< 10^{-12}$) running off-main-thread inside a dedicated **Web Worker** (`docs/fpc-worker.js`).
   - Upload real `.fa`/`.fna`/`.fastq`/`.gz` files (decompressed natively via browser `DecompressionStream`), fetch real accessions live from **NCBI Entrez E-utilities**, inspect interactive 2-D chromosome $z$-score tracks + rotatable 3-D WebGL Cognitive Terrain & 3-D MDS Phylogenetic Constellations, and export real `.bed`, `.bedgraph`, `.tsv`, and `.nwk` files.

---

## Method in Brief

The model streams a biological sequence through a **Chaos Game Representation (CGR)**, mapping every prefix context into the unit square $[0,1]^2$ where sequences sharing an $L$-symbol suffix land within $O(r^L)$ distance. It maintains an online population of predictive prototypes in $[0,1]^2$, allocating new prototypes where its own predictive surprise $-\log_2 P(s_t \mid \mathbf{c}_t)$ spikes and drifting existing prototypes toward observed contexts:

- **Prediction:** Gaussian-kernel-smoothed next-symbol distribution $P(s_t \mid \mathbf{c}_t)$ over local $k$-nearest prototypes blended with a global Dirichlet prior.
- **Megabase HGT / Pathogenicity-Island Scanning:** Excess predictive surprise $\text{CE}_{\text{ref}}(W) - H_0(W)$ (reference-vs-query mode or buffered $K$-fold leave-one-segment-out self-scan mode) with robust median/MAD $z$-scores and Benjamini–Hochberg FDR control.
- **Classification:** Minimum-description-length assignment $\arg\min_c \min\big(\text{CE}_c(q), \text{CE}_c(\overline{q})\big)$ with calibrated posterior probabilities across classes.
- **Phylogenetics:** Strand-invariant symmetric cross-entropy divergence $D(A,B) = \tfrac{1}{2}[\text{CE}_A(B) - \text{CE}_B(B)] + \tfrac{1}{2}[\text{CE}_B(A) - \text{CE}_A(A)] \to$ Saitou–Nei Neighbor-Joining Newick tree & 3-D classical MDS embedding.

---

## Installation

```bash
git clone https://github.com/rafi28-png/genocortex.git
cd genocortex
pip install -e .            # compiles ANSI C kernel (_fast) + installs `genocortex` CLI
pip install -e ".[all]"     # + plotly, scikit-learn, matplotlib, pytest, dendropy
```

Requires Python $\ge 3.10$, `numpy`, and `scipy`. Verify compiled C acceleration:

```bash
genocortex info
```

---

## Command-Line Interface (`genocortex`)

```bash
# 1. Train a double-strand reference cortex (zero-pickle .fpc.npz archive)
genocortex train data/bacteria/E_coli_K12_MG1655.fna.gz \
  -o k12.fpc.npz --order 8 --double-strand

# 2. Scan a complete pathogen chromosome against a reference cortex -> BED6 + BedGraph + TSV + 3D HTML
genocortex scan data/bacteria/E_coli_O157_H7_Sakai.fna.gz \
  --ref k12.fpc.npz \
  --window 2000 --step 500 --z-threshold 3.5 --fdr 0.05 \
  --bed islands.bed --bedgraph surprise.bedgraph -o islands.tsv --html lee_map.html

#    Or run a reference-free K-fold self-island scan on a single genome:
genocortex scan mystery_contig.fna.gz --window 1500 --bed self_islands.bed -o self_islands.tsv

# 3. Train & run a strand-aware FASTA/FASTQ classifier (with Phred quality masking)
genocortex classify --train ref_panel/ --save-model panel.clf.npz --order 7
genocortex classify reads.fastq.gz --model panel.clf.npz --min-phred 20 -o calls.tsv

# 4. Build an alignment-free Neighbor-Joining Newick tree + distance matrix + 3D HTML
genocortex tree genomes/*.fna.gz --order 7 \
  -o tree.nwk --matrix dist.tsv --html phylogeny_3d.html

# 5. Next-symbol probability distribution & sequence continuation
genocortex predict --seq ACGTACGTACGT --context ACGT --generate 30
```

You can also invoke the CLI via `python -m fpc <subcommand>`.

---

## Python API Quick Start

```python
from fpc import DNA, FractalPredictiveCortex, io_seq, islands, phylo, tasks

# 1. Fit a double-strand cortex & persist safely (allow_pickle=False)
rec = io_seq.read_sequences("genome.fna.gz")[0]
model = FractalPredictiveCortex(DNA, order=8, max_prototypes=65536)
track = model.fit(DNA.encode(rec.seq), double_strand=True)
model.save("genome.fpc.npz")

# 2. Reference-vs-query or self genomic island scanning -> BED / BedGraph
called, win_tracks, _ = islands.scan_against_reference(
    ref_records=[k12_rec],
    query_records=[sakai_rec],
    alphabet=DNA,
    order=8,
    window=2000,
    step=500,
)
islands.write_bed(called, "islands.bed", only_significant=True)
islands.write_bedgraph(win_tracks, "surprise.bedgraph")

# 3. Strand-invariant distance matrix -> Neighbor-Joining Newick tree
D = tasks.distance_matrix(seqs, DNA, order=7, strand_invariant=True)
tree = phylo.neighbor_joining(D, names)
print(tree.to_newick())
```

---

## Real-Genome Validation & Benchmarks

All numbers are reproducible from `examples/` and `benchmarks/` (`python run_all.py --real`; NCBI downloads are cached locally after first run). Losses are reported alongside wins.

### Megabase Bacterial Pathogenicity-Island Study (10.14 Mb)

`python examples/bacterial_islands.py` trains a double-strand `order=8` cortex on commensal ***Escherichia coli* K-12 MG1655** (`NC_000913.3`, 4,641,652 bp) and scans enterohemorrhagic ***E. coli* O157:H7 str. Sakai** (`NC_002695.2`, 5,498,578 bp) — **10,140,230 bp total** — in **~15.5 seconds** on a single CPU thread (`67,146` learned prototypes), recovering **5/5** canonical Sakai pathogenicity islands and Shiga-toxin prophage loci at Benjamini–Hochberg FDR $q < 0.01$ with zero alignment and zero supervision:

| Canonical Sakai Virulence Locus (`NC_002695.2`) | Coordinates (bp) | Peak $z$ | Island GC | BH FDR $q$-value | Status |
|---|---|---|---|---|---|
| **LEE / SpLE4** (Type III secretion PAI) | `4,580,000–4,625,000` | `6.77` | `36.6%` (`-13.9%`) | `2.61e-09` | **Recovered** (`27.0 kb` overlap) |
| **SpLE3** (`efa1`/`lifA` lymphostatin PAI) | `3,850,000–3,876,000` | `6.22` | `37.7%` (`-12.8%`) | `5.41e-08` | **Recovered** (`9.5 kb` overlap) |
| **SpLE1** (Tellurite resistance / `iha` PAI) | `1,368,000–1,456,000` | `6.10` | `36.5%` (`-14.0%`) | `9.74e-08` | **Recovered** (`17.5 kb` overlap) |
| **Sp5** (`Stx2` Shiga-toxin converting prophage) | `1,245,000–1,310,000` | `5.92` | `56.2%` (`+5.7%`) | `2.27e-07` | **Recovered** (`5.5 kb` `stx2` moron) |
| **Sp15** (`Stx1` Shiga-toxin converting prophage) | `2,920,000–2,975,000` | `3.52` | `41.1%` (`-9.4%`) | `8.34e-03` | **Recovered** (`2.0 kb` `stx1` moron) |

Because *E. coli* K-12 MG1655 already carries cryptic lambdoid prophages (`Rac`, `DLP12`, `Qin`, `CP4-57`), GenoCortex ignores conserved lambdoid phage structural backbones in `Sp5`/`Sp15` and surgically pinpoints both the **GC-rich** `stx2` moron cassette (`56.2% GC`) and the **AT-rich** `LEE`/`SpLE1`/`SpLE3`/`stx1` islands (`36.5%–41.1% GC`).

### Comparative Multi-Task Benchmark Summary

| Task | FPC | Baselines | Summary |
|---|---|---|---|
|Megabase HGT / PAI scan (10.14 Mb *E. coli* K-12 $\to$ O157:H7) | **5/5 canonical loci** in **15.5 s** | — | Recovers both AT-rich PAIs (`LEE`) and GC-rich `stx2` morons at $q < 0.01$ |
| Phylogeny, 25 synthetic trees (Pearson $r$) | `0.93` | FFP `0.93`, NCCD `0.93`, NRC `0.92`, Mash `0.91`, $k$-mer cosine `0.78` | Statistically tied with FFP/Mash/NRC/NCCD (Wilcoxon $p \ge 0.24$); beats $k$-mer cosine ($p < 10^{-6}$) |
| Phylogeny, 17 mammalian mitogenomes (norm. Robinson–Foulds, lower better) | `0.43` | NRC `0.36`, NCCD/FFP/Mash `0.43`, $k$-mer cosine `0.71` | Ties FFP, Mash, and NCCD; slightly behind NRC |
| Short-read classification, 150 bp | `100%` | $k$-mer+1NN `97%`, $k$-mer+logistic `94%` | Strand-aware MDL classification remains robust on short reads |
| Compression (bits/symbol) | Beats `gzip`; near best Markov | Hand-tuned Markov-$k$ | Competitive without choosing fixed $k$ |
| Foreign-insert detection (ROC-AUC) | `~1.0` | 8-mer novelty `~1.0` | Parity; adds calibrated per-position surprise & FDR $q$-values |
| Real viral families, deep divergence (clustering purity) | `0.82` | FFP `0.93` | FFP is more accurate at deep viral divergence |

---

## Repository Layout

```text
fpc/                Core Python + ANSI C engine
  _fast.c           PEP 3118 C11 adaptive spatial-grid kernel (CGR + online cortex + batch score)
  alphabet.py       O(N) LUT encoding + reverse-complement support (DNA / RNA / PROTEIN / custom)
  cgr.py            Chaos Game Representation embedding
  cortex.py         FractalPredictiveCortex + zero-pickle .fpc.npz persistence
  surprise.py       SurpriseTrack, Benjamini-Hochberg FDR, anomaly detection
  islands.py        Megabase HGT / pathogenicity-island scanner + BED6 / BedGraph / TSV writers
  phylo.py          Saitou-Nei Neighbor-Joining tree builder + Newick (.nwk) exporter
  io_seq.py         Streaming FASTA & FASTQ (.gz, stdin) reader with Phred quality masking
  tasks.py          Predict / detect / strand-aware FPCClassifier / strand-invariant distance_matrix
  cli.py            `genocortex` CLI (`train`, `scan`, `classify`, `tree`, `predict`, `info`)
  datasets.py       Reproducible synthetic benchmarks with ground truth
  fetch.py          Optional NCBI Entrez downloader with local disk caching
docs/               Live GitHub Pages browser workbench + research manuscript
  index.html        Interactive 4-tab browser workbench + manuscript
  fpc-engine.js     Universal ES2022 Float64Array FPC engine (parity-tested vs Python/C)
  fpc-worker.js     Off-main-thread Web Worker + live NCBI E-utilities fetcher
  workbench.js      2D Canvas chromosome track + lazy 3D WebGL Plotly surfaces & constellations
viz/                Standalone offline 3-D HTML cognitive maps & phylogeny constellations
benchmarks/         Baselines (Markov, gzip, k-mer, Mash, FFP, NRC, NCCD) & statistical tests
examples/           Reproducible real-genome studies (coronaviruses, 28-virus panel, proteins, 10.14 Mb E. coli)
tests/              28 unit, C/NumPy/JS bit-parity, bioinformatics I/O, NJ, island, and CLI tests
paper/              Manuscript (paper.md, cross-platform build_pdf.py, cover_letter.md)
```

Run the test suite and reproduction pipeline:

```bash
pytest tests/ -v
python run_all.py           # offline tests + synthetic benchmarks + 3D visuals
python run_all.py --real    # + real viral, protein, mitogenome, and 10.14 Mb bacterial studies
```

---

## Limitations

- Not state-of-the-art on every isolated task: matched Markov/PPM/CTW models achieve lower bitrates on stationary sources, and FFP achieves higher clustering purity at deep viral family divergence.
- Effective context horizon is bounded by `order` and CGR's exponential contraction ($r = 0.5$ for DNA/RNA), so distal long-range dependencies (e.g., distant enhancer-promoter loops) are not modelled.
- Requires sufficient training sequence to populate prototype space; very short single-domain protein families (~150 aa) favour direct alignment or $k$-mer methods.
- Symmetric cross-entropy sequence divergence builds on established compression-distance principles (see NRC/NCCD discussion in `paper/paper.md`).

## License

MIT (see [`LICENSE`](LICENSE)).
