"""
Production verification suite for GenoCortex v1.0.
==================================================

Tests:
1. ANSI C PEP 3118 spatial-grid kernel vs reference NumPy bit-exact parity (< 1e-12)
   across DNA, RNA, and Protein (including wildcards and prototype pruning).
2. Python/C vs client-side JavaScript (docs/fpc-engine.js) bit-exact parity (< 1e-12)
   on CGR trajectories, online learning, frozen scoring, distance matrices, and Newick trees.
3. Reverse-complement algebra & exact strand invariance of symmetric KL distances.
4. Strand-aware (+/-) 150 bp read classification & zero-pickle .fpc.npz persistence.
5. FASTQ + .fastq.gz streaming (including '@' Q=31 quality lines) & Phred masking.
6. Saitou-Nei Neighbour-Joining exact additive tree recovery & DendroPy RF=0 check.
7. Reference-guided & reference-free K-fold genomic island scanners + BED/BedGraph export.
8. End-to-end CLI subcommands (train, scan, classify, tree, predict, info, error codes).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import pytest

from fpc import (
    DNA,
    RNA,
    PROTEIN,
    FractalPredictiveCortex,
    HAS_FAST_KERNEL,
    benjamini_hochberg,
    datasets,
    io_seq,
    islands,
    phylo,
    tasks,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    nvm_hits = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/node")))
    return nvm_hits[-1] if nvm_hits else None


# --------------------------------------------------------------------- #
#  1. C vs NumPy bit-exact parity                                       #
# --------------------------------------------------------------------- #
def test_fast_c_kernel_is_compiled():
    assert HAS_FAST_KERNEL is True, "Expected compiled fpc._fast C extension to be active"


@pytest.mark.parametrize("alphabet,order", [(DNA, 9), (RNA, 8), (PROTEIN, 3)])
def test_c_vs_numpy_bit_exact_parity(alphabet, order):
    rng = np.random.default_rng(101)
    syms = list(alphabet.symbols) + [alphabet.wildcard]
    probs = [0.95 / alphabet.size] * alphabet.size + [0.05]
    train_seq = "".join(rng.choice(syms, size=9000, p=probs))
    test_seq = "".join(rng.choice(syms, size=5000, p=probs))
    ids_tr = alphabet.encode(train_seq)
    ids_te = alphabet.encode(test_seq)

    c_fast = FractalPredictiveCortex(
        alphabet, order=order, max_prototypes=320, prune_interval=1200, use_fast=True
    )
    c_slow = FractalPredictiveCortex(
        alphabet, order=order, max_prototypes=320, prune_interval=1200, use_fast=False
    )

    tr_fast = c_fast.fit(ids_tr)
    tr_slow = c_slow.fit(ids_tr)
    assert c_fast.n_prototypes == c_slow.n_prototypes
    assert np.array_equal(np.isnan(tr_fast.bits), np.isnan(tr_slow.bits))
    assert np.nanmax(np.abs(tr_fast.bits - tr_slow.bits)) < 1e-12

    sc_fast = c_fast.score(ids_te)
    sc_slow = c_slow.score(ids_te)
    assert np.array_equal(np.isnan(sc_fast.bits), np.isnan(sc_slow.bits))
    assert np.nanmax(np.abs(sc_fast.bits - sc_slow.bits)) < 1e-12


# --------------------------------------------------------------------- #
#  2. Python/C vs JavaScript (docs/fpc-engine.js) parity                #
# --------------------------------------------------------------------- #
def test_python_vs_javascript_engine_parity():
    node_bin = _find_node()
    if not node_bin:
        pytest.skip("Node.js runtime not installed")

    rng = np.random.default_rng(2026)
    seq1 = "".join(rng.choice(list("ACGTN"), size=4500, p=[0.28, 0.22, 0.22, 0.25, 0.03]))
    seq2 = "".join(rng.choice(list("ACGTN"), size=2500, p=[0.20, 0.30, 0.30, 0.18, 0.02]))
    taxa = [datasets.markov_sequence(1000, DNA, order=2, concentration=0.2, seed=10 + i) for i in range(5)]
    names = [f"taxon_{i}" for i in range(5)]

    cx = FractalPredictiveCortex(DNA, order=8, max_prototypes=280, prune_interval=900)
    fit_py = cx.fit(DNA.encode(seq1)).bits
    sc_py = cx.score(DNA.encode(seq2)).bits
    D_py = tasks.distance_matrix(taxa, DNA, order=5, strand_invariant=True)
    nwk_py = phylo.newick_from_distance_matrix(D_py, names)

    js_script = """
    import * as fpc from "./docs/fpc-engine.js";
    import fs from "node:fs";
    const inp = JSON.parse(fs.readFileSync(0, "utf8"));
    const cx = new fpc.FractalPredictiveCortex(fpc.DNA, {
      order: 8, maxPrototypes: 280, pruneInterval: 900
    });
    const fitJs = Array.from(cx.fit(fpc.DNA.encode(inp.seq1)).bits);
    const scJs = Array.from(cx.score(fpc.DNA.encode(inp.seq2)).bits);
    const D = fpc.distanceMatrix(inp.taxa, { alphabet: fpc.DNA, order: 5, strandInvariant: true });
    const { newick } = fpc.neighborJoining(D, inp.names);
    process.stdout.write(JSON.stringify({
      nProto: cx.nPrototypes,
      fitJs,
      scJs,
      D: D.map(r => Array.from(r)),
      newick
    }));
    """
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-e", js_script],
        cwd=REPO_ROOT,
        input=json.dumps({"seq1": seq1, "seq2": seq2, "taxa": taxa, "names": names}),
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    fit_js = np.array([np.nan if v is None else v for v in payload["fitJs"]], dtype=np.float64)
    sc_js = np.array([np.nan if v is None else v for v in payload["scJs"]], dtype=np.float64)
    D_js = np.array(payload["D"], dtype=np.float64)

    assert payload["nProto"] == cx.n_prototypes
    assert np.nanmax(np.abs(fit_py - fit_js)) < 1e-12
    assert np.nanmax(np.abs(sc_py - sc_js)) < 1e-12
    assert np.max(np.abs(D_py - D_js)) < 1e-12

    import dendropy
    tns = dendropy.TaxonNamespace()
    t_py = dendropy.Tree.get(data=nwk_py, schema="newick", taxon_namespace=tns, rooting="force-unrooted")
    t_js = dendropy.Tree.get(data=payload["newick"], schema="newick", taxon_namespace=tns, rooting="force-unrooted")
    assert dendropy.calculate.treecompare.symmetric_difference(t_py, t_js) == 0
    assert dendropy.calculate.treecompare.euclidean_distance(t_py, t_js) < 1e-5


# --------------------------------------------------------------------- #
#  3. Double-strand algebra & strand-invariant distances                #
# --------------------------------------------------------------------- #
def test_reverse_complement_and_strand_invariant_distance():
    assert DNA.reverse_complement("AACCGGTTN") == "NAACCGGTT"
    assert RNA.reverse_complement("AACCGUUN") == "NAACGGUU"
    with pytest.raises(ValueError):
        PROTEIN.reverse_complement("MKWV")

    seqs = [datasets.markov_sequence(1800, DNA, order=2, concentration=0.2, seed=30 + i) for i in range(4)]
    flipped = [seqs[0], DNA.reverse_complement(seqs[1]), seqs[2], DNA.reverse_complement(seqs[3])]

    D1 = tasks.distance_matrix(seqs, DNA, order=6, strand_invariant=True)
    D2 = tasks.distance_matrix(flipped, DNA, order=6, strand_invariant=True)
    assert np.allclose(D1, D2, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------- #
#  4. Strand-aware short-read classification & zero-pickle persistence  #
# --------------------------------------------------------------------- #
def test_strand_aware_classification_and_zero_pickle_save_load(tmp_path):
    genomes = {
        "Species_A": datasets.markov_sequence(12000, DNA, order=3, concentration=0.12, seed=11),
        "Species_B": datasets.markov_sequence(12000, DNA, order=3, concentration=0.12, seed=22),
        "Species_C": datasets.markov_sequence(12000, DNA, order=3, concentration=0.12, seed=33),
    }
    clf = tasks.FPCClassifier(DNA, order=7, strand_aware=True, double_strand_train=False)
    clf.fit(list(genomes.values()), list(genomes.keys()))

    rng = np.random.default_rng(99)
    queries = []
    truth = []
    for label, gseq in genomes.items():
        for r_idx in range(6):
            start = int(rng.integers(500, len(gseq) - 300))
            sub = gseq[start : start + 160]
            strand = "-" if (r_idx % 2 == 1) else "+"
            qseq = DNA.reverse_complement(sub) if strand == "-" else sub
            queries.append(io_seq.SeqRecord(id=f"{label}_r{r_idx}", description="", seq=qseq))
            truth.append((label, strand))

    calls = clf.classify_records(queries)
    assert len(calls) == len(truth)
    for call, (exp_cls, exp_strand) in zip(calls, truth):
        assert call.predicted_class == exp_cls
        assert call.strand == exp_strand
        assert call.posterior > 0.34
        assert call.delta_bits > 0.0

    # Save and reload with allow_pickle=False
    model_path = str(tmp_path / "panel.fpc.npz")
    clf.save(model_path)
    with np.load(model_path, allow_pickle=False) as raw:
        assert "meta_json" in raw.files

    reloaded = tasks.FPCClassifier.load(model_path)
    calls_back = reloaded.classify_records(queries)
    for c1, c2 in zip(calls, calls_back):
        assert c1.predicted_class == c2.predicted_class
        assert c1.strand == c2.strand
        assert c1.bits_per_base == pytest.approx(c2.bits_per_base, abs=1e-12)


def test_single_cortex_zero_pickle_roundtrip(tmp_path):
    seq = datasets.markov_sequence(6000, DNA, order=2, concentration=0.15, seed=77)
    cx = FractalPredictiveCortex(DNA, order=7)
    cx.fit(DNA.encode(seq[:4000]), double_strand=True)
    path = str(tmp_path / "brain.fpc.npz")
    cx.save(path)

    cx2 = FractalPredictiveCortex.load(path)
    s1 = cx.score(DNA.encode(seq[4000:])).bits
    s2 = cx2.score(DNA.encode(seq[4000:])).bits
    assert np.allclose(s1, s2, rtol=1e-13, atol=1e-13, equal_nan=True)


# --------------------------------------------------------------------- #
#  5. FASTQ.gz streaming & Phred quality masking                        #
# --------------------------------------------------------------------- #
def test_fastq_gz_streaming_and_phred_masking(tmp_path):
    fq_gz = tmp_path / "reads.fastq.gz"
    # Note: '@' is Phred+33 Q=31, '#' is Phred+33 Q=2
    content = (
        "@read_alpha lane=1\n"
        "ACGTACGTACGT\n"
        "+\n"
        "@@@@####IIII\n"
        "@read_beta wrapped\n"
        "GGGGCCCC\n"
        "AAAATTTT\n"
        "+\n"
        "IIIIIIII\n"
        "####@@@@\n"
    )
    with gzip.open(fq_gz, "wt", encoding="utf-8") as fh:
        fh.write(content)

    recs = io_seq.read_sequences(str(fq_gz), min_phred=20, wildcard="N")
    assert len(recs) == 2
    assert recs[0].id == "read_alpha"
    assert recs[0].seq == "ACGTNNNNACGT"
    assert recs[1].id == "read_beta"
    assert recs[1].seq == "GGGGCCCCNNNNTTTT"


# --------------------------------------------------------------------- #
#  6. Neighbour-Joining exact additive tree & Benjamini-Hochberg FDR    #
# --------------------------------------------------------------------- #
def test_neighbor_joining_recovers_exact_additive_tree():
    import dendropy

    nwk_true = "((A:0.12,B:0.23):0.09,(C:0.31,D:0.17):0.14,E:0.28);"
    tns = dendropy.TaxonNamespace()
    t_true = dendropy.Tree.get(data=nwk_true, schema="newick", taxon_namespace=tns, rooting="force-unrooted")
    pdm = t_true.phylogenetic_distance_matrix()
    names = ["A", "B", "C", "D", "E"]
    tax_map = {tx.label: tx for tx in tns}
    D = np.zeros((5, 5), dtype=np.float64)
    for i in range(5):
        for j in range(5):
            D[i, j] = pdm(tax_map[names[i]], tax_map[names[j]])

    nj_tree = phylo.neighbor_joining(D, names)
    D_pat = phylo.patristic_distance_matrix(nj_tree, names)
    assert np.allclose(D_pat, D, atol=1e-12)

    t_est = dendropy.Tree.get(
        data=nj_tree.to_newick(), schema="newick", taxon_namespace=tns, rooting="force-unrooted"
    )
    rf = dendropy.calculate.treecompare.symmetric_difference(t_true, t_est)
    assert rf == 0


def test_benjamini_hochberg_monotonicity():
    p = np.array([0.0001, 0.012, 0.004, 0.25])
    q = benjamini_hochberg(p, total_tests=20)
    assert q[0] == pytest.approx(0.0001 * 20 / 1)
    assert q[2] == pytest.approx(0.004 * 20 / 2)
    assert q[1] == pytest.approx(0.012 * 20 / 3)
    assert (q >= p).all() and (q <= 1.0).all()


# --------------------------------------------------------------------- #
#  7. Genomic island discovery (reference-guided & K-fold self-scan)    #
# --------------------------------------------------------------------- #
def test_island_scanners_and_bed_bedgraph_exports(tmp_path):
    host_ref = ("AATTGCATATGCGTTATTAAGCAT" * 1200)          # 28.8 kb AT-rich structured host
    core_q = ("AATTGCATATGCGTTATTAAGCAT" * 1000)            # 24.0 kb host backbone
    foreign_1 = ("GCGCACGGCCGGCGCAGGCGCGCC" * 90)           # 2,160 bp GC-rich island
    foreign_2 = ("CGCGTACGGCCGGCGCGCGCACGG" * 80)           # 1,920 bp GC-rich island

    query_seq = core_q[:7000] + foreign_1 + core_q[7000:17000] + foreign_2 + core_q[17000:]
    isl1_range = (7000, 7000 + len(foreign_1))
    isl2_range = (17000 + len(foreign_1), 17000 + len(foreign_1) + len(foreign_2))

    # A. Reference-guided scan
    ref_isl, tracks, _ = islands.scan_against_reference(
        [("host", host_ref)],
        [("chr_q", query_seq)],
        alphabet=DNA,
        order=6,
        window=500,
        step=125,
        z_threshold=3.0,
        merge_gap=1000,
    )
    sig_ref = [x for x in ref_isl if x.significant]
    assert any(min(x.end, isl1_range[1]) - max(x.start, isl1_range[0]) > 1000 for x in sig_ref)
    assert any(min(x.end, isl2_range[1]) - max(x.start, isl2_range[0]) > 1000 for x in sig_ref)

    # B. Reference-free buffered K-fold self-scan
    self_isl, _ = islands.scan_self_islands(
        [("chr_q", query_seq)],
        alphabet=DNA,
        order=6,
        n_folds=6,
        guard_bp=2500,
        window=500,
        step=125,
        z_threshold=2.8,
        merge_gap=1000,
    )
    assert any(min(x.end, isl1_range[1]) - max(x.start, isl1_range[0]) > 1000 for x in self_isl)
    assert any(min(x.end, isl2_range[1]) - max(x.start, isl2_range[0]) > 1000 for x in self_isl)

    bed_file = tmp_path / "islands.bed"
    bg_file = tmp_path / "surprise.bedgraph"
    tsv_file = tmp_path / "islands.tsv"
    assert islands.write_bed(sig_ref, str(bed_file)) >= 2
    assert islands.write_bedgraph(tracks, str(bg_file)) > 10
    islands.write_islands_tsv(ref_isl, str(tsv_file))

    bed_lines = bed_file.read_text(encoding="utf-8").strip().splitlines()
    cols = bed_lines[0].split("\t")
    assert len(cols) == 6 and cols[0] == "chr_q" and int(cols[1]) < int(cols[2])


# --------------------------------------------------------------------- #
#  8. End-to-end CLI subcommand tests                                   #
# --------------------------------------------------------------------- #
def test_cli_end_to_end_subcommands(tmp_path):
    sp_a = tmp_path / "sp_alpha.fna"
    sp_b = tmp_path / "sp_beta.fna"
    sp_c = tmp_path / "sp_gamma.fna"
    seq_a = datasets.markov_sequence(6000, DNA, order=2, concentration=0.14, seed=1)
    seq_b = datasets.markov_sequence(6000, DNA, order=2, concentration=0.14, seed=2)
    seq_c = datasets.markov_sequence(6000, DNA, order=2, concentration=0.14, seed=3)
    io_seq.write_fasta(str(sp_a), [("sp_alpha", seq_a)])
    io_seq.write_fasta(str(sp_b), [("sp_beta", seq_b)])
    io_seq.write_fasta(str(sp_c), [("sp_gamma", seq_c)])

    def run_cli(*args, expect_ok=True):
        res = subprocess.run(
            [sys.executable, "-m", "fpc", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        if expect_ok:
            assert res.returncode == 0, f"CLI failed: {res.stderr}"
        return res

    # 1. info
    info_out = json.loads(run_cli("info").stdout)
    assert info_out["fast_c_kernel"] is True

    # 2. train --by-file + classify (with reverse-complement read)
    panel_npz = tmp_path / "panel.fpc.npz"
    run_cli("train", "--by-file", str(sp_a), str(sp_b), str(sp_c), "-k", "6", "-o", str(panel_npz), "-q")

    reads_fq = tmp_path / "queries.fastq"
    rc_b = DNA.reverse_complement(seq_b[1000:1200])
    reads_fq.write_text(f"@q_beta_rc\n{rc_b}\n+\n{'I' * len(rc_b)}\n", encoding="utf-8")
    calls_tsv = tmp_path / "calls.tsv"
    run_cli("classify", str(reads_fq), "--model", str(panel_npz), "-o", str(calls_tsv), "-q")
    rows = calls_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    fields = rows[1].split("\t")
    assert fields[1] == "sp_beta" and fields[2] == "-"

    # 3. tree
    tree_nwk = tmp_path / "tree.nwk"
    dist_tsv = tmp_path / "dist.tsv"
    run_cli("tree", str(sp_a), str(sp_b), str(sp_c), "-k", "5", "-o", str(tree_nwk), "--matrix", str(dist_tsv), "-q")
    assert tree_nwk.read_text(encoding="utf-8").strip().endswith(";")
    assert len(dist_tsv.read_text(encoding="utf-8").strip().splitlines()) == 4

    # 4. scan
    chimera_fa = tmp_path / "chimera.fna"
    chimera_seq = seq_a[:2500] + ("GCGCACGGCCGGCGCA" * 70) + seq_a[2500:]
    io_seq.write_fasta(str(chimera_fa), [("chimera", chimera_seq)])
    out_bed = tmp_path / "scan.bed"
    run_cli("scan", str(chimera_fa), "--ref", str(sp_a), "-k", "6", "-w", "400", "-s", "100", "-z", "2.8", "--bed", str(out_bed), "-q")
    assert len(out_bed.read_text(encoding="utf-8").strip().splitlines()) >= 1

    # 5. predict
    pred_out = run_cli("predict", "--train", str(sp_a), "--prefix", "ACGTACGT", "-n", "30").stdout.strip()
    assert len(pred_out) == 38 and pred_out.startswith("ACGTACGT")

    # 6. non-zero exit code on missing file (zero silent mock fallback)
    bad = run_cli("scan", str(tmp_path / "does_not_exist.fa"), expect_ok=False)
    assert bad.returncode != 0


def test_readme_cli_aliases_and_unclassified_low_phred_reads(tmp_path):
    ref_dir = tmp_path / "ref_panel"
    ref_dir.mkdir()
    seq_a = datasets.markov_sequence(5000, DNA, order=2, concentration=0.14, seed=11)
    seq_b = datasets.markov_sequence(5000, DNA, order=2, concentration=0.14, seed=12)
    fa_a = ref_dir / "species_a.fna"
    fa_b = ref_dir / "species_b.fna"
    io_seq.write_fasta(str(fa_a), [("species_a", seq_a)])
    io_seq.write_fasta(str(fa_b), [("species_b", seq_b)])

    def run_cli(*args):
        res = subprocess.run(
            [sys.executable, "-m", "fpc", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        assert res.returncode == 0, f"CLI failed ({args}): {res.stderr}"
        return res

    # README pattern 1 & 2: train --double-strand -> scan --ref <.fpc.npz> --fdr 0.05 -o <tsv>
    k12_npz = tmp_path / "k12.fpc.npz"
    run_cli("train", str(fa_a), "-o", str(k12_npz), "--order", "6", "--double-strand", "-q")

    chimera_fa = tmp_path / "chimera.fna"
    io_seq.write_fasta(str(chimera_fa), [("chimera", seq_a[:2000] + ("GCGCACGGCCGGCGCA" * 60) + seq_a[2000:])])
    isl_tsv = tmp_path / "islands.tsv"
    run_cli(
        "scan", str(chimera_fa), "--ref", str(k12_npz),
        "--window", "400", "--step", "100", "--z-threshold", "2.8", "--fdr", "0.05",
        "-o", str(isl_tsv), "-q",
    )
    assert len(isl_tsv.read_text(encoding="utf-8").strip().splitlines()) >= 2

    # README pattern 3: classify --train <dir> --save-model <npz>, plus all-masked UNCLASSIFIED FASTQ read
    clf_npz = tmp_path / "panel.clf.npz"
    run_cli("classify", "--train", str(ref_dir), "--by-file", "--save-model", str(clf_npz), "--order", "6", "-q")

    fq = tmp_path / "reads.fastq"
    good_rc = DNA.reverse_complement(seq_b[500:680])
    good_fwd = seq_a[900:1080]
    lowq_read = seq_a[500:680]
    fq.write_text(
        f"@good_rc\n{good_rc}\n+\n{'I' * len(good_rc)}\n"
        f"@good_fwd\n{good_fwd}\n+\n{'I' * len(good_fwd)}\n"
        f"@all_masked\n{lowq_read}\n+\n{'#' * len(lowq_read)}\n",
        encoding="utf-8",
    )
    calls_out = tmp_path / "calls.tsv"
    run_cli("classify", str(fq), "--model", str(clf_npz), "--min-phred", "20", "-o", str(calls_out), "-q")
    lines = calls_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    r_rc = lines[1].split("\t")
    r_fwd = lines[2].split("\t")
    r_mask = lines[3].split("\t")
    assert r_rc[1] == "species_b" and r_rc[2] == "-"
    assert r_fwd[1] == "species_a" and r_fwd[2] == "+"
    assert r_mask[1] == "UNCLASSIFIED" and r_mask[2] == "." and r_mask[3] == "NA"

    # README pattern 5: predict --seq ... --context ... --generate ...
    gen = run_cli("predict", "--seq", "ACGTACGT" * 40, "--context", "ACGT", "--generate", "20").stdout.strip()
    assert len(gen) == 24 and gen.startswith("ACGT")


def test_web_worker_all_tabs_end_to_end():
    node_bin = _find_node()
    if not node_bin:
        pytest.skip("Node.js runtime not installed")

    worker_test_js = """
    const msgs = [];
    globalThis.self = { postMessage: (m) => msgs.push(m), onmessage: null };
    await import("./docs/fpc-worker.js");

    self.onmessage({ data: { id: 1, task: "scan", payload: {
      alphabet: "DNA", order: 6, window: 200, step: 50, zThreshold: 2.8,
      refText: ">ref\\n" + "AATTGCATATGCGTTA".repeat(350),
      queryText: ">qry\\n" + "AATTGCATATGCGTTA".repeat(180) + "GCGCACGGCCGGCGCA".repeat(35) + "AATTGCATATGCGTTA".repeat(180)
    }}});
    self.onmessage({ data: { id: 2, task: "scan", payload: {
      alphabet: "DNA", order: 6, window: 200, step: 50, zThreshold: 2.8,
      refText: "",
      queryText: ">qry_self\\n" + "AATTGCATATGCGTTA".repeat(260) + "GCGCACGGCCGGCGCA".repeat(35) + "AATTGCATATGCGTTA".repeat(260)
    }}});
    self.onmessage({ data: { id: 3, task: "tree", payload: {
      alphabet: "DNA", order: 5, strandInvariant: true,
      fastaText: ">t1\\n" + "ACGTTGCA".repeat(250) + "\\n>t2\\n" + "ACGTTGCG".repeat(250) + "\\n>t3\\n" + "CGGATCCA".repeat(250)
    }}});
    self.onmessage({ data: { id: 4, task: "classify", payload: {
      alphabet: "DNA", order: 5, minPhred: 20, strandAware: true,
      refText: ">clsA\\n" + "ACCTGAGT".repeat(250) + "\\n>clsB\\n" + "CGGAAGCT".repeat(250),
      queryText: "@r1_rc\\n" + "ACTCAGGT".repeat(18) + "\\n+\\n" + "I".repeat(144) +
                 "\\n@r2_fwd\\n" + "ACCTGAGT".repeat(18) + "\\n+\\n" + "I".repeat(144) +
                 "\\n@r3_masked\\n" + "ACCTGAGT".repeat(18) + "\\n+\\n" + "#".repeat(144)
    }}});
    self.onmessage({ data: { id: 5, task: "predict", payload: {
      alphabet: "DNA", order: 6, trainText: ">tr\\n" + "ACCTGAGT".repeat(250), prefix: "ACCTGAGT", length: 24, sample: false
    }}});

    const out = msgs.filter(m => m.type !== "progress");
    process.stdout.write(JSON.stringify(out));
    """
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-e", worker_test_js],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    results = json.loads(proc.stdout)
    assert len(results) == 5
    assert all(r["type"] == "result" for r in results)
    assert len(results[0]["result"]["islands"]) >= 1
    assert len(results[1]["result"]["islands"]) >= 1
    assert results[2]["result"]["newick"].endswith(";")
    calls = results[3]["result"]["calls"]
    assert calls[0]["predictedClass"] == "clsA" and calls[0]["strand"] == "-"
    assert calls[1]["predictedClass"] == "clsA" and calls[1]["strand"] == "+"
    assert calls[2]["predictedClass"] == "UNCLASSIFIED" and calls[2]["strand"] == "."
    assert "NA" in results[3]["result"]["downloads"]["tsv"]
    assert len(results[4]["result"]["continuation"]) == 24


def test_prototype_pruning_evicts_stale_regime_and_preserves_cap():
    with pytest.raises(ValueError):
        FractalPredictiveCortex(DNA, max_prototypes=0)

    # Regime A visits all 64 distinct 6-mer {A, T} contexts (> max_prototypes=16),
    # saturating memory completely before Regime B (periodic GC motif) begins.
    rng = np.random.default_rng(42)
    regime_a = "".join(rng.choice(["A", "T"], size=2400))
    regime_b = "GGCCGCGG" * 400
    stream_ids = DNA.encode(regime_a + regime_b)
    probe_b = DNA.encode("GGCCGCGG" * 80)

    c_fast = FractalPredictiveCortex(
        DNA, order=6, max_prototypes=16, prune_interval=200, usage_decay=0.985, use_fast=True
    )
    c_slow = FractalPredictiveCortex(
        DNA, order=6, max_prototypes=16, prune_interval=200, usage_decay=0.985, use_fast=False
    )
    # Verify Regime A genuinely saturates max_prototypes=16 first
    c_fast.fit(DNA.encode(regime_a))
    c_slow.fit(DNA.encode(regime_a))
    assert c_fast.n_prototypes == 16
    assert c_slow.n_prototypes == 16

    # Reset and stream A -> B continuously
    c_fast = FractalPredictiveCortex(
        DNA, order=6, max_prototypes=16, prune_interval=200, usage_decay=0.985, use_fast=True
    )
    c_slow = FractalPredictiveCortex(
        DNA, order=6, max_prototypes=16, prune_interval=200, usage_decay=0.985, use_fast=False
    )
    tr_fast = c_fast.fit(stream_ids)
    tr_slow = c_slow.fit(stream_ids)

    assert c_fast.n_prototypes == 16
    assert c_slow.n_prototypes == 16
    assert np.nanmax(np.abs(tr_fast.bits - tr_slow.bits)) < 1e-12

    # Active pruning must have evicted stale AT prototypes in favour of GC prototypes
    assert c_fast.score(probe_b).mean_bits < 0.35


def test_cognitive_terrain_context_alignment_and_protein_scan(tmp_path):
    from viz.viz3d import _surprise_grid, cognitive_map_html

    # Train on pure 'G' (CGR corner (1, 1)), then score terminal shock "GGGGGGGGA":
    # The ~9-bit 'A' shock occurs AT context 'GGGGGGGG' near (1, 1) (top-right),
    # NOT shifted post-step to (0.5, 0.5) (centre).
    c_py = FractalPredictiveCortex(DNA, order=6, max_prototypes=64)
    c_py.fit(DNA.encode("G" * 400))
    q_ids = DNA.encode("GGGGGGGGA")
    out_html = tmp_path / "cog_map.html"
    cognitive_map_html(c_py, q_ids, str(out_html), grid=16)
    assert out_html.exists() and "surprise" in out_html.read_text(encoding="utf-8")
    z_py, mask_py = _surprise_grid(c_py.embedding.context_points(q_ids), c_py.score(q_ids).bits, 16)
    assert not mask_py[7, 7]
    assert z_py[15, 15] > 1.5 and z_py[15, 15] > z_py[8, 8]

    node_bin = _find_node()
    if not node_bin:
        pytest.skip("Node.js runtime not installed")

    js_check = """
    import * as fpc from "./docs/fpc-engine.js";
    const cx = new fpc.FractalPredictiveCortex(fpc.DNA, { order: 6, maxPrototypes: 64 });
    cx.fit(fpc.DNA.encode("G".repeat(400)));
    const queryIds = fpc.DNA.encode("GGGGGGGGA");
    const terrain = fpc.buildCognitiveTerrain(cx, queryIds, 16);
    const topRight = terrain.zRows[15][15];
    const mid = terrain.zRows[8][8];
    const aCorner = terrain.zRows[0][0];

    // Also verify defaultMaxPrototypes parity & short-window protein separation
    const dmp8 = fpc.defaultMaxPrototypes(fpc.DNA, 8);
    const hostAA = "ACDEFGHIKL".repeat(45);
    const qryAA = "ACDEFGHIKL".repeat(6) + "WWWWYYYYWW".repeat(4) +
                  "ACDEFGHIKL".repeat(22) + "WWWWYYYYWW".repeat(4) +
                  "ACDEFGHIKL".repeat(6);
    const scanAA = fpc.scanAgainstReference(
      [{ id: "ref_prot", seq: hostAA }],
      [{ id: "qry_prot", seq: qryAA }],
      { alphabet: fpc.PROTEIN, order: 3, window: 30, step: 7, mergeGap: 60, zThreshold: 2.5 }
    );
    process.stdout.write(JSON.stringify({
      topRight, mid, aCorner, dmp8, nProtIslands: scanAA.islands.length
    }));
    """
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-e", js_check],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    assert out["dmp8"] == islands.default_max_prototypes(DNA, 8)
    assert out["aCorner"] is None
    assert out["topRight"] is not None and out["topRight"] > 1.2
    assert out["mid"] is not None and out["topRight"] > out["mid"]
    assert out["nProtIslands"] == 2


# --------------------------------------------------------------------- #
#  9. Exhaustive mathematical, logical & buffer-safety regression tests #
# --------------------------------------------------------------------- #
def test_all_wildcard_distance_matrix_and_nan_nj_rejected():
    with pytest.raises(ValueError, match="0 canonical"):
        tasks.distance_matrix(["ACGTACGT" * 10, "NNNNNNNN" * 10, "TTTTAAAA" * 10], DNA, order=4)

    with pytest.raises(ValueError, match="non-finite"):
        phylo.neighbor_joining(
            np.array([[0.0, np.nan, 1.0], [np.nan, 0.0, 1.0], [1.0, 1.0, 0.0]]),
            ["A", "B", "C"],
        )

    node_bin = _find_node()
    if node_bin:
        js_check = """
        import * as fpc from "./docs/fpc-engine.js";
        let dmErr = false, njErr = false;
        try {
          fpc.distanceMatrix(["ACGTACGT".repeat(10), "NNNNNNNN".repeat(10)], { alphabet: fpc.DNA, order: 4 });
        } catch (e) { dmErr = true; }
        try {
          fpc.neighborJoining([[0, NaN, 1], [NaN, 0, 1], [1, 1, 0]], ["A", "B", "C"]);
        } catch (e) { njErr = true; }
        process.stdout.write(JSON.stringify({ dmErr, njErr }));
        """
        proc = subprocess.run(
            [node_bin, "--input-type=module", "-e", js_check],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        res = json.loads(proc.stdout)
        assert res["dmErr"] is True and res["njErr"] is True


def test_mds_embedding_shapes_for_small_n_and_nan_rejection():
    assert tasks.mds_embedding(np.zeros((0, 0)), dims=3).shape == (0, 3)
    assert tasks.mds_embedding(np.zeros((1, 1)), dims=3).shape == (1, 3)
    coords2 = tasks.mds_embedding(np.array([[0.0, 2.0], [2.0, 0.0]]), dims=3)
    assert coords2.shape == (2, 3)
    assert np.linalg.norm(coords2[0] - coords2[1]) == pytest.approx(2.0)
    assert np.allclose(coords2[:, 2], 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        tasks.mds_embedding(np.array([[0.0, np.nan], [np.nan, 0.0]]), dims=3)


def test_cognitive_map_html_all_wildcard_query_finite_prototypes(tmp_path):
    import warnings
    from viz.viz3d import cognitive_map_html

    cx = FractalPredictiveCortex(DNA, order=4)
    cx.fit(DNA.encode("ACGTACGT" * 20))
    out_html = tmp_path / "all_n_cog.html"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cognitive_map_html(cx, DNA.encode("NNNNNN"), str(out_html), grid=16)
    html_text = out_html.read_text(encoding="utf-8")
    assert "predicts " in html_text
    assert "0.08" in html_text


def test_equidistant_prototype_tie_breaking_parity():
    def build_sym(use_fast: bool) -> FractalPredictiveCortex:
        c = FractalPredictiveCortex(DNA, order=2, n_neighbors=1, use_fast=use_fast)
        c._ensure_capacity(8)
        c._m = 4
        c.X[:4] = [[0.0, 0.0], [1.0, 1.0], [0.75, 0.25], [0.25, 0.75]]
        c.N[:4] = [[1, 1, 1, 1], [1, 1, 1, 1], [50, 0, 0, 0], [0, 0, 0, 50]]
        c.usage[:4] = [1.0, 1.0, 1.0, 1.0]
        return c

    c_fast = build_sym(True)
    c_slow = build_sym(False)
    p_slow, nn_slow, _ = c_slow.predict_dist([0.5, 0.5])
    assert nn_slow == 2
    assert p_slow[0] > 0.85

    q = DNA.encode("A")
    b_fast = float(c_fast.score(q).bits[0])
    b_slow_score = float(c_slow.score(q).bits[0])
    b_slow_stream = float(c_slow._stream(q, update=False).bits[0])
    assert abs(b_fast - b_slow_score) < 1e-12
    assert abs(b_fast - b_slow_stream) < 1e-12

    node_bin = _find_node()
    if node_bin:
        js_sym = """
        import * as fpc from "./docs/fpc-engine.js";
        const c = new fpc.FractalPredictiveCortex(fpc.DNA, { order: 2, nNeighbors: 1 });
        c._ensureCapacity(8);
        c._m = 4;
        c.X.set([0.0, 0.0, 1.0, 1.0, 0.75, 0.25, 0.25, 0.75], 0);
        c.N.set([1,1,1,1, 1,1,1,1, 50,0,0,0, 0,0,0,50], 0);
        c.usage.set([1, 1, 1, 1], 0);
        c._grid.rebuild(c.X, 4);
        const pd = c.predictDist(0.5, 0.5);
        const b = c.score(fpc.DNA.encode("A")).bits[0];
        process.stdout.write(JSON.stringify({ nearest: pd.nearest, b }));
        """
        proc = subprocess.run(
            [node_bin, "--input-type=module", "-e", js_sym],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        out = json.loads(proc.stdout)
        assert out["nearest"] == 2
        assert abs(out["b"] - b_fast) < 1e-12


def test_hyperparameter_validation_and_empty_cortex_score_clipping():
    bad_kwargs = [
        {"order": -1},
        {"n_neighbors": 0},
        {"alpha": 0.0},
        {"alpha": -0.5},
        {"background_weight": -0.01},
        {"background_weight": 1.01},
        {"learn_rate": -0.01},
        {"learn_rate": 1.5},
        {"usage_decay": 0.0},
        {"usage_decay": 1.01},
    ]
    for kw in bad_kwargs:
        with pytest.raises(ValueError):
            FractalPredictiveCortex(DNA, **kw)

    c_empty_fast = FractalPredictiveCortex(DNA, use_fast=True)
    c_empty_slow = FractalPredictiveCortex(DNA, use_fast=False)
    c_empty_fast.global_counts[:] = [1.0, 0.0, 0.0, 0.0]
    c_empty_slow.global_counts[:] = [1.0, 0.0, 0.0, 0.0]
    b_ef = float(c_empty_fast.score(DNA.encode("C")).bits[0])
    b_es = float(c_empty_slow.score(DNA.encode("C")).bits[0])
    assert b_ef == pytest.approx(-np.log2(1e-18), rel=1e-12)
    assert b_es == pytest.approx(-np.log2(1e-18), rel=1e-12)

    node_bin = _find_node()
    if node_bin:
        js_val = """
        import * as fpc from "./docs/fpc-engine.js";
        const cases = [
          { order: -1 }, { nNeighbors: 0 }, { alpha: 0 }, { alpha: -1 },
          { backgroundWeight: -0.1 }, { backgroundWeight: 1.1 },
          { learnRate: -0.1 }, { learnRate: 1.2 },
          { usageDecay: 0 }, { usageDecay: 1.1 }
        ];
        let rejected = 0;
        for (const opts of cases) {
          try { new fpc.FractalPredictiveCortex(fpc.DNA, opts); }
          catch (e) { if (e instanceof RangeError) rejected++; }
        }
        const ce = new fpc.FractalPredictiveCortex(fpc.DNA);
        ce.globalCounts.set([1, 0, 0, 0]);
        const bEmpty = ce.score(fpc.DNA.encode("C")).bits[0];
        process.stdout.write(JSON.stringify({ rejected, total: cases.length, bEmpty }));
        """
        proc = subprocess.run(
            [node_bin, "--input-type=module", "-e", js_val],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        out = json.loads(proc.stdout)
        assert out["rejected"] == out["total"]
        assert out["bEmpty"] == pytest.approx(-np.log2(1e-18), rel=1e-12)


def test_c_kernel_rejects_readonly_and_undersized_buffers():
    from fpc import _fast

    cx = FractalPredictiveCortex(DNA)
    verts = np.ascontiguousarray(cx.embedding._all_vertices, dtype=np.float64)
    ids1 = np.array([0], dtype=np.int64)
    ro_bytes = bytes([7] * 16)
    with pytest.raises((TypeError, BufferError)):
        _fast.cgr_embed(ids1, verts, 0.5, 0.5, 0.5, ro_bytes)
    assert ro_bytes == bytes([7] * 16)

    C1 = np.zeros((1, 2), dtype=np.float64)
    X_tiny = np.zeros((1, 2), dtype=np.float64)
    N_tiny = np.zeros((1, 4), dtype=np.float64)
    gc = np.ones(4, dtype=np.float64)
    bits1 = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError, match="Invalid buffer dimensions"):
        _fast.cortex_score_batch(ids1, C1, X_tiny, N_tiny, gc, bits1, 4, 5, 1, 0.1, 0.5, 0.02)


def test_mask_low_quality_non_ascii_and_baselines_and_isolated_imports(tmp_path):
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from benchmarks.afbaselines import mash_sketch
    from benchmarks.baselines import nrc_nccd_distance_matrices

    masked = io_seq.mask_low_quality("AC\ufffdT?G", "II\ufffdI#I", min_phred=20, wildcard="N")
    assert masked == "ACNTNG"

    s_dna = mash_sketch("ATGCTAGCTAGCTAGC", k=8, canonical=True)
    s_rna = mash_sketch("AUGCUAGCUAGCUAGC", k=8, canonical=True)
    assert s_dna == s_rna and len(s_dna) > 0

    s1 = "ACGTACGT" * 100
    s2 = "TTTTAAAA" * 100
    _, nccd_clean = nrc_nccd_distance_matrices([s1, s2], DNA, k=4)
    _, nccd_gaps = nrc_nccd_distance_matrices([s1, s2 + "N" * 1000], DNA, k=4)
    assert np.allclose(nccd_clean, nccd_gaps, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="0 canonical"):
        nrc_nccd_distance_matrices([s1, "NNNNNNNN"], DNA, k=4)

    check_cmd = "\n".join([
        "import importlib.util, os",
        f"root = {REPO_ROOT!r}",
        "for rel in ('examples/quickstart.py', 'examples/make_visuals.py', 'benchmarks/bench.py'):",
        "    p = os.path.join(root, rel)",
        "    spec = importlib.util.spec_from_file_location('m', p)",
        "    mod = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(mod)",
        "    assert hasattr(mod, 'main')",
    ])
    res = subprocess.run(
        [sys.executable, "-c", check_cmd],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0, res.stderr





def test_alphabet_validation_and_c_kernel_nan_guards():
    from fpc import Alphabet, _fast

    with pytest.raises(ValueError, match="at least 1 symbol"):
        Alphabet("EMPTY", "")
    with pytest.raises(ValueError, match="single ASCII"):
        Alphabet("BADW", "ACGT", wildcard="NN")
    with pytest.raises(ValueError, match="duplicate"):
        Alphabet("DUP", "AACG")
    with pytest.raises(ValueError, match="wildcard"):
        Alphabet("COLLIDE", "ACGTN", wildcard="N")

    q_alpha = Alphabet("Q", "AB?", wildcard="*")
    assert q_alpha.encode("A\ufffdB?").tolist() == [0, 3, 1, 2]

    cx = FractalPredictiveCortex(DNA)
    verts = np.ascontiguousarray(cx.embedding._all_vertices, dtype=np.float64)
    ids1 = np.array([0], dtype=np.int64)
    out1 = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="cgr_embed"):
        _fast.cgr_embed(ids1, verts, 0.5, 0.5, float("nan"), out1)

    C1 = np.zeros((1, 2), dtype=np.float64)
    X1 = np.zeros((4, 2), dtype=np.float64)
    N1 = np.zeros((4, 4), dtype=np.float64)
    gc = np.ones(4, dtype=np.float64)
    bits1 = np.zeros(1, dtype=np.float64)
    cx_nan = FractalPredictiveCortex(DNA, use_fast=True)
    cx_nan.sigma = float("nan")
    with pytest.raises(ValueError, match="cortex_stream"):
        cx_nan.fit(ids1)
    with pytest.raises(ValueError, match="cortex_score_batch"):
        _fast.cortex_score_batch(
            ids1, C1, X1, N1, gc, bits1, 4, 1, 1, float("nan"), 0.5, 0.02
        )
    with pytest.raises(ValueError, match="cortex_score_batch"):
        _fast.cortex_score_batch(
            ids1, C1, X1, N1, gc, bits1, 4, 1, 1, 0.1, 0.5, 1.5
        )


def test_surprise_anomalies_cli_predict_classifier_mst_fasta_and_datasets(tmp_path):
    import re
    import warnings

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from benchmarks.afbaselines import ffp_distance_matrix
    from fpc import io_fasta
    from fpc.surprise import SurpriseTrack
    from viz.viz3d import phylogeny_3d_html

    # 1. Wildcard-bridged anomaly spike produces finite mean_bits with zero RuntimeWarnings
    arr = np.array(
        [1.0] * 20 + [1.0, 1.0, 10.0, np.nan, np.nan, 10.0, 1.0, 1.0] + [1.0] * 20,
        dtype=np.float64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        anoms = SurpriseTrack(arr).anomalies(window=5, z_threshold=3.5)
    assert len(anoms) >= 1 and all(np.isfinite(a.mean_bits) for a in anoms)

    # 2. CLI predict --eval scores multi-record FASTA independently and emits valid RFC-8259 JSON on all-N
    tr_fa = tmp_path / "tr.fa"
    ev_fa = tmp_path / "ev.fa"
    n_fa = tmp_path / "alln.fa"
    r1_seq = "ACGT" * 25
    r2_seq = "ACGT" * 25
    io_seq.write_fasta(str(tr_fa), [("tr", "ACGT" * 200 + "TTTT" * 200)])
    io_seq.write_fasta(str(ev_fa), [("r1", r1_seq + "TTTT"), ("r2", r2_seq)])
    io_seq.write_fasta(str(n_fa), [("n1", "NNNNNNNN")])

    cx = FractalPredictiveCortex(DNA, order=6)
    cx.fit(DNA.encode("ACGT" * 200 + "TTTT" * 200))
    b1 = tasks.evaluate_prediction(cx, r1_seq + "TTTT")
    b2 = tasks.evaluate_prediction(cx, r2_seq)
    expected_ce = (b1["cross_entropy"] * b1["n"] + b2["cross_entropy"] * b2["n"]) / (
        b1["n"] + b2["n"]
    )
    res_ev = subprocess.run(
        [sys.executable, "-m", "fpc", "predict", "--train", str(tr_fa), "--eval", str(ev_fa), "-k", "6"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    )
    ev_json = json.loads(res_ev.stdout)
    assert ev_json["cross_entropy"] == pytest.approx(expected_ce, rel=1e-12)

    res_n = subprocess.run(
        [sys.executable, "-m", "fpc", "predict", "--train", str(tr_fa), "--eval", str(n_fa), "-k", "6"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    )
    n_json = json.loads(res_n.stdout)
    assert n_json["n"] == 0 and n_json["cross_entropy"] is None and n_json["accuracy"] is None

    # 3. FPCClassifier.predict on all-N returns UNCLASSIFIED and fit rejects invalid classes/lengths
    clf = tasks.FPCClassifier(DNA, order=4).fit(["AAAA" * 50, "CCCC" * 50], ["sp_a", "sp_c"])
    assert clf.predict(["NNNNNNNN", "AAAA" * 20]) == ["UNCLASSIFIED", "sp_a"]
    with pytest.raises(ValueError, match="length"):
        tasks.FPCClassifier(DNA).fit(["ACGT" * 20], ["a", "b"])
    with pytest.raises(ValueError, match="0 canonical"):
        tasks.FPCClassifier(DNA).fit(["ACGT" * 20, "NNNNNNNN"], ["a", "b"])

    # 4. viz3d phylogeny_3d_html preserves 0.0-distance MST edges between identical taxa
    D_zero = np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.8, 0.8, 0.0]], dtype=np.float64)
    coords_zero = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=np.float64)
    html_p = tmp_path / "phylo.html"
    phylogeny_3d_html(
        coords_zero, ["g1", "g1", "g2"], str(html_p),
        names=["t1", "t2", "t3"], distance_matrix=D_zero,
    )
    html_txt = html_p.read_text(encoding="utf-8")
    assert "lineage (MST)" in html_txt
    assert "[10.0,20.0,null,10.0,30.0,null]" in html_txt.replace(" ", "")

    # 5. io_fasta rejects non-FASTA / empty headers; datasets RNA gc & empty host; ffp all-zero k-mers; islands guard_bp
    bad_fq = tmp_path / "bad.fq"
    bad_fq.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed FASTA"):
        io_fasta.read_fasta(str(bad_fq))

    rna_seq = datasets.random_sequence(4000, RNA, gc=0.85, seed=7)
    rna_gc = (rna_seq.count("G") + rna_seq.count("C")) / len(rna_seq)
    assert 0.80 <= rna_gc <= 0.90
    chim, bounds = datasets.insert_foreign("", "ACGT", seed=1)
    assert chim == "ACGT" and bounds == (0, 3)

    with pytest.raises(ValueError, match="0 canonical"):
        ffp_distance_matrix(["AAAA", "TTTT"], DNA, k=6)
    with pytest.raises(ValueError, match="guard_bp"):
        islands.scan_self_islands([("c1", "ACGT" * 500)], guard_bp=-10)
    with pytest.raises(ValueError, match="0 canonical"):
        islands.scan_against_reference([("ref", "NNNNNNNN")], [("q", "ACGT" * 500)])

    # 6. classification_accuracy, cluster, and clustering_purity validation
    with pytest.raises(ValueError, match="pred length"):
        tasks.classification_accuracy(["A"], ["A", "B"])
    assert np.isnan(tasks.classification_accuracy([], []))
    with pytest.raises(ValueError, match="pred length"):
        tasks.clustering_purity([1], [1, 2])
    assert np.isnan(tasks.clustering_purity([], []))
    assert list(tasks.cluster(np.zeros((0, 0)), 2)) == []
    assert list(tasks.cluster([[0.0]], 2)) == [1]
    with pytest.raises(ValueError, match="n_clusters"):
        tasks.cluster(np.zeros((2, 2)), 0)
    with pytest.raises(ValueError, match="non-finite"):
        tasks.cluster(np.array([[0.0, np.nan], [np.nan, 0.0]]), 2)


def test_js_engine_and_worker_edge_case_validations():
    node_bin = _find_node()
    if not node_bin:
        pytest.skip("Node.js runtime not installed")

    js_code = r"""
    import * as fpc from "./docs/fpc-engine.js";

    let checks = 0;
    try { new fpc.Alphabet("E", ""); } catch (e) { if (e instanceof RangeError) checks++; }
    try { new fpc.Alphabet("D", "AACG"); } catch (e) { if (e instanceof RangeError) checks++; }
    try { new fpc.Alphabet("W", "ACGT", "A"); } catch (e) { if (e instanceof RangeError) checks++; }
    try { new fpc.ChaosGameEmbedding(fpc.DNA, 1.5); } catch (e) { if (e instanceof RangeError) checks++; }
    try { new fpc.ChaosGameEmbedding(fpc.DNA, NaN); } catch (e) { if (e instanceof RangeError) checks++; }
    try { fpc.scanSelfIslands([{ id: "c", seq: "ACGT".repeat(300) }], { window: NaN }); }
    catch (e) { if (e instanceof RangeError) checks++; }
    try { fpc.scanSelfIslands([{ id: "c", seq: "ACGT".repeat(300) }], { guardBp: -5 }); }
    catch (e) { if (e instanceof RangeError) checks++; }
    try { fpc.scanAgainstReference([{ id: "r", seq: "NNNNNN" }], [{ id: "q", seq: "ACGT".repeat(200) }]); }
    catch (e) { checks++; }
    try { new fpc.FPCClassifier(fpc.DNA).fit([{ label: "a", seq: "NNNN" }, { label: "b", seq: "ACGT".repeat(50) }]); }
    catch (e) { checks++; }

    const qAlpha = new fpc.Alphabet("Q", "AB?", "*");
    const enc = Array.from(qAlpha.encode("A\u00ffB?"));

    const arr = new Float64Array(48);
    arr.fill(1.0);
    arr[22] = 10.0; arr[23] = NaN; arr[24] = NaN; arr[25] = 10.0;
    const anoms = new fpc.SurpriseTrack(arr).anomalies(5, 3.5);
    const anomFinite = anoms.length >= 1 && anoms.every(a => Number.isFinite(a.meanBits));

    const msgs = [];
    globalThis.self = { postMessage: (m) => msgs.push(m), onmessage: null };
    await import("./docs/fpc-worker.js");

    self.onmessage({ data: { id: 101, task: "predict", payload: {
      alphabet: "DNA", order: 0, trainText: ">tr\n" + "ACGT".repeat(200), prefix: "ACGT", length: 8, sample: false
    }}});
    self.onmessage({ data: { id: 102, task: "classify", payload: {
      alphabet: "DNA", order: 5, minPhred: 0, strandAware: true,
      refText: ">dup\n" + "ACGT".repeat(100) + "\n>dup\n" + "TTTT".repeat(100),
      queryText: ">q1\nACGTACGT"
    }}});
    const m101 = msgs.find(m => m.id === 101 && m.type === "result");
    const m102 = msgs.find(m => m.id === 102 && m.type === "error");
    process.stdout.write(JSON.stringify({
      checks,
      enc,
      anomFinite,
      order0Protos: m101 ? m101.result.nPrototypes : -1,
      dupRejected: Boolean(m102 && m102.message && m102.message.includes("distinct")),
    }));
    """
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-e", js_code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["checks"] == 9
    assert out["enc"] == [0, 3, 1, 2]
    assert out["anomFinite"] is True
    assert out["order0Protos"] == 1
    assert out["dupRejected"] is True
