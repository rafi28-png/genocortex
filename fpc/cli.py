"""
GenoCortex Production Command-Line Interface (``genocortex`` / ``python -m fpc``).
==================================================================================

Subcommands:
    train     Train & save a single-genome cortex or multi-class MDL classifier (.fpc.npz)
    scan      Detect horizontal gene transfer, pathogenicity islands & anomalies (BED/BedGraph/TSV)
    classify  Classify FASTA/FASTQ reads or contigs with strand calling & posteriors
    tree      Alignment-free cross-entropy distance matrix & Neighbour-Joining Newick tree
    predict   Evaluate next-symbol predictive perplexity or generate continuations
    info      Inspect compiled C kernel status and .fpc.npz model archives
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import numpy as np

from . import __version__
from .alphabet import get_alphabet
from .cortex import FractalPredictiveCortex, HAS_FAST_KERNEL
from .io_seq import SeqRecord, iter_sequences, read_sequences
from . import islands, phylo, tasks


def _stem_name(path: str) -> str:
    base = os.path.basename(path.rstrip("/\\"))
    if base.lower().endswith(".gz"):
        base = base[:-3]
    root, _ = os.path.splitext(base)
    return root or base


def _expand_input_files(paths: list[str]) -> list[str]:
    from .io_seq import SEQ_EXTENSIONS

    out: list[str] = []
    for p in paths:
        if p != "-" and os.path.isdir(p):
            matched = [
                os.path.join(p, name)
                for name in sorted(os.listdir(p))
                if name.lower().endswith(SEQ_EXTENSIONS)
            ]
            if not matched:
                raise ValueError(f"No FASTA/FASTQ files found in directory {p!r}")
            out.extend(matched)
        else:
            out.append(p)
    return out


# ===================================================================== #
#  Subcommand implementations                                           #
# ===================================================================== #
def _cmd_train(args: argparse.Namespace) -> int:
    alphabet = get_alphabet(args.alphabet)
    t0 = time.perf_counter()
    files = _expand_input_files(args.inputs)

    if args.by_file or args.by_record:
        eff_max_proto = (
            args.max_prototypes
            if args.max_prototypes is not None
            else islands.default_max_prototypes(alphabet, args.order)
        )
        seqs: list[str] = []
        labels: list[str] = []
        total_bp = 0
        if args.by_file:
            for fp in files:
                label = _stem_name(fp)
                for rec in iter_sequences(fp, min_phred=args.min_phred, wildcard=alphabet.wildcard):
                    seqs.append(rec.seq)
                    labels.append(label)
                    total_bp += len(rec.seq)
        else:
            for fp in files:
                for rec in iter_sequences(fp, min_phred=args.min_phred, wildcard=alphabet.wildcard):
                    seqs.append(rec.seq)
                    labels.append(rec.id)
                    total_bp += len(rec.seq)

        if not seqs:
            raise ValueError("No sequences found for classifier training")
        if len(set(labels)) < 1:
            raise ValueError("At least 1 class label is required")

        clf = tasks.FPCClassifier(
            alphabet=alphabet,
            order=args.order,
            epochs=args.epochs,
            strand_aware=not args.single_strand,
            double_strand_train=bool(args.double_strand),
            max_prototypes=eff_max_proto,
        )
        clf.fit(seqs, labels)
        clf.save(args.output)
        dt = time.perf_counter() - t0
        total_protos = sum(m.n_prototypes for m in clf.models.values())
        if not args.quiet:
            print(
                f"[genocortex train] Saved FPCClassifier ({len(clf.classes_)} classes, "
                f"{total_protos:,} total prototypes, {total_bp:,} bp) -> {args.output} "
                f"in {dt:.2f}s ({total_bp / max(dt, 1e-9) / 1e6:.2f} Mb/s)",
                file=sys.stderr,
            )
        return 0

    eff_max_proto = (
        args.max_prototypes
        if args.max_prototypes is not None
        else islands.default_max_prototypes(alphabet, args.order)
    )
    cortex = FractalPredictiveCortex(
        alphabet=alphabet,
        order=args.order,
        max_prototypes=eff_max_proto,
    )
    double_strand = (not args.single_strand) and alphabet.is_complementary
    n_records = 0
    total_bp = 0
    for fp in files:
        for rec in iter_sequences(fp, min_phred=args.min_phred, wildcard=alphabet.wildcard):
            ids = alphabet.encode(rec.seq)
            cortex.fit(ids, epochs=args.epochs, double_strand=double_strand)
            n_records += 1
            total_bp += len(ids)

    if n_records == 0 or total_bp == 0:
        raise ValueError("No sequence bases found to train cortex")

    cortex.save(args.output)
    dt = time.perf_counter() - t0
    if not args.quiet:
        print(
            f"[genocortex train] Saved FractalPredictiveCortex "
            f"(alphabet={alphabet.name}, order={args.order}, prototypes={cortex.n_prototypes:,}, "
            f"records={n_records}, bases={total_bp:,}) -> {args.output} "
            f"in {dt:.2f}s ({total_bp / max(dt, 1e-9) / 1e6:.2f} Mb/s)",
            file=sys.stderr,
        )
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    if args.ref and args.model:
        raise ValueError("Specify either --ref or --model, not both")

    t0 = time.perf_counter()
    model_path = args.model
    ref_fasta = args.ref
    if ref_fasta and ref_fasta.lower().endswith(".npz"):
        model_path = ref_fasta
        ref_fasta = None

    if model_path:
        loaded_cortex = FractalPredictiveCortex.load(model_path)
        alphabet = loaded_cortex.alphabet
    else:
        loaded_cortex = None
        alphabet = get_alphabet(args.alphabet)

    query_recs = read_sequences(args.query, min_phred=args.min_phred, wildcard=alphabet.wildcard)
    query_bp = sum(len(r.seq) for r in query_recs)
    double_strand = (not args.single_strand) and alphabet.is_complementary

    if loaded_cortex is not None:
        isl_list, win_tracks, cortex = islands.scan_against_reference(
            ref_records=[],
            query_records=query_recs,
            alphabet=alphabet,
            order=loaded_cortex.order,
            window=args.window,
            step=args.step,
            z_threshold=args.z_threshold,
            alpha=args.alpha,
            merge_gap=args.merge_gap,
            min_length=args.min_length,
            cortex=loaded_cortex,
        )
        mode_desc = f"model={model_path}"
    elif ref_fasta:
        ref_recs = read_sequences(ref_fasta, min_phred=args.min_phred, wildcard=alphabet.wildcard)
        isl_list, win_tracks, cortex = islands.scan_against_reference(
            ref_records=ref_recs,
            query_records=query_recs,
            alphabet=alphabet,
            order=args.order,
            max_prototypes=args.max_prototypes,
            double_strand=double_strand,
            window=args.window,
            step=args.step,
            z_threshold=args.z_threshold,
            alpha=args.alpha,
            merge_gap=args.merge_gap,
            min_length=args.min_length,
        )
        mode_desc = f"ref={ref_fasta}"
    else:
        isl_list, win_tracks = islands.scan_self_islands(
            query_records=query_recs,
            alphabet=alphabet,
            order=args.order,
            max_prototypes=args.max_prototypes,
            double_strand=double_strand,
            n_folds=args.folds,
            window=args.window,
            step=args.step,
            z_threshold=args.z_threshold,
            alpha=args.alpha,
            merge_gap=args.merge_gap,
            min_length=args.min_length,
        )
        cortex = None
        mode_desc = f"self-{args.folds}fold"

    if args.bed:
        islands.write_bed(isl_list, args.bed, only_significant=args.only_significant)
    if args.bedgraph:
        islands.write_bedgraph(win_tracks, args.bedgraph, metric=args.bedgraph_metric)
    if args.tsv:
        islands.write_islands_tsv(isl_list, args.tsv)
    if args.html:
        from viz import viz3d
        if cortex is None:
            cortex = FractalPredictiveCortex(
                alphabet,
                order=args.order,
                max_prototypes=islands.default_max_prototypes(alphabet, args.order),
            )
            for r in query_recs:
                cortex.fit(alphabet.encode(r.seq), double_strand=double_strand)
        first_ids = alphabet.encode(query_recs[0].seq)
        viz3d.cognitive_map_html(
            cortex,
            first_ids,
            args.html,
            title=f"GenoCortex Cognitive Map — {query_recs[0].id}",
        )

    dt = time.perf_counter() - t0
    n_sig = sum(1 for x in isl_list if x.significant)
    if not args.quiet:
        print(
            f"[genocortex scan] Scanned {query_bp:,} bp ({mode_desc}) in {dt:.2f}s "
            f"({query_bp / max(dt, 1e-9) / 1e6:.2f} Mb/s): "
            f"{len(isl_list)} candidate island(s), {n_sig} FDR-significant (q <= {args.alpha})",
            file=sys.stderr,
        )

    # Emit TSV table to stdout when no tabular file flag was specified (or --stdout)
    if args.stdout or (not args.bed and not args.tsv):
        sys.stdout.write(
            "chrom\tstart\tend\tlength\tpeak_pos\tpeak_z\tmean_bits\tdelta_bits\t"
            "gc_content\tdelta_gc\tp_value\tq_value\tsignificant\n"
        )
        for isl in isl_list:
            if args.only_significant and not isl.significant:
                continue
            gc_s = f"{isl.gc_content:.4f}" if np.isfinite(isl.gc_content) else "NA"
            dgc_s = f"{isl.delta_gc:+.4f}" if np.isfinite(isl.delta_gc) else "NA"
            sys.stdout.write(
                f"{isl.chrom}\t{isl.start}\t{isl.end}\t{isl.length}\t{isl.peak_pos}\t"
                f"{isl.peak_z:.3f}\t{isl.mean_bits:.4f}\t{isl.delta_bits:+.4f}\t"
                f"{gc_s}\t{dgc_s}\t{isl.p_value:.3e}\t{isl.q_value:.3e}\t"
                f"{str(isl.significant).lower()}\n"
            )
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    ref_inputs = args.refs or args.train_refs
    if bool(args.model) == bool(ref_inputs):
        raise ValueError("Specify exactly one of --model <panel.fpc.npz> or --refs/--train <fasta...>")

    t0 = time.perf_counter()
    if args.model:
        clf = tasks.FPCClassifier.load(args.model)
        if args.single_strand:
            clf.strand_aware = False
        alphabet = clf.alphabet
    else:
        alphabet = get_alphabet(args.alphabet)
        ref_files = _expand_input_files(ref_inputs)
        eff_max_proto = (
            args.max_prototypes
            if args.max_prototypes is not None
            else islands.default_max_prototypes(alphabet, args.order)
        )
        seqs: list[str] = []
        labels: list[str] = []
        by_rec = args.by_record or (len(ref_files) == 1 and not args.by_file)
        for fp in ref_files:
            stem = _stem_name(fp)
            for rec in iter_sequences(fp, min_phred=args.min_phred, wildcard=alphabet.wildcard):
                seqs.append(rec.seq)
                labels.append(rec.id if by_rec else stem)
        clf = tasks.FPCClassifier(
            alphabet=alphabet,
            order=args.order,
            strand_aware=not args.single_strand,
            double_strand_train=bool(args.double_strand),
            max_prototypes=eff_max_proto,
        )
        clf.fit(seqs, labels)

    if args.save_model:
        clf.save(args.save_model)
        if not args.queries:
            dt = time.perf_counter() - t0
            if not args.quiet:
                print(
                    f"[genocortex classify] Trained & saved FPCClassifier "
                    f"({len(clf.classes_)} classes) -> {args.save_model} in {dt:.2f}s",
                    file=sys.stderr,
                )
            return 0

    if not args.queries:
        raise ValueError("Provide query FASTA/FASTQ file(s) to classify (or --save-model to train & save)")

    query_files = _expand_input_files(args.queries)
    query_recs: list[SeqRecord] = []
    for qf in query_files:
        query_recs.extend(read_sequences(qf, min_phred=args.min_phred, wildcard=alphabet.wildcard))

    calls = clf.classify_records(query_recs)
    dt = time.perf_counter() - t0

    lines = ["id\tpredicted_class\tstrand\tbits_per_base\tposterior\trunner_up\tdelta_bits\n"]
    for c in calls:
        ru = c.runner_up if c.runner_up is not None else "NA"
        bpb_s = f"{c.bits_per_base:.4f}" if np.isfinite(c.bits_per_base) else "NA"
        db_s = f"{c.delta_bits:.4f}" if np.isfinite(c.delta_bits) else "0.0000"
        lines.append(
            f"{c.id}\t{c.predicted_class}\t{c.strand}\t{bpb_s}\t"
            f"{c.posterior:.4f}\t{ru}\t{db_s}\n"
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(lines)
    else:
        sys.stdout.writelines(lines)

    if not args.quiet:
        total_bp = sum(len(r.seq) for r in query_recs)
        print(
            f"[genocortex classify] Classified {len(calls):,} record(s) ({total_bp:,} bp) "
            f"across {len(clf.classes_)} classes in {dt:.2f}s",
            file=sys.stderr,
        )
    return 0


def _cmd_tree(args: argparse.Namespace) -> int:
    alphabet = get_alphabet(args.alphabet)
    t0 = time.perf_counter()
    files = _expand_input_files(args.inputs)

    names: list[str] = []
    seqs: list[str] = []
    per_file = args.per_file or (len(files) > 1 and not args.per_record)

    if per_file:
        spacer = alphabet.wildcard * max(1, args.order)
        for fp in files:
            recs = read_sequences(fp, min_phred=args.min_phred, wildcard=alphabet.wildcard)
            names.append(_stem_name(fp))
            seqs.append(spacer.join(r.seq for r in recs))
    else:
        for fp in files:
            for rec in iter_sequences(fp, min_phred=args.min_phred, wildcard=alphabet.wildcard):
                names.append(rec.id)
                seqs.append(rec.seq)

    if len(seqs) < 2:
        raise ValueError(f"At least 2 taxa are required to build a tree, got {len(seqs)}")

    eff_max_proto = (
        args.max_prototypes
        if args.max_prototypes is not None
        else islands.default_max_prototypes(alphabet, args.order)
    )
    strand_inv = (not args.strand_specific) and alphabet.is_complementary
    D = tasks.distance_matrix(
        seqs,
        alphabet=alphabet,
        order=args.order,
        strand_invariant=strand_inv,
        max_prototypes=eff_max_proto,
    )
    tree = phylo.neighbor_joining(D, names)
    nwk = tree.to_newick(precision=6)

    if args.output:
        phylo.write_newick(nwk, args.output)
    else:
        sys.stdout.write(nwk + "\n")

    if args.matrix:
        phylo.write_distance_tsv(D, names, args.matrix)

    if args.html:
        from viz import viz3d
        coords = tasks.mds_embedding(D, dims=3)
        k_clusters = min(len(seqs), max(1, args.clusters))
        labels = tasks.cluster(D, n_clusters=k_clusters) if k_clusters > 1 and len(seqs) >= 2 else [1] * len(seqs)
        viz3d.phylogeny_3d_html(
            coords,
            labels,
            args.html,
            names=names,
            distance_matrix=D,
            title="GenoCortex Cross-Entropy Phylogeny",
        )

    dt = time.perf_counter() - t0
    if not args.quiet:
        total_bp = sum(len(s) for s in seqs)
        print(
            f"[genocortex tree] Built Neighbour-Joining tree for {len(seqs)} taxa "
            f"({total_bp:,} bp, order={args.order}, strand_invariant={strand_inv}) in {dt:.2f}s",
            file=sys.stderr,
        )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    sources = int(bool(args.model)) + int(bool(args.train)) + int(bool(args.seq))
    if sources != 1:
        raise ValueError("Specify exactly one of --model <cortex.fpc.npz>, --train <seq.fa>, or --seq <SEQUENCE>")

    if args.model:
        cortex = FractalPredictiveCortex.load(args.model)
        alphabet = cortex.alphabet
    else:
        alphabet = get_alphabet(args.alphabet)
        cortex = FractalPredictiveCortex(
            alphabet=alphabet,
            order=args.order,
            max_prototypes=islands.default_max_prototypes(alphabet, args.order),
            seed=args.seed,
        )
        if args.seq:
            cortex.fit(alphabet.encode(args.seq.strip().upper()))
        else:
            recs = read_sequences(args.train, wildcard=alphabet.wildcard)
            for r in recs:
                cortex.fit(alphabet.encode(r.seq))

    if args.eval:
        eval_recs = read_sequences(args.eval, wildcard=alphabet.wildcard)
        joined = "".join(r.seq for r in eval_recs)
        metrics = tasks.evaluate_prediction(cortex, joined)
        sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
        return 0

    if not args.prefix:
        raise ValueError("Provide --prefix/--context <SEQ> for continuation or --eval <FASTA> for evaluation")

    cortex._rng = np.random.default_rng(args.seed)
    generated = tasks.complete(
        cortex,
        prefix=args.prefix.upper(),
        length=args.length,
        sample=args.sample,
        temperature=args.temperature,
    )
    sys.stdout.write(generated + "\n")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    info: dict = {
        "genocortex_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "fast_c_kernel": bool(HAS_FAST_KERNEL),
        "alphabets": ["DNA", "RNA", "PROTEIN"],
    }
    if args.model:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model!r}")
        with np.load(args.model, allow_pickle=False) as archive:
            meta = json.loads(bytes(np.asarray(archive["meta_json"], dtype=np.uint8)).decode("utf-8"))
        info["model_file"] = args.model
        info["model_meta"] = meta
    sys.stdout.write(json.dumps(info, indent=2) + "\n")
    return 0


# ===================================================================== #
#  Argument parser                                                      #
# ===================================================================== #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genocortex",
        description=(
            "GenoCortex v1.0 — Fractal Predictive Cortex for megabase genomics, "
            "HGT/pathogenicity island detection, strand-aware read classification, "
            "and alignment-free Neighbour-Joining phylogenetics."
        ),
    )
    p.add_argument("--version", action="version", version=f"genocortex {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # -- train -----------------------------------------------------------
    pt = sub.add_parser("train", help="Train and save a cortex or multi-class classifier (.fpc.npz)")
    pt.add_argument("inputs", nargs="+", help="Input FASTA/FASTQ(.gz) file(s) or directory")
    pt.add_argument("-o", "--output", required=True, help="Output .fpc.npz model path")
    pt.add_argument("-a", "--alphabet", default="dna", help="Alphabet: dna, rna, or protein (default: dna)")
    pt.add_argument("-k", "--order", type=int, default=7, help="Effective CGR context order (default: 7)")
    pt.add_argument("--max-prototypes", type=int, default=None, help="Maximum prototype budget")
    pt.add_argument("--epochs", type=int, default=1, help="Number of streaming passes (default: 1)")
    pt.add_argument("--by-file", action="store_true", help="Train a multi-class FPCClassifier (one class per input file)")
    pt.add_argument("--by-record", action="store_true", help="Train a multi-class FPCClassifier (one class per record ID)")
    pt.add_argument("--double-strand", action="store_true", help="Enable double-strand reverse-complement learning (default for DNA/RNA)")
    pt.add_argument("--single-strand", action="store_true", help="Disable double-strand reverse-complement learning")
    pt.add_argument("--min-phred", type=int, default=0, help="Mask FASTQ bases with Phred quality < MIN_PHRED")
    pt.add_argument("-q", "--quiet", action="store_true", help="Suppress stderr progress summary")
    pt.set_defaults(func=_cmd_train)

    # -- scan ------------------------------------------------------------
    ps = sub.add_parser("scan", help="Scan chromosomes/contigs for HGT & pathogenicity islands")
    ps.add_argument("query", help="Query FASTA/FASTQ(.gz) file or '-' for stdin")
    ps.add_argument("--ref", default=None, help="Reference genome FASTA/FASTQ(.gz) or pre-trained .fpc.npz model")
    ps.add_argument("--model", default=None, help="Pre-trained reference cortex (.fpc.npz)")
    ps.add_argument("--self", dest="self_scan", action="store_true", help="Reference-free buffered K-fold island scan (default if no --ref/--model)")
    ps.add_argument("-a", "--alphabet", default="dna", help="Alphabet: dna, rna, or protein (default: dna)")
    ps.add_argument("-k", "--order", type=int, default=7, help="Effective CGR context order (default: 7)")
    ps.add_argument("--max-prototypes", type=int, default=None, help="Maximum prototype budget")
    ps.add_argument("--folds", type=int, default=8, help="Number of buffered folds for reference-free scan (default: 8)")
    ps.add_argument("-w", "--window", type=int, default=1000, help="Sliding window width in bp (default: 1000)")
    ps.add_argument("-s", "--step", type=int, default=250, help="Sliding window step in bp (default: 250)")
    ps.add_argument("-z", "--z-threshold", type=float, default=3.5, help="Robust z-score threshold (default: 3.5)")
    ps.add_argument("--alpha", "--fdr", dest="alpha", type=float, default=0.05, help="Benjamini-Hochberg FDR q-value cutoff (default: 0.05)")
    ps.add_argument("--merge-gap", type=int, default=2000, help="Max bp gap to merge adjacent island windows (default: 2000)")
    ps.add_argument("--min-length", type=int, default=None, help="Minimum island length in bp (default: window)")
    ps.add_argument("--double-strand", action="store_true", help="Enable double-strand reference training (default for DNA/RNA)")
    ps.add_argument("--single-strand", action="store_true", help="Disable double-strand reference training")
    ps.add_argument("--min-phred", type=int, default=0, help="Mask FASTQ bases with Phred quality < MIN_PHRED")
    ps.add_argument("--bed", default=None, help="Write standard 6-column BED file")
    ps.add_argument("--bedgraph", default=None, help="Write UCSC/IGV BedGraph track")
    ps.add_argument("--bedgraph-metric", choices=["z_score", "bits"], default="z_score", help="BedGraph value column (default: z_score)")
    ps.add_argument("-o", "--output", "--tsv", dest="tsv", default=None, help="Write full statistical TSV island report")
    ps.add_argument("--html", default=None, help="Write offline 3-D Plotly cognitive map HTML")
    ps.add_argument("--only-significant", action="store_true", help="Only emit islands passing Benjamini-Hochberg FDR")
    ps.add_argument("--stdout", action="store_true", help="Also print TSV island table to stdout")
    ps.add_argument("-q", "--quiet", action="store_true", help="Suppress stderr summary")
    ps.set_defaults(func=_cmd_scan)

    # -- classify --------------------------------------------------------
    pc = sub.add_parser("classify", help="Classify FASTA/FASTQ reads or contigs by minimum description length")
    pc.add_argument("queries", nargs="*", help="Query FASTA/FASTQ(.gz) file(s), directory, or '-'")
    pc.add_argument("--model", default=None, help="Saved FPCClassifier archive (.fpc.npz)")
    pc.add_argument("--refs", nargs="+", default=None, help="Reference FASTA/FASTQ file(s) or directory to train on the fly")
    pc.add_argument("--train", dest="train_refs", nargs="+", default=None, help="Alias for --refs (train multi-class classifier from file(s) or directory)")
    pc.add_argument("--save-model", default=None, help="Save trained FPCClassifier to .fpc.npz")
    pc.add_argument("--by-file", action="store_true", help="Group --refs/--train by file stem")
    pc.add_argument("--by-record", action="store_true", help="Group --refs/--train by sequence record ID")
    pc.add_argument("-a", "--alphabet", default="dna", help="Alphabet when using --refs/--train (default: dna)")
    pc.add_argument("-k", "--order", type=int, default=7, help="Context order when using --refs/--train (default: 7)")
    pc.add_argument("--max-prototypes", type=int, default=None, help="Maximum prototypes per class")
    pc.add_argument("--double-strand", action="store_true", help="Also train each reference class on its reverse complement")
    pc.add_argument("--single-strand", action="store_true", help="Disable reverse-complement strand evaluation")
    pc.add_argument("--min-phred", type=int, default=0, help="Mask FASTQ bases with Phred quality < MIN_PHRED")
    pc.add_argument("-o", "--output", default=None, help="Output TSV file (default: stdout)")
    pc.add_argument("-q", "--quiet", action="store_true", help="Suppress stderr summary")
    pc.set_defaults(func=_cmd_classify)

    # -- tree ------------------------------------------------------------
    ptr = sub.add_parser("tree", help="Compute alignment-free cross-entropy distances & Neighbour-Joining Newick tree")
    ptr.add_argument("inputs", nargs="+", help="Input multi-FASTA file, multiple genome FASTA files, or directory")
    ptr.add_argument("-o", "--output", default=None, help="Output Newick (.nwk) file (default: stdout)")
    ptr.add_argument("--matrix", default=None, help="Optional output pairwise distance matrix (.tsv)")
    ptr.add_argument("--html", default=None, help="Optional offline 3-D MDS constellation HTML")
    ptr.add_argument("--clusters", type=int, default=3, help="Cluster count for 3-D HTML colouring (default: 3)")
    ptr.add_argument("-a", "--alphabet", default="dna", help="Alphabet: dna, rna, or protein (default: dna)")
    ptr.add_argument("-k", "--order", type=int, default=6, help="Context order (default: 6)")
    ptr.add_argument("--max-prototypes", type=int, default=None, help="Maximum prototypes per taxon")
    ptr.add_argument("--per-file", action="store_true", help="Treat each input file as one taxon (concatenating contigs)")
    ptr.add_argument("--per-record", action="store_true", help="Treat each sequence record as one taxon")
    ptr.add_argument("--strand-specific", action="store_true", help="Disable strand-invariant symmetrisation")
    ptr.add_argument("--min-phred", type=int, default=0, help="Mask FASTQ bases with Phred quality < MIN_PHRED")
    ptr.add_argument("-q", "--quiet", action="store_true", help="Suppress stderr summary")
    ptr.set_defaults(func=_cmd_tree)

    # -- predict ---------------------------------------------------------
    pp = sub.add_parser("predict", help="Evaluate next-symbol perplexity or generate sequence continuations")
    pp.add_argument("--model", default=None, help="Pre-trained cortex (.fpc.npz)")
    pp.add_argument("--train", default=None, help="Training FASTA/FASTQ file")
    pp.add_argument("--seq", default=None, help="Inline raw sequence string to learn from")
    pp.add_argument("--eval", default=None, help="Held-out FASTA/FASTQ file to evaluate accuracy & perplexity")
    pp.add_argument("--prefix", "--context", dest="prefix", default=None, help="Seed prefix string to continue")
    pp.add_argument("-n", "--length", "--generate", dest="length", type=int, default=100, help="Number of symbols to generate (default: 100)")
    pp.add_argument("--sample", action="store_true", help="Sample stochastically instead of greedy argmax")
    pp.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    pp.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    pp.add_argument("-a", "--alphabet", default="dna", help="Alphabet when using --train/--seq (default: dna)")
    pp.add_argument("-k", "--order", type=int, default=7, help="Context order when using --train/--seq (default: 7)")
    pp.set_defaults(func=_cmd_predict)

    # -- info ------------------------------------------------------------
    pi = sub.add_parser("info", help="Display C kernel status and inspect .fpc.npz archives")
    pi.add_argument("model", nargs="?", default=None, help="Optional .fpc.npz archive to inspect")
    pi.set_defaults(func=_cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"genocortex error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
