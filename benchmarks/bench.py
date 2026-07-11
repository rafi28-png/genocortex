"""
Benchmark suite: Fractal Predictive Cortex vs. respected baselines.
===================================================================

These experiments are designed to be *honest*, not flattering. On data that a
fixed-order Markov model generated, a matched fixed-order Markov model is the
oracle and will win -- so we do not claim wins there. Instead each experiment
probes a property FPC is actually supposed to have:

    1. Compression   -- ADAPTIVITY: on heterogeneous / limited data, no single
                        fixed order k is best; the cortex adapts. Also vs gzip.
    2. Prediction    -- next-symbol accuracy learning curve vs best Markov.
    3. Classification-- ROBUSTNESS on short reads vs k-mer + kNN / logistic.
    4. Phylogeny     -- distance quality vs true patristic distance (vs k-mer).
    5. Detection     -- SENSITIVITY to a *cryptic* insert (a compositional
                        cousin that shares k-mers), where raw novelty is blind.

Run:  python benchmarks/bench.py
Writes figures to output/figures/ and a JSON summary to output/benchmarks.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpc import FractalPredictiveCortex, DNA, tasks, datasets
from baselines import KmerMarkov, gzip_bits, kmer_vector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

C_FPC = "#2a78d6"; C_MARKOV = "#eb6834"; C_GZIP = "#898781"
C_A = "#1baf7a"; C_B = "#4a3aa7"; C_C = "#e34948"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


# ===================================================================== #
def _split_blocks(genome: str, n_blocks: int, block: int, frac: float = 0.7):
    """Split a block-structured genome into same-mixture train / test."""
    tr, te = [], []
    for i in range(n_blocks):
        seg = genome[i * block:(i + 1) * block]
        cut = int(frac * len(seg))
        tr.append(seg[:cut]); te.append(seg[cut:])
    return "".join(tr), "".join(te)


def bench_compression() -> dict:
    print("[1] compression: adaptivity on heterogeneous & limited data ...")
    scenarios = {}

    # (a) heterogeneous genome: no single k is right everywhere
    block = 1600
    het = datasets.heterogeneous_genome(DNA, seed=1, block=block)
    scenarios["heterogeneous\n(mixed order + repeats)"] = _split_blocks(het, 5, block)

    # (b) genuinely high-order source, but LIMITED training data -> fixed high-k
    #     overfits (sparse counts), low-k underfits; FPC adapts.
    hi = datasets.markov_sequence(9000, DNA, order=5, concentration=0.2, seed=21)
    scenarios["high-order (5),\nlimited data"] = (hi[:4000], hi[4000:6000])

    # (c) an easy control where a matched Markov SHOULD win (we report it honestly)
    lo = datasets.markov_sequence(16000, DNA, order=2, concentration=0.1, seed=22)
    scenarios["low-order (2)\n[Markov oracle]"] = (lo[:12000], lo[12000:])

    ks = [2, 4, 6, 8]
    results = {}
    for name, (tr, te) in scenarios.items():
        row = {}
        cx = FractalPredictiveCortex(DNA, order=12, seed=0)
        cx.fit(DNA.encode(tr))
        row["FPC"] = cx.cross_entropy(DNA.encode(te))
        for k in ks:
            row[f"Markov-{k}"] = KmerMarkov(DNA, k=k).fit(DNA.encode(tr)).cross_entropy(DNA.encode(te))
        row["gzip"] = gzip_bits(te)
        row["best_fixed_markov"] = min(row[f"Markov-{k}"] for k in ks)
        results[name] = row

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = list(results); x = np.arange(len(labels))
    series = ["FPC", "Markov-2", "Markov-4", "Markov-6", "Markov-8", "gzip"]
    colors = [C_FPC, "#f6c3ac", "#f0a884", "#eb8a5c", "#c14a1c", C_GZIP]
    width = 0.13
    for i, s in enumerate(series):
        vals = [results[l][s] for l in labels]
        ax.bar(x + (i - len(series) / 2) * width + width / 2, vals, width, label=s, color=colors[i])
    ax.axhline(2.0, ls="--", lw=1, color="#333", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("bits / symbol on held-out  (lower = better)")
    ax.set_title("Compression: one adaptive model vs. having to pick k")
    ax.legend(ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_compression.png")); plt.close(fig)
    return results


# ===================================================================== #
def bench_prediction() -> dict:
    print("[2] prediction: learning curve vs best Markov ...")
    full = datasets.markov_sequence(40000, DNA, order=3, concentration=0.12, seed=5)
    test = full[36000:]
    sizes = [1000, 2000, 4000, 8000, 16000, 32000]
    fpc_acc, mk_acc = [], []
    for n in sizes:
        cx = FractalPredictiveCortex(DNA, order=10, seed=0); cx.fit(DNA.encode(full[:n]))
        fpc_acc.append(tasks.evaluate_prediction(cx, test)["accuracy"])
        mk_acc.append(KmerMarkov(DNA, k=6).fit(DNA.encode(full[:n])).accuracy(DNA.encode(test)))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(sizes, fpc_acc, "-o", color=C_FPC, lw=2, label="Fractal Predictive Cortex")
    ax.plot(sizes, mk_acc, "-s", color=C_MARKOV, lw=2, label="Markov-6")
    ax.axhline(0.25, ls="--", lw=1, color="#333", alpha=0.6)
    ax.text(sizes[0], 0.26, "chance (0.25)", fontsize=8.5, color="#333")
    ax.set_xscale("log"); ax.set_xlabel("training length (symbols)")
    ax.set_ylabel("next-base accuracy"); ax.set_title("Prediction: accuracy vs. training data")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_prediction.png")); plt.close(fig)
    return {"sizes": sizes, "fpc": fpc_acc, "markov6": mk_acc}


# ===================================================================== #
def bench_classification() -> dict:
    print("[3] classification: robustness on short reads ...")
    n_species = 6
    trans = datasets.species_transitions(n_species, DNA, order=2, concentration=0.18, seed=31)
    # training: long references per species
    Xtr, ytr = [], []
    for sp in range(n_species):
        for r in range(3):
            Xtr.append(datasets.markov_sequence(3000, DNA, order=2, seed=1000 + sp * 10 + r,
                                                transition=trans[sp]))
            ytr.append(f"sp{sp}")
    clf = tasks.FPCClassifier(DNA, order=7).fit(Xtr, ytr)

    k = 4
    Vtr = np.array([kmer_vector(s, DNA, k) for s in Xtr])
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(Vtr, ytr)
    lr = LogisticRegression(max_iter=3000).fit(Vtr, ytr)

    lengths = [150, 300, 600, 1200, 2400]
    fpc_c, knn_c, lr_c = [], [], []
    for L in lengths:
        Xte, yte = [], []
        for sp in range(n_species):
            for r in range(6):
                Xte.append(datasets.markov_sequence(L, DNA, order=2, seed=9000 + sp * 20 + r,
                                                    transition=trans[sp]))
                yte.append(f"sp{sp}")
        fpc_c.append(tasks.classification_accuracy(clf.predict(Xte), yte))
        Vte = np.array([kmer_vector(s, DNA, k) for s in Xte])
        knn_c.append(float(np.mean(knn.predict(Vte) == np.array(yte))))
        lr_c.append(float(np.mean(lr.predict(Vte) == np.array(yte))))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(lengths, fpc_c, "-o", color=C_FPC, lw=2, label="FPC (cross-entropy)")
    ax.plot(lengths, knn_c, "-s", color=C_A, lw=2, label=f"{k}-mer + 1-NN")
    ax.plot(lengths, lr_c, "-^", color=C_B, lw=2, label=f"{k}-mer + logistic")
    ax.axhline(1 / n_species, ls="--", lw=1, color="#333", alpha=0.6)
    ax.text(lengths[0], 1 / n_species + 0.02, f"chance ({1/n_species:.2f})", fontsize=8.5, color="#333")
    ax.set_xscale("log"); ax.set_xlabel("query read length (bp)")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Classifying {n_species} species from short reads")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_classification.png")); plt.close(fig)
    return {"lengths": lengths, "fpc": fpc_c, "knn": knn_c, "logistic": lr_c}


# ===================================================================== #
def bench_phylogeny() -> dict:
    print("[4] phylogeny: FPC distance vs k-mer cosine ...")
    phy = datasets.phylogeny(root_length=5000, depth=3, branch_sub=0.05, order=2, seed=13)
    names = list(phy.leaves); seqs = [phy.leaves[n] for n in names]; n = len(names)

    D_fpc = tasks.distance_matrix(seqs, DNA, order=6)
    k = 5
    V = np.array([kmer_vector(s, DNA, k) for s in seqs])
    D_km = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D_km[i, j] = D_km[j, i] = cosine(V[i], V[j])

    tr, e_fpc, e_km = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            tr.append(phy.truth_distance[(names[i], names[j])])
            e_fpc.append(D_fpc[i, j]); e_km.append(D_km[i, j])
    r_fpc = float(np.corrcoef(tr, e_fpc)[0, 1]); r_km = float(np.corrcoef(tr, e_km)[0, 1])
    true_clade = [nm[1] for nm in names]
    pur_fpc = tasks.clustering_purity(tasks.cluster(D_fpc, 2), true_clade)
    pur_km = tasks.clustering_purity(tasks.cluster(D_km, 2), true_clade)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    ax1.scatter(tr, e_fpc, color=C_FPC, s=28)
    ax1.set_title(f"FPC cross-entropy distance\nr = {r_fpc:.3f}, clade purity = {pur_fpc:.2f}")
    ax1.set_xlabel("true patristic distance"); ax1.set_ylabel("recovered distance")
    ax2.scatter(tr, e_km, color=C_MARKOV, s=28)
    ax2.set_title(f"{k}-mer cosine distance\nr = {r_km:.3f}, clade purity = {pur_km:.2f}")
    ax2.set_xlabel("true patristic distance"); ax2.set_ylabel("recovered distance")
    fig.suptitle("Phylogeny recovery (unsupervised, alignment-free)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_phylogeny.png"), bbox_inches="tight"); plt.close(fig)
    return {"corr_fpc": r_fpc, "corr_kmer": r_km, "purity_fpc": pur_fpc, "purity_kmer": pur_km}


# ===================================================================== #
def _roc_for_insert(host_trans, donor_trans, k_novelty=8, order=9, seed=0):
    host = datasets.markov_sequence(8000, DNA, order=2, seed=17, transition=host_trans)
    donor = datasets.markov_sequence(600, DNA, order=2, seed=3, transition=donor_trans)
    chimera, (a, b) = datasets.insert_foreign(host, donor, position=4000, seed=seed)
    labels = np.zeros(len(chimera), dtype=int); labels[a:b + 1] = 1

    ref = FractalPredictiveCortex(DNA, order=order, seed=0); ref.fit(DNA.encode(host))
    fpc_score = ref.score(DNA.encode(chimera)).smoothed(window=40)
    fpc_score = np.nan_to_num(fpc_score, nan=np.nanmin(fpc_score))

    ref_ids = DNA.encode(host); ref_set = set()
    for t in range(len(ref_ids) - k_novelty + 1):
        w = ref_ids[t:t + k_novelty]
        if (w >= DNA.size).any():
            continue
        ref_set.add(bytes(int(x) for x in w))
    cids = DNA.encode(chimera); novel = np.zeros(len(chimera))
    for t in range(len(cids) - k_novelty + 1):
        w = cids[t:t + k_novelty]
        if (w >= DNA.size).any():
            continue
        if bytes(int(x) for x in w) not in ref_set:
            novel[t + k_novelty - 1] = 1.0
    novel_score = np.convolve(novel, np.ones(40) / 40, mode="same")

    f1, t1, _ = roc_curve(labels, fpc_score); a1 = auc(f1, t1)
    f2, t2, _ = roc_curve(labels, novel_score); a2 = auc(f2, t2)
    return (f1, t1, a1), (f2, t2, a2)


def bench_detection() -> dict:
    print("[5] detection: cryptic (compositional cousin) insert ...")
    trans = datasets.species_transitions(1, DNA, order=2, concentration=0.15, seed=41)[0]
    # unrelated donor (easy) and a cryptic cousin (hard: shares k-mers)
    unrelated = datasets.species_transitions(1, DNA, order=2, concentration=0.45, seed=888)[0]
    cousin = datasets.related_transition(trans, strength=0.30, concentration=0.15, seed=7)

    (f1e, t1e, a1e), (f2e, t2e, a2e) = _roc_for_insert(trans, unrelated)
    (f1h, t1h, a1h), (f2h, t2h, a2h) = _roc_for_insert(trans, cousin)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (fpcr, novr), ttl in [
        (ax1, ((f1e, t1e, a1e), (f2e, t2e, a2e)), "Unrelated insert (easy)"),
        (ax2, ((f1h, t1h, a1h), (f2h, t2h, a2h)), "Cryptic cousin insert (hard)"),
    ]:
        ax.plot(fpcr[0], fpcr[1], color=C_FPC, lw=2.4, label=f"FPC surprise (AUC={fpcr[2]:.3f})")
        ax.plot(novr[0], novr[1], color=C_MARKOV, lw=2.0, label=f"8-mer novelty (AUC={novr[2]:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title(ttl); ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Foreign-insert detection (per-position ROC)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_detection_roc.png"), bbox_inches="tight"); plt.close(fig)
    return {"easy": {"auc_fpc": float(a1e), "auc_novelty": float(a2e)},
            "cryptic": {"auc_fpc": float(a1h), "auc_novelty": float(a2h)}}


# ===================================================================== #
def main():
    results = {}
    results["compression"] = bench_compression()
    results["prediction"] = bench_prediction()
    results["classification"] = bench_classification()
    results["phylogeny"] = bench_phylogeny()
    results["detection"] = bench_detection()

    with open(os.path.join(OUT, "benchmarks.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 66)
    print("SUMMARY (honest -- Markov oracle wins on matched data, as it should)")
    print("=" * 66)
    print("\n-- Compression (bits/symbol, held-out; lower better) --")
    for name, row in results["compression"].items():
        tag = name.replace("\n", " ")
        print(f"  {tag:<34}: FPC={row['FPC']:.3f}  bestFixedMarkov={row['best_fixed_markov']:.3f}  gzip={row['gzip']:.3f}")
    print("\n-- Classification accuracy on short reads --")
    c = results["classification"]
    for i, L in enumerate(c["lengths"]):
        print(f"  {L:>5} bp : FPC={c['fpc'][i]:.3f}  kmer+1NN={c['knn'][i]:.3f}  kmer+logistic={c['logistic'][i]:.3f}")
    print("\n-- Phylogeny (corr with truth / clade purity) --")
    p = results["phylogeny"]
    print(f"  FPC          : r={p['corr_fpc']:.3f}  purity={p['purity_fpc']:.2f}")
    print(f"  k-mer cosine : r={p['corr_kmer']:.3f}  purity={p['purity_kmer']:.2f}")
    print("\n-- Detection ROC-AUC --")
    d = results["detection"]
    print(f"  easy    insert: FPC={d['easy']['auc_fpc']:.3f}   8-mer novelty={d['easy']['auc_novelty']:.3f}")
    print(f"  cryptic insert: FPC={d['cryptic']['auc_fpc']:.3f}   8-mer novelty={d['cryptic']['auc_novelty']:.3f}")
    print(f"\nFigures written to {FIG}")


if __name__ == "__main__":
    main()
