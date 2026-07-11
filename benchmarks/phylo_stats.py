"""
Statistically-honest phylogeny comparison: FPC vs standard alignment-free tools.
================================================================================

Instead of one number from one tree, we recover many INDEPENDENT random
phylogenies (varying depth of divergence, source order, and composition), score
each method's distance--vs--true-patristic-distance correlation per tree, and
report:

  * mean correlation with a 95% bootstrap confidence interval per method,
  * a paired Wilcoxon signed-rank test between FPC and each baseline,

so a claim like "FPC is competitive with FFP" is backed by a distribution and a
p-value, not a single run. This is the rigor a methods reviewer expects.

Run:  python benchmarks/phylo_stats.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import DNA, tasks, datasets
from benchmarks.baselines import kmer_vector, nrc_nccd_distance_matrices
from benchmarks.afbaselines import mash_distance_matrix, ffp_distance_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

N_TREES = 25
C_FPC = "#2a78d6"; C_FFP = "#1baf7a"; C_MASH = "#eb6834"; C_KM = "#898781"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "figure.facecolor": "white"})


def _kmer_cos(seqs, k=5):
    V = np.array([kmer_vector(s, DNA, k) for s in seqs])
    n = len(seqs); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = cosine(V[i], V[j])
    return D


def _corr_to_truth(D, names, truth):
    n = len(names)
    tr = [truth[(names[i], names[j])] for i in range(n) for j in range(i + 1, n)]
    es = [D[i, j] for i in range(n) for j in range(i + 1, n)]
    return float(np.corrcoef(tr, es)[0, 1])


def bootstrap_ci(x, iters=5000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    methods = ["FPC", "NRC", "NCCD", "FFP", "Mash", "5-mer cos"]
    corrs = {m: [] for m in methods}

    for t in range(N_TREES):
        rng = np.random.default_rng(1000 + t)
        branch = float(rng.uniform(0.02, 0.08))
        order = int(rng.integers(1, 4))
        conc = float(rng.uniform(0.15, 0.30))
        phy = datasets.phylogeny(root_length=4000, depth=3, branch_sub=branch,
                                 order=order, concentration=conc, seed=1000 + t)
        names = list(phy.leaves); seqs = [phy.leaves[n] for n in names]
        truth = phy.truth_distance

        corrs["FPC"].append(_corr_to_truth(tasks.distance_matrix(seqs, DNA, order=6), names, truth))
        d_nrc, d_nccd = nrc_nccd_distance_matrices(seqs, DNA, k=8)
        corrs["NRC"].append(_corr_to_truth(d_nrc, names, truth))
        corrs["NCCD"].append(_corr_to_truth(d_nccd, names, truth))
        corrs["FFP"].append(_corr_to_truth(ffp_distance_matrix(seqs, DNA, k=6), names, truth))
        corrs["Mash"].append(_corr_to_truth(mash_distance_matrix(seqs, k=12, sketch_size=500), names, truth))
        corrs["5-mer cos"].append(_corr_to_truth(_kmer_cos(seqs), names, truth))
        print(f"  tree {t+1}/{N_TREES}  (branch={branch:.3f}, order={order})", end="\r")
    print()

    # summary
    summary = {}
    for m in methods:
        x = np.array(corrs[m])
        lo, hi = bootstrap_ci(x)
        summary[m] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
                      "ci95": [lo, hi], "n": len(x)}
    # paired tests: FPC vs each baseline
    tests = {}
    for m in ["NRC", "NCCD", "FFP", "Mash", "5-mer cos"]:
        stat, p = wilcoxon(corrs["FPC"], corrs[m])
        diff = np.mean(np.array(corrs["FPC"]) - np.array(corrs[m]))
        tests[f"FPC_vs_{m}"] = {"mean_diff": float(diff), "wilcoxon_p": float(p)}

    with open(os.path.join(OUT, "phylo_stats.json"), "w") as fh:
        json.dump({"summary": summary, "paired_tests": tests, "n_trees": N_TREES,
                   "raw_correlations": {m: corrs[m] for m in methods}}, fh, indent=2)

    print("\n=== Phylogeny correlation with true patristic distance "
          f"(mean over {N_TREES} independent trees) ===")
    for m in methods:
        s = summary[m]
        print(f"  {m:<10}: {s['mean']:.3f}  (95% CI {s['ci95'][0]:.3f}-{s['ci95'][1]:.3f})")
    print("\n=== Paired Wilcoxon: FPC vs baseline (negative mean_diff => FPC lower) ===")
    for k, v in tests.items():
        sig = "significant" if v["wilcoxon_p"] < 0.05 else "n.s."
        print(f"  {k:<18}: mean_diff={v['mean_diff']:+.3f}  p={v['wilcoxon_p']:.4f} ({sig})")

    # figure: boxplot of per-tree correlations
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    data = [corrs[m] for m in methods]
    palette = {"FPC": C_FPC, "NRC": "#4a3aa7", "NCCD": "#9085e9",
               "FFP": C_FFP, "Mash": C_MASH, "5-mer cos": C_KM}
    colors = [palette[m] for m in methods]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.6, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_ylabel("correlation with true patristic distance")
    ax.set_title(f"Phylogeny recovery across {N_TREES} independent trees (higher = better)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_phylo_stats.png")); plt.close(fig)
    print(f"\nFigure -> {os.path.join(FIG, 'fig8_phylo_stats.png')}")


if __name__ == "__main__":
    main()
