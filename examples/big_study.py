"""
A larger real-genome study: ~30 viral genomes across several families.
=======================================================================

Ground-truth family labels let us score two things honestly on real data:

    1. Family classification (train/test split) -- FPC vs k-mer + kNN/logistic.
    2. Unsupervised phylogeny -- do FPC cross-entropy distances recover the
       real family structure? (family clustering purity, MDS 3-D, dendrogram)

Accessions are RefSeq; any that fail to download are skipped, and only families
with enough successful members are used for the supervised part. Genomes are
subsampled to a fixed central window so the comparison is length-fair and fast.

Run:  python examples/big_study.py   (downloads on first run, caches offline)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import DNA, tasks, io_fasta, fetch
from benchmarks.baselines import kmer_vector
from benchmarks.afbaselines import mash_distance_matrix, ffp_distance_matrix
from viz import phylogeny_3d_html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cosine

EMAIL = "rafiur2810@gmail.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "viruses")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

WINDOW = 6000  # central bp used from each genome (length-fair, fast)

# name : (accession, family)
CATALOG = {
    # Coronaviridae (~27-31 kb)
    "SARS-CoV-2": ("NC_045512.2", "Coronaviridae"),
    "SARS-CoV-1": ("NC_004718.3", "Coronaviridae"),
    "MERS-CoV":   ("NC_019843.3", "Coronaviridae"),
    "HCoV-OC43":  ("NC_006213.1", "Coronaviridae"),
    "HCoV-HKU1":  ("NC_006577.2", "Coronaviridae"),
    "HCoV-229E":  ("NC_002645.1", "Coronaviridae"),
    "HCoV-NL63":  ("NC_005831.2", "Coronaviridae"),
    # Flaviviridae (~10-11 kb)
    "Zika":       ("NC_012532.1", "Flaviviridae"),
    "Dengue-1":   ("NC_001477.1", "Flaviviridae"),
    "Dengue-2":   ("NC_001474.2", "Flaviviridae"),
    "Dengue-3":   ("NC_001475.2", "Flaviviridae"),
    "Dengue-4":   ("NC_002640.1", "Flaviviridae"),
    "WestNile":   ("NC_009942.1", "Flaviviridae"),
    "YellowFever":("NC_002031.1", "Flaviviridae"),
    "JEV":        ("NC_001437.1", "Flaviviridae"),
    # Filoviridae (~19 kb)
    "Ebola-Zaire":     ("NC_002549.1", "Filoviridae"),
    "Ebola-Sudan":     ("NC_006432.1", "Filoviridae"),
    "Ebola-Bundibugyo":("NC_014373.1", "Filoviridae"),
    "Ebola-Reston":    ("NC_004161.1", "Filoviridae"),
    "Marburg":         ("NC_001608.3", "Filoviridae"),
    # Picornaviridae (~7-8 kb)
    "Poliovirus":  ("NC_002058.3", "Picornaviridae"),
    "HepatitisA":  ("NC_001489.1", "Picornaviridae"),
    "EMCV":        ("NC_001479.1", "Picornaviridae"),
    "Rhinovirus":  ("NC_001617.1", "Picornaviridae"),
    "CoxsackieA":  ("NC_001612.1", "Picornaviridae"),
    # Retroviridae (~9-10 kb)
    "HIV-1":       ("NC_001802.1", "Retroviridae"),
    "HIV-2":       ("NC_001722.1", "Retroviridae"),
    "SIV":         ("NC_001549.1", "Retroviridae"),
}

FAMILY_COLORS = {}  # filled after we know which families survive


def get(name, acc):
    path = os.path.join(DATA, f"{name}.fasta")
    if os.path.exists(path):
        return io_fasta.read_fasta(path)[0].seq
    rec = fetch.ncbi_fasta(acc, email=EMAIL)
    io_fasta.write_fasta(path, [(name, rec.seq)])
    return rec.seq


def central(seq, w=WINDOW):
    if len(seq) <= w:
        return seq
    mid = len(seq) // 2
    return seq[mid - w // 2: mid + w // 2]


def main():
    print(f"Downloading/loading up to {len(CATALOG)} genomes (cached offline after run 1)...")
    seqs, fam = {}, {}
    for name, (acc, family) in CATALOG.items():
        try:
            s = central(get(name, acc))
            if len(s) >= WINDOW // 2:
                seqs[name] = s; fam[name] = family
        except Exception as e:
            print(f"    skip {name} ({acc}): {str(e)[:60]}")
    names = list(seqs)
    from collections import Counter
    counts = Counter(fam.values())
    if len(names) < 6 or len(counts) < 2:
        raise RuntimeError(
            f"Too few viral genomes loaded ({len(names)}); check network or data/viruses cache."
        )
    print(f"  loaded {len(names)} genomes across families: {dict(counts)}")

    # ---------- 2. unsupervised phylogeny (all genomes) ----------
    print("\n[phylogeny] all-vs-all FPC cross-entropy distance ...")
    seq_list = [seqs[n] for n in names]
    D = tasks.distance_matrix(seq_list, DNA, order=7)
    fam_labels = [fam[n] for n in names]
    n_fam = len(set(fam_labels))
    pur_fpc = tasks.clustering_purity(tasks.cluster(D, n_fam), fam_labels)

    # standard alignment-free baselines a reviewer expects
    k = 5
    V = np.array([kmer_vector(s, DNA, k) for s in seq_list])
    Dk = np.zeros_like(D)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Dk[i, j] = Dk[j, i] = cosine(V[i], V[j])
    D_mash = mash_distance_matrix(seq_list, k=14, sketch_size=500)
    D_ffp = ffp_distance_matrix(seq_list, DNA, k=6)
    pur_km = tasks.clustering_purity(tasks.cluster(Dk, n_fam), fam_labels)
    pur_mash = tasks.clustering_purity(tasks.cluster(D_mash, n_fam), fam_labels)
    pur_ffp = tasks.clustering_purity(tasks.cluster(D_ffp, n_fam), fam_labels)
    print(f"  family clustering purity ({n_fam} clusters):")
    print(f"    FPC (cross-entropy) = {pur_fpc:.3f}")
    print(f"    FFP  (JSD)          = {pur_ffp:.3f}")
    print(f"    Mash (MinHash)      = {pur_mash:.3f}")
    print(f"    {k}-mer cosine       = {pur_km:.3f}")

    # 3-D constellation coloured by family
    coords = tasks.mds_embedding(D, dims=3)
    phylogeny_3d_html(coords, fam_labels, os.path.join(OUT, "big_phylogeny.html"),
                      names=names, distance_matrix=D,
                      title=f"{len(names)} viral genomes, {n_fam} families "
                            f"(GenoCortex, unsupervised) - purity {pur_fpc:.2f}",
                      png_path=os.path.join(OUT, "big_phylogeny.png"))
    print("  -> output/big_phylogeny.html")

    # dendrogram figure
    iu = np.triu_indices(len(names), k=1)
    Z = linkage(D[iu], method="average")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    dendrogram(Z, labels=[f"{fam[n][:5]}|{n}" for n in names], ax=ax,
               leaf_rotation=90, color_threshold=0.6 * D[iu].max())
    ax.set_title(f"Unsupervised phylogeny of {len(names)} viral genomes "
                 f"(FPC cross-entropy; family purity {pur_fpc:.2f})")
    ax.set_ylabel("cross-entropy distance")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_big_dendrogram.png")); plt.close(fig)
    print("  -> output/figures/fig7_big_dendrogram.png")

    # ---------- 1. family classification (families with >=4 members) ----------
    print("\n[classification] family ID (families with >=4 genomes) ...")
    big_fams = [f for f, c in counts.items() if c >= 4]
    members = {f: [n for n in names if fam[n] == f] for f in big_fams}
    Xtr, ytr, Xte, yte = [], [], [], []
    for f, ms in members.items():
        for i, n in enumerate(ms):
            (Xtr if i % 2 == 0 else Xte).append(seqs[n])
            (ytr if i % 2 == 0 else yte).append(f)
    if len(set(yte)) >= 2:
        clf = tasks.FPCClassifier(DNA, order=7).fit(Xtr, ytr)
        acc_fpc = tasks.classification_accuracy(clf.predict(Xte), yte)
        Vtr = np.array([kmer_vector(s, DNA, 4) for s in Xtr])
        Vte = np.array([kmer_vector(s, DNA, 4) for s in Xte])
        knn = KNeighborsClassifier(1, metric="cosine").fit(Vtr, ytr)
        lr = LogisticRegression(max_iter=3000).fit(Vtr, ytr)
        acc_knn = float(np.mean(knn.predict(Vte) == np.array(yte)))
        acc_lr = float(np.mean(lr.predict(Vte) == np.array(yte)))
        print(f"  families: {big_fams}")
        print(f"  test accuracy:  FPC={acc_fpc:.3f}  4mer+1NN={acc_knn:.3f}  4mer+logistic={acc_lr:.3f}")

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        bars = {"FPC": acc_fpc, "4-mer+1NN": acc_knn, "4-mer+logistic": acc_lr}
        ax.bar(list(bars), list(bars.values()), color=["#2a78d6", "#1baf7a", "#4a3aa7"])
        for i, v in enumerate(bars.values()):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.08); ax.set_ylabel("test accuracy")
        ax.set_title(f"Viral family classification ({len(big_fams)} families, real genomes)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_family_classification.png")); plt.close(fig)
        print("  -> output/figures/fig6_family_classification.png")
    else:
        print("  not enough test genomes for classification")

    print("\nDone. A larger, honest, real-genome study.")


if __name__ == "__main__":
    main()
