"""
Real PROTEIN study: does the *same* engine work on the 20-letter alphabet?
==========================================================================

The core claim of GenoCortex is that it is alphabet-agnostic. This tests it on
REAL proteins (fetched from NCBI by name), with no code change beyond swapping
DNA -> PROTEIN. Four protein families across several vertebrate species give
ground-truth family labels for:

    * classification  -- FPC (least-surprised model) vs k-mer + kNN,
    * phylogeny        -- FPC cross-entropy distance vs FFP (JSD), family purity.

Run:  python examples/protein_study.py   (downloads on first run, caches offline)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import PROTEIN, tasks, io_fasta, fetch
from benchmarks.baselines import kmer_vector
from benchmarks.afbaselines import ffp_distance_matrix
from viz import phylogeny_3d_html
from sklearn.neighbors import KNeighborsClassifier

EMAIL = "rafiur2810@gmail.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "proteins")
OUT = os.path.join(ROOT, "output")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

FAMILIES = ["cytochrome c", "hemoglobin beta", "myoglobin", "lysozyme"]
SPECIES = ["Homo sapiens", "Mus musculus", "Rattus norvegicus", "Bos taurus",
           "Gallus gallus", "Sus scrofa", "Danio rerio", "Xenopus laevis",
           "Canis lupus familiaris", "Equus caballus"]


def get_protein(fam, sp):
    key = f"{fam}_{sp}".replace(" ", "_")
    path = os.path.join(DATA, f"{key}.fasta")
    if os.path.exists(path):
        return io_fasta.read_fasta(path)[0].seq
    term = f'{fam} AND "{sp}"[Organism] AND refseq[Filter] AND 50:250[Sequence Length]'
    uids = fetch.ncbi_search(term, db="protein", email=EMAIL, retmax=1)
    time.sleep(0.34)
    if not uids:
        raise ValueError("no hit")
    rec = fetch.ncbi_fasta(uids[0], db="protein", email=EMAIL)
    time.sleep(0.34)
    io_fasta.write_fasta(path, [(key, rec.seq)])
    return rec.seq


def main():
    print("Fetching real proteins (cached offline after first run)...")
    seqs, labels, names = [], [], []
    for fam in FAMILIES:
        got = 0
        for sp in SPECIES:
            try:
                s = get_protein(fam, sp)
                if 30 <= len(s) <= 2000:
                    seqs.append(s); labels.append(fam)
                    names.append(f"{fam[:6]}|{sp.split()[0][:4]}"); got += 1
            except Exception as e:
                print(f"   skip {fam} / {sp}: {str(e)[:50]}")
        print(f"   {fam}: {got} sequences")
    n = len(seqs)
    print(f"  total {n} real proteins across {len(set(labels))} families "
          f"(lengths {min(len(s) for s in seqs)}-{max(len(s) for s in seqs)} aa)")
    if n < 8:
        print("  too few proteins fetched; aborting."); return

    # ---- classification (family) ----
    from collections import defaultdict
    buck = defaultdict(list)
    for s, l in zip(seqs, labels):
        buck[l].append(s)
    Xtr, ytr, Xte, yte = [], [], [], []
    for l, ss in buck.items():
        half = max(1, len(ss) // 2)
        Xtr += ss[:half]; ytr += [l] * half
        Xte += ss[half:]; yte += [l] * (len(ss) - half)
    print("\n[classification] protein family (FPC vs 3-mer+1NN):")
    if len(set(yte)) >= 2 and Xte:
        clf = tasks.FPCClassifier(PROTEIN, order=4).fit(Xtr, ytr)
        acc_fpc = tasks.classification_accuracy(clf.predict(Xte), yte)
        k = 3
        Vtr = np.array([kmer_vector(s, PROTEIN, k) for s in Xtr])
        Vte = np.array([kmer_vector(s, PROTEIN, k) for s in Xte])
        knn = KNeighborsClassifier(1, metric="cosine").fit(Vtr, ytr)
        acc_knn = float(np.mean(knn.predict(Vte) == np.array(yte)))
        print(f"   FPC={acc_fpc:.3f}   3-mer+1NN={acc_knn:.3f}   (chance={1/len(set(labels)):.2f})")

    # ---- phylogeny / family clustering ----
    print("\n[phylogeny] family clustering purity (FPC vs FFP):")
    D_fpc = tasks.distance_matrix(seqs, PROTEIN, order=4)
    D_ffp = ffp_distance_matrix(seqs, PROTEIN, k=2)
    n_fam = len(set(labels))
    pur_fpc = tasks.clustering_purity(tasks.cluster(D_fpc, n_fam), labels)
    pur_ffp = tasks.clustering_purity(tasks.cluster(D_ffp, n_fam), labels)
    print(f"   FPC={pur_fpc:.3f}   FFP(JSD,k=2)={pur_ffp:.3f}")

    coords = tasks.mds_embedding(D_fpc, dims=3)
    phylogeny_3d_html(coords, labels, os.path.join(OUT, "protein_phylogeny.html"),
                      names=names, distance_matrix=D_fpc,
                      title=f"{n} real proteins, {n_fam} families "
                            f"(GenoCortex on the 20-letter alphabet) - purity {pur_fpc:.2f}",
                      png_path=os.path.join(OUT, "protein_phylogeny.png"))
    print("   -> output/protein_phylogeny.html")
    print("\nSame engine, DNA -> PROTEIN, one-line change. Alphabet-agnostic on real data.")


if __name__ == "__main__":
    main()
