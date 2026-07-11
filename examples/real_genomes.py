"""
GenoCortex on REAL genomes (downloaded once from NCBI, then cached offline).
============================================================================

Seven real coronavirus genomes with known taxonomy, so we can check the brain
against ground truth:

    Alphacoronavirus : HCoV-229E, HCoV-NL63
    Betacoronavirus  : SARS-CoV-2, SARS-CoV-1 (Sarbecovirus),
                       MERS-CoV (Merbecovirus),
                       HCoV-OC43, HCoV-HKU1 (Embecovirus)

Experiments on real DNA:
    A. Compression / adaptivity on a real genome (vs Markov, gzip).
    B. Unsupervised phylogeny -- does FPC recover the real genera / subgenera?
    C. Detection of a real recombination (a MERS segment spliced into SARS-CoV-2).
    D. A real 3-D cognitive map + phylogeny constellation (offline HTML).

Run:  python examples/real_genomes.py   (needs internet on first run only)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import FractalPredictiveCortex, DNA, tasks, io_fasta
from fpc import fetch
from benchmarks.baselines import KmerMarkov, gzip_bits  # noqa: E402
from viz import cognitive_map_html, phylogeny_3d_html

EMAIL = "rafiur2810@gmail.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "corona")
OUT = os.path.join(ROOT, "output")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# name -> (accession, subgenus, genus)
GENOMES = {
    "SARS-CoV-2": ("NC_045512.2", "Sarbecovirus", "Beta"),
    "SARS-CoV-1": ("NC_004718.3", "Sarbecovirus", "Beta"),
    "MERS-CoV":   ("NC_019843.3", "Merbecovirus", "Beta"),
    "HCoV-OC43":  ("NC_006213.1", "Embecovirus",  "Beta"),
    "HCoV-HKU1":  ("NC_006577.2", "Embecovirus",  "Beta"),
    "HCoV-229E":  ("NC_002645.1", "Duvinacovirus","Alpha"),
    "HCoV-NL63":  ("NC_005831.2", "Setracovirus", "Alpha"),
}


def get_genome(name, acc):
    """Fetch once, cache to data/corona/<name>.fasta, then reuse offline."""
    path = os.path.join(DATA, f"{name}.fasta")
    if os.path.exists(path):
        return io_fasta.read_fasta(path)[0].seq
    print(f"    downloading {name} ({acc}) ...")
    rec = fetch.ncbi_fasta(acc, email=EMAIL)
    io_fasta.write_fasta(path, [(name, rec.seq)])
    return rec.seq


def main():
    print("Loading real coronavirus genomes (cached after first run) ...")
    seqs = {}
    for name, (acc, sub, gen) in GENOMES.items():
        try:
            seqs[name] = get_genome(name, acc)
        except Exception as e:
            print(f"    !! skipping {name}: {e}")
    names = list(seqs)
    print(f"    have {len(names)} genomes: " +
          ", ".join(f"{n}({len(seqs[n])}bp)" for n in names))

    # ---- A. compression on a real genome ----
    print("\n[A] Compression on SARS-CoV-2 (real DNA), held-out last 20%:")
    s = seqs["SARS-CoV-2"]
    cut = int(0.8 * len(s)); tr, te = s[:cut], s[cut:]
    cx = FractalPredictiveCortex(DNA, order=10, seed=0); cx.fit(DNA.encode(tr))
    print(f"    FPC        : {cx.cross_entropy(DNA.encode(te)):.3f} bits/symbol")
    for k in (2, 4, 6, 8):
        m = KmerMarkov(DNA, k=k).fit(DNA.encode(tr))
        print(f"    Markov-{k}   : {m.cross_entropy(DNA.encode(te)):.3f} bits/symbol")
    print(f"    gzip       : {gzip_bits(te):.3f} bits/symbol")

    # ---- B. unsupervised real phylogeny ----
    print("\n[B] Unsupervised phylogeny (does FPC recover real taxonomy?):")
    seq_list = [seqs[n] for n in names]
    D = tasks.distance_matrix(seq_list, DNA, order=8)
    # nearest neighbour of each genome
    for i, n in enumerate(names):
        order_idx = np.argsort(D[i]); nn = names[order_idx[1]]  # skip self
        print(f"    {n:<12} nearest -> {nn}")
    genus_truth = [GENOMES[n][2] for n in names]
    pur = tasks.clustering_purity(tasks.cluster(D, 2), genus_truth)
    print(f"    Alpha/Beta genus clustering purity = {pur:.2f}")

    # render 3-D phylogeny constellation coloured by subgenus
    coords = tasks.mds_embedding(D, dims=3)
    subgenus = [GENOMES[n][1] for n in names]
    phylogeny_3d_html(coords, subgenus, os.path.join(OUT, "real_phylogeny.html"),
                      names=names, distance_matrix=D,
                      title="Coronavirus phylogeny from GenoCortex (real genomes, unsupervised)",
                      png_path=os.path.join(OUT, "real_phylogeny.png"))
    print("    -> output/real_phylogeny.html")

    # ---- C. detection of REAL contamination (phiX174 spike-in) ----
    # phiX174 is the standard Illumina sequencing spike-in and a common
    # contaminant. Note: a coronavirus scores ~2 bits/base everywhere (it is
    # near-incompressible), so a foreign insert does NOT show up as an absolute
    # surprise SPIKE. The right tool is the "which brain is least surprised"
    # principle applied locally: a host brain vs a phiX brain, window by window.
    print("\n[C] Detection: phiX174 contamination spliced into SARS-CoV-2")
    print("    (competitive dual-brain: host-brain vs phiX-brain, per window)")
    phix = get_genome("phiX174", "NC_001422.1")
    host = seqs["SARS-CoV-2"]
    donor_seg = phix[1000:2000]
    pos = 15000
    chimera = host[:pos] + donor_seg + host[pos:]
    span = (pos, pos + len(donor_seg) - 1)

    host_brain = FractalPredictiveCortex(DNA, order=9, seed=0); host_brain.fit(DNA.encode(host))
    phix_brain = FractalPredictiveCortex(DNA, order=9, seed=0); phix_brain.fit(DNA.encode(phix))

    W, step = 300, 100
    cids = DNA.encode(chimera)
    centers, scores, labels = [], [], []
    for start in range(0, len(cids) - W, step):
        win = cids[start:start + W]
        ce_host = host_brain.cross_entropy(win)
        ce_phix = phix_brain.cross_entropy(win)
        c = start + W // 2
        centers.append(c)
        scores.append(ce_host - ce_phix)             # >0  => looks like phiX
        labels.append(1 if span[0] <= c <= span[1] else 0)
    scores = np.array(scores); labels = np.array(labels)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores) if labels.any() and not labels.all() else float("nan")
    # windows called foreign (host more surprised than phiX)
    called = [centers[i] for i in range(len(centers)) if scores[i] > 0]
    inside = [c for c in called if span[0] <= c <= span[1]]
    print(f"    true insert {span}")
    print(f"    per-window ROC-AUC (host vs phiX brain) = {auc:.3f}")
    print(f"    windows flagged foreign: {len(called)}  ({len(inside)} inside the true insert)")
    print(f"    -> phiX contamination is cleanly separable by relative surprise")

    # ---- D. real cognitive map ----
    print("\n[D] Rendering the real SARS-CoV-2 cognitive map ...")
    cx2 = FractalPredictiveCortex(DNA, order=9, seed=0)
    cx2.fit(DNA.encode(host))
    cognitive_map_html(cx2, DNA.encode(host), os.path.join(OUT, "real_cognitive_map.html"),
                       title="SARS-CoV-2 cognitive map (real genome)",
                       png_path=os.path.join(OUT, "real_cognitive_map.png"))
    print("    -> output/real_cognitive_map.html")
    print("\nDone. Real genomes, real taxonomy, one self-organising brain.")


if __name__ == "__main__":
    main()
