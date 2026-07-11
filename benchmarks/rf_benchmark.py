"""
Community-standard phylogeny benchmark: Robinson-Foulds on real mitogenomes.
============================================================================

The metric reviewers expect for genome phylogeny (as used by AFproject): build a
tree from each method's distances by Neighbor-Joining and score its topological
distance to a trusted reference tree by the (normalized) Robinson-Foulds metric
(lower = better). Dataset: real mammalian mitochondrial genomes (~16-17 kb) from
NCBI RefSeq. Reference topology: established mammalian phylogeny (Boreoeutheria =
Euarchontoglires + Laurasiatheria, sister to Afrotheria; Murphy et al. 2001) —
resolved where uncontroversial, polytomies where genuinely debated
(Laurasiatheria internal, Cetartiodactyla internal), rooted on a marsupial
outgroup.

Compares FPC vs Mash vs FFP vs 5-mer cosine.
Run:  python benchmarks/rf_benchmark.py   (downloads once, caches offline)
"""
import io
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dendropy
from dendropy.calculate import treecompare
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import DNA, tasks, io_fasta, fetch
from benchmarks.baselines import kmer_vector, nrc_nccd_distance_matrices
from benchmarks.afbaselines import mash_distance_matrix, ffp_distance_matrix

EMAIL = "rafiur2810@gmail.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "mito")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# label -> species (for NCBI search)
SPECIES = {
    "Human": "Homo sapiens", "Chimp": "Pan troglodytes", "Gorilla": "Gorilla gorilla",
    "Orangutan": "Pongo abelii", "Macaque": "Macaca mulatta",
    "Mouse": "Mus musculus", "Rat": "Rattus norvegicus", "GuineaPig": "Cavia porcellus",
    "Rabbit": "Oryctolagus cuniculus",
    "Dog": "Canis lupus familiaris", "Cat": "Felis catus",
    "Cow": "Bos taurus", "Pig": "Sus scrofa", "Whale": "Balaenoptera physalus",
    "Horse": "Equus caballus", "Bat": "Pteropus vampyrus",
    "Elephant": "Loxodonta africana",
    "Opossum": "Monodelphis domestica",
}

# Reference topology, built from parts so parentheses are guaranteed balanced.
# Resolved only where accepted; polytomies where genuinely debated.
_PRIMATES = "((((Human,Chimp),Gorilla),Orangutan),Macaque)"
_GLIRES = "(((Mouse,Rat),GuineaPig),Rabbit)"
_EUARCH = f"({_PRIMATES},{_GLIRES})"
_LAURASIA = "((Dog,Cat),(Cow,Pig,Whale),Horse,Bat)"      # internal polytomy (debated)
_BOREO = f"({_EUARCH},{_LAURASIA})"
_PLACENTALIA = f"({_BOREO},Elephant)"                    # Afrotheria = Elephant
REFERENCE = f"({_PLACENTALIA},Opossum);"                 # root on marsupial outgroup


def get_mito(label, species):
    path = os.path.join(DATA, f"{label}.fasta")
    if os.path.exists(path):
        return io_fasta.read_fasta(path)[0].seq
    term = (f'{species}[Organism] AND mitochondrion[Title] AND '
            f'"complete genome"[Title] AND refseq[Filter]')
    uids = fetch.ncbi_search(term, db="nucleotide", email=EMAIL, retmax=1)
    time.sleep(0.34)
    if not uids:
        raise ValueError("no hit")
    rec = fetch.ncbi_fasta(uids[0], db="nucleotide", email=EMAIL)
    time.sleep(0.34)
    io_fasta.write_fasta(path, [(label, rec.seq)])
    return rec.seq


def kmer_cos_matrix(seqs, k=5):
    V = np.array([kmer_vector(s, DNA, k) for s in seqs])
    n = len(seqs); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = cosine(V[i], V[j])
    return D


def nj_tree_from_matrix(D, labels, taxa):
    """Build a Neighbor-Joining tree from a distance matrix via dendropy."""
    # dendropy reads a full CSV matrix with a header row of taxon labels.
    buf = io.StringIO()
    buf.write("," + ",".join(labels) + "\n")
    for i, lab in enumerate(labels):
        buf.write(lab + "," + ",".join(f"{D[i, j]:.6f}" for j in range(len(labels))) + "\n")
    buf.seek(0)
    pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=buf, taxon_namespace=taxa,
                                                       delimiter=",")
    return pdm.nj_tree()


def main():
    print("Fetching real mammalian mitogenomes (cached offline after first run)...")
    seqs, labels = {}, []
    for label, sp in SPECIES.items():
        try:
            s = get_mito(label, sp)
            if len(s) > 10000:
                seqs[label] = s; labels.append(label)
            else:
                print(f"   skip {label}: short ({len(s)})")
        except Exception as e:
            print(f"   skip {label} ({sp}): {str(e)[:50]}")
    print(f"  loaded {len(labels)} mitogenomes "
          f"({min(len(s) for s in seqs.values())}-{max(len(s) for s in seqs.values())} bp)")
    if len(labels) < 6:
        print("  too few genomes; aborting."); return

    seq_list = [seqs[l] for l in labels]

    # shared taxon namespace + reference tree pruned to fetched taxa
    taxa = dendropy.TaxonNamespace(labels)
    ref = dendropy.Tree.get(data=REFERENCE, schema="newick", taxon_namespace=taxa)
    ref.retain_taxa_with_labels(labels)
    ref.encode_bipartitions()

    n = len(labels)
    rf_max = 2 * (n - 3)  # max RF for unrooted binary trees on n taxa

    print("\n[computing distance matrices] (NRC/NCCD are the prior-art cross-entropy distances)")
    d_nrc, d_nccd = nrc_nccd_distance_matrices(seq_list, DNA, k=9)
    mats = {
        "FPC": tasks.distance_matrix(seq_list, DNA, order=9),
        "NRC": d_nrc,
        "NCCD": d_nccd,
        "FFP": ffp_distance_matrix(seq_list, DNA, k=6),
        "Mash": mash_distance_matrix(seq_list, k=16, sketch_size=800),
        "5-mer cos": kmer_cos_matrix(seq_list, k=5),
    }

    print("\n=== Robinson-Foulds to the reference tree (lower = better) ===")
    results = {}
    for name, D in mats.items():
        tree = nj_tree_from_matrix(D, labels, taxa)
        tree.encode_bipartitions()
        rf = treecompare.symmetric_difference(ref, tree)
        results[name] = {"rf": int(rf), "nrf": rf / rf_max}
        print(f"  {name:<10}: RF={rf:>3}   nRF={rf/rf_max:.3f}")

    # figure
    fig, ax = plt.subplots(figsize=(7, 4.4))
    names = list(results); vals = [results[k]["nrf"] for k in names]
    palette = {"FPC": "#2a78d6", "NRC": "#4a3aa7", "NCCD": "#9085e9",
               "FFP": "#1baf7a", "Mash": "#eb6834", "5-mer cos": "#898781"}
    ax.bar(names, vals, color=[palette.get(nm, "#888888") for nm in names])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylabel("normalized Robinson-Foulds (lower = better)")
    ax.set_title(f"Genome phylogeny on {n} real mammalian mitogenomes\n"
                 f"(NJ tree vs. reference topology)")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig11_rf_benchmark.png"), dpi=160)
    plt.close(fig)

    import json
    with open(os.path.join(OUT, "rf_benchmark.json"), "w") as fh:
        json.dump({"n_taxa": n, "taxa": labels, "results": results,
                   "reference_newick": REFERENCE}, fh, indent=2)
    print(f"\nFigure -> {os.path.join(FIG, 'fig11_rf_benchmark.png')}")


if __name__ == "__main__":
    main()
