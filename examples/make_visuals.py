"""
Generate the interactive 3-D artifacts (all offline standalone HTML).

Run:  python examples/make_visuals.py
Outputs land in  output/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpc import FractalPredictiveCortex, DNA, tasks, datasets
from viz import cognitive_map_html, phylogeny_3d_html

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    # ---- 1. A brain's learned terrain (self-trained on a structured genome) ----
    print("[1/3] cognitive map of a learned genome ...")
    genome = (datasets.markov_sequence(9000, DNA, order=2, concentration=0.12, seed=4)
              + datasets.random_sequence(1500, DNA, gc=0.7, seed=8)   # a GC-rich island
              + datasets.markov_sequence(4000, DNA, order=2, concentration=0.12, seed=4))
    ids = DNA.encode(genome)
    cx = FractalPredictiveCortex(DNA, order=9, seed=0)
    cx.fit(ids)
    p1 = cognitive_map_html(cx, ids, os.path.join(OUT, "cognitive_map.html"),
                            title="Cognitive map - a genome the brain has learned",
                            png_path=os.path.join(OUT, "cognitive_map.png"))
    print("      ->", p1, f"({cx.n_prototypes} prototypes)")

    # ---- 2. An anomaly lighting up the landscape (foreign insert) ----
    print("[2/3] anomaly landscape (foreign insert vs a reference brain) ...")
    host = datasets.markov_sequence(7000, DNA, order=2, concentration=0.15, seed=2)
    donor = datasets.markov_sequence(500, DNA, order=3, concentration=0.4, seed=999)
    chimera, span = datasets.insert_foreign(host, donor, position=3500, seed=0)
    ref = FractalPredictiveCortex(DNA, order=9, seed=0)
    ref.fit(DNA.encode(host))
    cids = DNA.encode(chimera)
    p2 = cognitive_map_html(ref, cids, os.path.join(OUT, "anomaly_map.html"),
                            show_prototypes=False,
                            title=f"Surprise landscape - foreign insert at {span[0]}-{span[1]}",
                            png_path=os.path.join(OUT, "anomaly_map.png"))
    print("      ->", p2, f"(insert span {span})")

    # ---- 3. Unsupervised phylogeny constellation ----
    print("[3/3] cross-entropy phylogeny constellation ...")
    phy = datasets.phylogeny(root_length=4000, depth=3, branch_sub=0.04, order=2, seed=3)
    names = list(phy.leaves)
    seqs = [phy.leaves[n] for n in names]
    D = tasks.distance_matrix(seqs, DNA, order=6)
    coords = tasks.mds_embedding(D, dims=3)
    clusters = tasks.cluster(D, n_clusters=4)
    p3 = phylogeny_3d_html(coords, clusters, os.path.join(OUT, "phylogeny.html"),
                           names=names, distance_matrix=D,
                           png_path=os.path.join(OUT, "phylogeny.png"))
    print("      ->", p3)

    print("\nDone. Open the HTML files in any browser (they work fully offline).")


if __name__ == "__main__":
    main()
