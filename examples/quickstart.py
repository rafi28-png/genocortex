"""
Quickstart: all four abilities of the Fractal Predictive Cortex, from one engine.

Run:  python examples/quickstart.py
"""
from collections import defaultdict
import numpy as np

from fpc import FractalPredictiveCortex, DNA, tasks, datasets


def main():
    print("=" * 62)
    print("1. PREDICT  -- next-base accuracy on a learnable organism")
    print("=" * 62)
    seq = datasets.markov_sequence(20000, DNA, order=2, concentration=0.15, seed=1)
    cx = FractalPredictiveCortex(DNA, order=8)
    cx.fit(DNA.encode(seq[:16000]))          # fit() returns the SurpriseTrack, not the cortex
    ev = tasks.evaluate_prediction(cx, seq[16000:])
    print(f"   accuracy={ev['accuracy']:.3f}  perplexity={ev['perplexity']:.3f}  (chance acc=0.25)")

    print("\n" + "=" * 62)
    print("2. DETECT   -- a foreign insert (vs a reference brain)")
    print("=" * 62)
    host = datasets.markov_sequence(6000, DNA, order=2, concentration=0.15, seed=2)
    donor = datasets.markov_sequence(400, DNA, order=3, concentration=0.4, seed=999)
    chimera, span = datasets.insert_foreign(host, donor, position=3000, seed=0)
    res = tasks.detect_against_reference(host, chimera, DNA, order=8, window=40, z_threshold=3.0)
    hit = any(not (a.end < span[0] or a.start > span[1]) for a in res.anomalies)
    print(f"   truth insert {span}  ->  recovered: {hit}")

    print("\n" + "=" * 62)
    print("3. CLASSIFY -- which brain is least surprised (no engineered labels)")
    print("=" * 62)
    ds = datasets.make_species(n_species=4, per_species=6, length=2500, seed=7)
    buck = defaultdict(list)
    for s, l in zip(ds.sequences, ds.labels):
        buck[l].append(s)
    Xtr, ytr, Xte, yte = [], [], [], []
    for l, seqs in buck.items():
        Xtr += seqs[:4]; ytr += [l] * 4
        Xte += seqs[4:]; yte += [l] * (len(seqs) - 4)
    clf = tasks.FPCClassifier(DNA, order=6).fit(Xtr, ytr)
    print(f"   test accuracy = {tasks.classification_accuracy(clf.predict(Xte), yte):.3f}"
          f"  (chance = {1/len(clf.classes_):.2f})")

    print("\n" + "=" * 62)
    print("4. DISCOVER -- an unsupervised, alignment-free phylogeny")
    print("=" * 62)
    phy = datasets.phylogeny(root_length=4000, depth=3, branch_sub=0.04, seed=3)
    names = list(phy.leaves)
    D = tasks.distance_matrix([phy.leaves[n] for n in names], DNA, order=6)
    purity = tasks.clustering_purity(tasks.cluster(D, 2), [n[1] for n in names])
    tr, es = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tr.append(phy.truth_distance[(names[i], names[j])]); es.append(D[i, j])
    print(f"   clade purity = {purity:.2f}   corr(recovered, true) = {np.corrcoef(tr, es)[0,1]:.3f}")

    print("\nAll four abilities ran from the ONE self-organising engine.")


if __name__ == "__main__":
    main()
