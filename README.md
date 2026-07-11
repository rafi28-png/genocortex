# GenoCortex — the Fractal Predictive Cortex (FPC)

A training-free, alignment-free, self-organising model for biological sequences.
A single model performs next-symbol prediction, mutation/anomaly detection,
classification, and phylogenetic distance estimation, without training labels,
sequence alignment, pretrained weights, or a fixed k-mer order. The core engine
depends only on NumPy and SciPy and runs offline.

## Method in brief

The model reads a sequence left to right through a Chaos Game Representation
(CGR), which maps the recent context to a point in the unit square. It maintains
a growing set of predictive "prototypes" in that space and adds a new prototype
at the locations where its own next-symbol prediction is weakest. Its single
objective is to minimise cumulative surprise, `-log2 P(next | context)`, which
equals the sequence's description length under the model. The four capabilities
are read-outs of that one surprise signal:

- prediction: the next-symbol distribution `P(next | context)`;
- detection: mutations or foreign DNA appear as local increases in surprise
  (reported with robust z-scores and p-values);
- classification: assign a query to the model that is least surprised by it;
- distance/phylogeny: a symmetric cross-entropy between two sequences' learned
  models.

Because the model lives in a 2-D geometry, its internal state can be rendered as
a 3-D "surprise landscape".

## Installation

```bash
git clone https://github.com/rafi28-png/genocortex.git && cd genocortex
pip install -e .            # core: numpy + scipy
pip install -e .[all]       # + plotly, scikit-learn, matplotlib, pytest
```

Python >= 3.10. The phylogeny Robinson-Foulds benchmark also uses `dendropy`.

## Quick start

```python
from fpc import FractalPredictiveCortex, DNA, tasks, datasets

genome = datasets.markov_sequence(20000, DNA, order=2, seed=1)
model  = FractalPredictiveCortex(DNA, order=8)
track  = model.fit(DNA.encode(genome))
print(track)                                   # mean bits/symbol, perplexity

# predict the next base
p = model.predict_next(DNA.encode("ACGTACG"))
print(dict(zip("ACGT", p.round(3))))

# detect a foreign insert against a reference
host  = datasets.markov_sequence(6000, DNA, seed=2)
donor = datasets.markov_sequence(400,  DNA, seed=9)
chimera, span = datasets.insert_foreign(host, donor, position=3000)
res = tasks.detect_against_reference(host, chimera, DNA)
print(span, [(a.start, a.end) for a in res.anomalies])

# classify by least-surprised model
ds  = datasets.make_species(n_species=4, per_species=6, length=2500)
clf = tasks.FPCClassifier(DNA, order=6).fit(ds.sequences[::2], ds.labels[::2])
print(tasks.classification_accuracy(clf.predict(ds.sequences[1::2]), ds.labels[1::2]))

# phylogenetic distances -> MDS coordinates
phy   = datasets.phylogeny(depth=3, seed=3)
names = list(phy.leaves)
D     = tasks.distance_matrix([phy.leaves[n] for n in names], DNA)
coords = tasks.mds_embedding(D, dims=3)
```

Interactive 3-D visualisations (standalone offline HTML):

```bash
python examples/make_visuals.py     # writes output/*.html and .png snapshots
```

## Real data

The engine is alphabet-agnostic; use `PROTEIN` (or a custom `Alphabet`) with no
other change. `fpc/fetch.py` (the only component that uses the network) retrieves
records from NCBI and caches them for offline re-runs.

```bash
python examples/real_genomes.py     # 7 coronaviruses
python examples/big_study.py        # 28 genomes, 5 viral families
python examples/protein_study.py    # 40 proteins, 4 families
python benchmarks/rf_benchmark.py   # 17 mammalian mitogenomes, Robinson-Foulds
```

## Results

All numbers are reproducible from the scripts in `benchmarks/` and `examples/`
(fixed seeds; real data cached after first download). Losses are reported
alongside wins.

| Task | FPC | Baselines | Summary |
|---|---|---|---|
| Phylogeny, 25 synthetic trees (Pearson r) | 0.93 | FFP 0.93, NCCD 0.93, NRC 0.92, Mash 0.91, k-mer cosine 0.78 | statistically tied with FFP/Mash/NRC/NCCD (Wilcoxon p >= 0.24); beats k-mer cosine (p < 1e-6) |
| Phylogeny, 17 real mitogenomes (norm. Robinson-Foulds, lower better) | 0.43 | NRC 0.36, NCCD/FFP/Mash 0.43, k-mer cosine 0.71 | ties the standard tools; slightly behind NRC |
| Short-read classification, 150 bp | 100% | k-mer+1NN 97%, k-mer+logistic 94% | robust as reads shorten |
| Compression (bits/symbol) | beats gzip; near best Markov | best hand-tuned Markov-k | competitive without choosing k |
| Foreign-insert detection (ROC-AUC) | ~1.0 | 8-mer novelty ~1.0 | parity; adds calibrated per-position surprise |
| Real viral families, deep divergence (clustering purity) | 0.82 | FFP 0.93 | FFP is more accurate here |

FPC matches standard alignment-free tools on several tasks and is beaten on
others (compression on Markov data, deep-divergence family clustering). The
contribution is that one label-free, alignment-free, order-free model provides all
of these read-outs together, with an interpretable per-position surprise signal
and a 3-D view of its state. See `paper/paper.md` for the full study, honest
positioning, and prior work.

## Repository layout

```
fpc/            core engine (NumPy/SciPy only)
  alphabet.py   symbol handling (DNA/RNA/PROTEIN/custom)
  cgr.py        Chaos Game Representation
  cortex.py     the Fractal Predictive Cortex
  surprise.py   surprise trace + anomaly scan
  tasks.py      predict / detect / classify / distance
  datasets.py   reproducible synthetic data with ground truth
  io_fasta.py   FASTA reader/writer (gzip-aware)
  fetch.py      optional NCBI download for real data (public, no key)
viz/            3-D visualisation -> offline HTML
benchmarks/     baselines (Markov, gzip, k-mer, Mash, FFP, NRC, NCCD), experiments, stats
examples/       runnable demos and real-data studies
tests/          correctness tests
paper/          manuscript (paper.md, build_pdf.py, cover_letter.md)
```

Run everything: `python run_all.py` (add `--real` for the real-data studies).

## Limitations

- Not state-of-the-art on any single task; matched Markov models and PPM/CTW beat
  it on stationary compression, and FFP beats it on deep-divergence phylogeny.
- Effective context is bounded by the `order` parameter and by CGR's decaying
  memory, so very long-range dependencies are not captured.
- It is a learned model and needs enough sequence to train; short, few-example
  classes (e.g. ~150-aa protein families) favour simple k-mer methods.
- Self-anomaly detection can learn a long insert away; reference-based or relative
  detection is appropriate there.
- The cross-entropy distance is not novel (see NRC/NCCD in `paper/paper.md`).

## License

MIT (see `LICENSE`).
