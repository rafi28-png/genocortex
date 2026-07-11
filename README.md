<h1 align="center">🧠 GenoCortex — the Fractal Predictive Cortex</h1>

<p align="center">
<b>A training-free, alignment-free, self-organising "brain" for biological sequences.</b><br>
One model — no labels, no alignment, no pretrained weights, no fixed k-mer order —
that <b>predicts</b>, <b>detects mutations</b>, <b>classifies</b>, and <b>discovers phylogeny</b>,
and draws a live <b>3-D map of its own reasoning</b>.<br>
Core engine: pure NumPy/SciPy. <b>Runs 100% offline. No API. No internet.</b>
</p>

---

## The one idea

Everything here comes from a single principle: **a good model of a sequence is one
that keeps predicting what comes next, and pays attention to its own *surprise*
when it is wrong.**

1. Stream the sequence through a **Chaos Game Representation** — a fractal map
   where *"the recent context" becomes "a point in the plane."*
2. Grow a cloud of little predictive memory cells (**prototypes**) *wherever the
   model is surprised.* It literally decides where to spend memory.
3. Minimise cumulative surprise. That single number **is** the sequence's
   description length (the MDL principle), and four abilities fall out of it:

| Ability | How surprise gives it |
|---|---|
| 🔮 **Predict** | the next-symbol distribution `P(next \| context)` |
| 🩺 **Detect** | mutations / foreign DNA = local *spikes* of surprise (with p-values) |
| 🏷️ **Classify** | the label whose brain is *least surprised* (a generative MDL classifier) |
| 🌳 **Discover** | phylogeny from the *excess* surprise between sequences |

Because the model *is* a geometry, you can **see** it: a 3-D "surprise landscape."

---

## Install

```bash
git clone <your-repo>  &&  cd genocortex
pip install -e .            # core: numpy + scipy only
pip install -e .[all]       # + plotly (3-D), scikit-learn/matplotlib (benchmarks), pytest
```

Requires Python ≥ 3.10.

---

## 60-second quickstart

```python
from fpc import FractalPredictiveCortex, DNA, tasks, datasets

# --- learn a genome (no labels, no alignment, no training set) ---
genome = datasets.markov_sequence(20000, DNA, order=2, seed=1)
brain  = FractalPredictiveCortex(DNA, order=8)
track  = brain.fit(DNA.encode(genome))
print(track)          # SurpriseTrack(mean_bits=..., perplexity=...)

# 🔮 PREDICT the next base after a context
p = brain.predict_next(DNA.encode("ACGTACG"))
print(dict(zip("ACGT", p.round(3))))

# 🩺 DETECT a foreign insert
host  = datasets.markov_sequence(6000, DNA, seed=2)
donor = datasets.markov_sequence(400,  DNA, seed=9)
chim, span = datasets.insert_foreign(host, donor, position=3000)
res = tasks.detect_against_reference(host, chim, DNA)
print("true insert:", span, "-> detected:", [(a.start, a.end) for a in res.anomalies])

# 🏷️ CLASSIFY (which brain is least surprised)
ds  = datasets.make_species(n_species=4, per_species=6, length=2500)
clf = tasks.FPCClassifier(DNA, order=6).fit(ds.sequences[::2], ds.labels[::2])
print("accuracy:", tasks.classification_accuracy(
    clf.predict(ds.sequences[1::2]), ds.labels[1::2]))

# 🌳 DISCOVER an (unsupervised, alignment-free) phylogeny
phy   = datasets.phylogeny(depth=3, seed=3)
names = list(phy.leaves)
D     = tasks.distance_matrix([phy.leaves[n] for n in names], DNA)
coords = tasks.mds_embedding(D, dims=3)   # -> feed to the 3-D viewer
```

### See the brain think (interactive 3-D, offline HTML)

```python
from viz import cognitive_map_html
cognitive_map_html(brain, DNA.encode(genome), "cognitive_map.html")
# open cognitive_map.html in any browser — works with no internet
```

Or generate all three flagship visuals at once:

```bash
python examples/make_visuals.py     # -> output/*.html  (and .png snapshots)
```

<p align="center"><i>the surprise landscape · a foreign-insert anomaly · an unsupervised phylogeny constellation</i></p>

---

## Works on your real data too

```python
from fpc import io_fasta, DNA, tasks
recs = io_fasta.read_fasta("my_genome.fasta")        # multi-record, gzip-aware
res  = tasks.detect_anomalies(recs[0].seq, DNA, order=9)
```

It is **alphabet-agnostic** — use `PROTEIN` (or define your own `Alphabet`) and the
exact same engine works on protein sequences. An *optional* `fpc/fetch.py` helper
(the only part that touches the internet) can pull real records from NCBI.

**Real-genome demo** — downloads 7 real coronaviruses and runs all four abilities:

```bash
python examples/real_genomes.py     # 7 coronaviruses: predict/detect/classify/phylogeny
python examples/big_study.py         # 28 genomes across 5 viral families
```

On real data it recovers the correct close relatives unsupervised (SARS-CoV-2↔SARS-CoV-1,
Dengue serotypes, Ebola species, HIV/SIV; **4 of 5 families as clean clades**) and flags spliced
**phiX174 contamination** at ROC-AUC 1.0 — while honestly showing where composition-only methods
are competitive-to-better (deep divergences; near-incompressible viral DNA). See `paper/paper.md`
§3.6–3.7.

---

## Optional: explain findings in plain English (the *only* online part)

The offline core produces numbers; an **optional, clearly-separated** layer
(`fpc/explain.py`) can turn them into a plain-language interpretation for a paper
or a collaborator, using the Claude API. Nothing in the core imports it.

```bash
pip install anthropic
setx ANTHROPIC_API_KEY "sk-ant-..."      # reopen the shell after this
python examples/explain_demo.py           # runs offline; calls Claude only if the key is set
```

```python
from fpc import DNA, tasks, datasets
from fpc.explain import explain_detection
res = tasks.detect_against_reference(host, chimera, DNA)
print(explain_detection(res, sequence_name="chimera"))   # -> readable write-up
```

Defaults to **Claude Fable 5** with an automatic fallback to Opus. The core still
runs 100% offline without any of this.

## Honest results (reproduce with `python benchmarks/bench.py`)

Synthetic data, known ground truth. We report losses as plainly as wins.

| Task | FPC | Best baseline | Verdict |
|---|---|---|---|
| **Phylogeny** (corr. w/ true distance) | **r = 0.94** | k-mer cosine r = 0.76 | ✅ clear win |
| **Short-read ID** (150 bp, 6 species) | **100%** | k-mer+1-NN 97% / logistic 94% | ✅ win |
| **Adaptivity** (bits/symbol) | beats gzip & Markov-2, no *k* to tune | best tuned Markov-*k* | ⚖️ competitive, no tuning |
| **Prediction** (next-base acc.) | 0.83→0.89 | Markov-6 0.86→0.90 | ⚖️ competitive |
| **Detection** (insert ROC-AUC) | ≈ 1.0 | 8-mer novelty ≈ 1.0 | ⚖️ parity (+ calibrated bits) |
| **Compression on Markov data** | 0.38 | Markov-2 **0.30** | ❌ oracle wins, as it should |

**The point is not to win every row** — specialised tools beat FPC on their home
turf. The point is that *one* label-free, alignment-free, order-free model does
**all** of these at once, and shows you its reasoning in 3-D.

---

## What's inside

```
fpc/                  the core brain (pure NumPy/SciPy, no API)
  alphabet.py         alphabet-agnostic symbol handling (DNA/RNA/PROTEIN/custom)
  cgr.py              the fractal mind-space (Chaos Game Representation)
  cortex.py           the Fractal Predictive Cortex — the self-organising model
  surprise.py         surprise trace + robust, p-valued anomaly scan
  tasks.py            the four abilities (predict / detect / classify / discover)
  datasets.py         reproducible synthetic genomes with ground truth
  io_fasta.py         dependency-free FASTA reader/writer (gzip-aware)
viz/viz3d.py          interactive 3-D cognitive map -> standalone offline HTML
benchmarks/           baselines (Markov, gzip, k-mer) + the 5 experiments + figures
tests/                15 correctness tests (incl. the CGR suffix-locality property)
examples/             runnable demos
paper/paper.md        the methods-paper draft (honest results + limitations)
```

---

## Limitations (read these)

- Not a state-of-the-art compressor or predictor; matched Markov / PPM / CTW beat
  it on stationary data.
- Effective context length is bounded by `order` and by CGR's decaying memory;
  very long-range dependencies are not captured.
- Self-anomaly detection can *learn* a long insert away — use reference-based
  detection there. Point-mutation precision drops in high-entropy regions.
- Benchmarks are synthetic-with-ground-truth (the right first step); large
  real-genome evaluation is future work.

See `paper/paper.md` §5 for the full discussion.

---

## License

MIT. Built as an open, reproducible research project.
