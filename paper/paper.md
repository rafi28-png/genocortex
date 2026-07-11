# The Fractal Predictive Cortex: a training-free, alignment-free, self-organising model that unifies prediction, anomaly detection, classification, and phylogenetics for biological sequences

**[AUTHOR NAME]¹, [Co-author / Supervisor Name]²**

¹ [Department, University, City, Country] — *corresponding author:*
`rafiur2810@gmail.com`
² [Supervisor's Department, Institution] *(recommended: add a faculty mentor)*

> ⚠️ *Author placeholders — replace `[AUTHOR NAME]` and affiliations before
> submission. A faculty co-author is strongly advised for a peer-reviewed
> submission.*

**Keywords:** alignment-free sequence comparison · Chaos Game Representation ·
predictive coding · minimum description length · self-organising maps ·
phylogenetics · anomaly detection · genome classification

*Working draft — a reproducible methods paper. Every number and figure is
produced by the accompanying open-source code (`benchmarks/` and `examples/`; see
§6) and is reported honestly, including the several cases where the method only
matches, or is beaten by, its baselines.*

---

## Abstract

Alignment-free sequence analysis usually means picking a task (compression,
classification, distance estimation, anomaly detection) and then picking a tool
built specifically for it, each with its own hyperparameters — most importantly
the k-mer length *k*. We present the **Fractal Predictive Cortex (FPC)**, a
single self-organising model that performs all four tasks at once, with **no
training labels, no sequence alignment, no pretrained weights, and no fixed
k-mer order**. FPC streams a sequence through a Chaos Game Representation, which
turns "the recent context" into "a point in the plane", and grows a cloud of
predictive **prototypes** wherever it is *surprised* — where its own next-symbol
prediction fails. Its single objective, cumulative surprise, is exactly the
sequence's description length (the Minimum Description Length principle), and the
four capabilities are four read-outs of that one quantity. On controlled data
with known ground truth, FPC (i) beats a general-purpose compressor (gzip) and
low-order Markov models and remains competitive with the *best hand-tuned*
fixed-order Markov model **without having to choose an order**; (ii) classifies
short reads (down to 150 bp) at 100% where k-mer + kNN/logistic baselines fall to
94–97%; (iii) recovers evolutionary distances **statistically on par with the
standard tools FFP and Mash** across 25 independent controlled phylogenies (mean
Pearson r = 0.93; paired Wilcoxon p = 0.29 vs FFP, p = 0.52 vs Mash) and
**significantly better than a naive k-mer-cosine baseline** (r = 0.78, p < 10⁻⁶);
and (iv) detects foreign inserts on par with a strong k-mer-novelty detector while
adding calibrated, per-position surprise. On real genomes — seven coronaviruses, a
28-genome five-family study, and 40 real proteins — it recovers the correct
closely-related groups unsupervised (SARS pair, Dengue serotypes, Ebola species,
HIV/SIV; four of five families as clean clades), works unchanged on the 20-letter
protein alphabet, and cleanly flags spliced phiX174 contamination (per-window
ROC-AUC = 1.0). On the community-standard test — Robinson–Foulds tree distance on
17 real mammalian mitogenomes — FPC **exactly ties Mash and FFP** (normalized RF
0.43) and beats k-mer cosine (0.71). We report the losses just as plainly: on the hardest *deep*-
divergence real families the composition baseline **FFP is clearly better** (family
purity 0.82 vs 0.93), and near-incompressible viral DNA limits compression for
every method. Because the model *is* a geometry, it renders directly as an
interactive 3-D **cognitive map** of its own reasoning. A systematic literature
review confirms that FPC's individual ingredients — CGR, self-organising maps,
prediction/compression-based comparison, the least-surprised-model classifier, and
the surprise = description-length identity — are all established prior art; the
contribution is their **unification into a single surprise-driven *growing* memory
in CGR space**, for which we find no near-identical precedent. The core engine is
~1300 lines of NumPy/SciPy and runs fully offline.

---

## 1. Introduction

A great deal of modern bioinformatics is *alignment-free*: rather than aligning
sequences base-by-base (which is expensive and assumes homology), we summarise a
sequence by the statistics of its short sub-words (*k*-mers) and compute on those
[Vinga & Almeida 2003; Zielezinski et al. 2017]. This is fast and general, but it
fragments into task-specific pipelines: one tool compresses, another classifies,
another builds trees, another scans for anomalies — and almost all of them hinge
on a hand-chosen word length *k* whose best value is unknown in advance and
differs from dataset to dataset.

We ask a different question: *is there a single model whose one internal signal
naturally answers all of these questions?* Our answer draws on two old, elegant,
and (individually) well-understood ideas — coupled, as we argue in §1.1, in a way
that has no near-identical precedent:

1. **Chaos Game Representation (CGR)** [Jeffrey 1990] maps a sequence to a
   trajectory of points in the unit square such that *positions sharing a long
   common suffix land at nearly the same point*. Geometry becomes context.

2. **Predictive coding / surprise minimisation** — the idea, from both data
   compression (MDL, PPM, context-tree weighting [Rissanen 1978; Cleary & Witten
   1984; Willems et al. 1995]) and theoretical neuroscience [Rao & Ballard 1999;
   Friston 2010], that a good model of a stream is one that predicts the next
   token well, and that *prediction error (surprise)* is the signal worth acting
   on.

FPC places a growing, self-organising set of predictive units **in CGR space**
and trains them online by surprise: it spends memory (grows a new unit) exactly
where its predictions fail. The result is a model that is simultaneously a
predictor, a compressor, a novelty detector, a generative classifier, and a
distance function — and, because it lives in a 2-D geometry, a picture.

**What this paper claims, and what it does not.** We do *not* claim
state-of-the-art on any single task: a matched fixed-order Markov model beats FPC
on Markov-generated compression, and the composition method FFP beats it on
deep-divergence real phylogeny — we report those losses plainly. We *do* claim
(a) a genuinely unified framework in which four capabilities fall out of one
mechanism and one objective; (b) *competitiveness* with standard alignment-free
tools — statistically indistinguishable from FFP and Mash on controlled
phylogenies (§3.4) and robust on short reads (§3.3); (c) freedom from
k-mer-order selection; and (d) a novel, interpretable 3-D visualisation of the
model's internal state and a calibrated per-position surprise signal that
single-purpose distance tools do not provide.

### 1.1 Related work and the precise locus of novelty

A systematic literature review (24 claims, independently verified) establishes
that **every individual ingredient of FPC is prior art**, and pins down exactly
what is not:

* **CGR + neural / self-organising models.** CGR/FCGR has been fed to CNNs
  (*CGRclust*; Alipour, Hill & Kari, 2024) and combined with a static k-mer
  "dynamical language" weighting for phylogeny (*CGRWDL*; Wang, Yu & Li, 2024).
  The closest CGR + self-organising-map work is *SOCGR* (Neme et al., 2008) — but
  it reorders a **fixed** 2ᵏ×2ᵏ lattice for topological continuity; it does not
  grow, predict, or use surprise. FPC shares the CGR substrate with these, not
  the mechanism.
* **The "least-surprised model" classifier is textbook.** Training one predictor
  per class and labelling a query by `argmax_i P^i(x)` (equivalently, minimum
  −log P = least surprise) is the winner-take-all variable-order-Markov classifier
  of Begleiter, El-Yaniv & Yona (2004), demonstrated on SCOP protein folds. We
  claim no novelty for this rule.
* **Prediction/compression comparison is mature.** Mixture-of-predictor DNA
  compressors (*GeCo3*, Silva et al. 2020; *Jarvis*, Pratas et al. 2019) and
  compression distances (NCD/IBD; La Rosa et al. 2013; Ferragina et al. 2007)
  embody the same prediction-as-modelling philosophy with fixed-topology mixers,
  not CGR or growing memory.
* **surprise = code length = MDL** is standard information theory (Rissanen;
  Cover & Thomas), and the FCGR of order *k* is provably equivalent to the k-mer
  frequency vector at that order (He, Kari & Millán-Arias, 2025) — so neither the
  objective nor the representation is itself novel.

**What is left, and is our contribution:** a single model that grows predictive
memory *specifically where it is most surprised*, in CGR space, and derives
prediction, detection, classification, and phylogeny from that one surprise
signal. The review found **no prior method** that (a) uses grow-where-surprised
memory for sequence prediction, (b) applies an explicit predictive-coding framing
to DNA/protein, or (c) defines phylogenetic distance as the cross-entropy between
two sequences' *learned online predictive models* (rather than compressed sizes,
as in NCD). We therefore frame FPC as a novel **unification/architecture**, and
(per the AFproject benchmark, below) make no claim of universal superiority.

---

## 2. Method

### 2.1 Notation and the alphabet

A sequence is a string over an alphabet of `a` canonical symbols (DNA: `a=4`;
protein: `a=20`), plus a wildcard for ambiguous characters. Everything below is
alphabet-agnostic; only the symbol-to-integer map knows about letters.

### 2.2 The fractal mind-space (Chaos Game Representation)

Fix a vertex `v(s) ∈ [0,1]²` for each canonical symbol (the four corners of the
unit square for DNA; a regular polygon on a circle otherwise). Starting from the
centre `p₀ = (½, ½)`, read the sequence and iterate the contraction map

```
    p_t = (1 − r) · p_{t−1} + r · v(s_t),        0 < r < 1.
```

For DNA we use the classical `r = ½`; in general `r = min(½, sin(π/a))` keeps the
per-symbol sub-copies from overlapping. Because the map is a contraction, the
influence of a symbol decays as `(1−r)` per step, so **`p_t` is an
exponentially-weighted memory of the recent suffix**. Concretely, two positions
whose last *k* symbols agree differ by at most `~(1−r)^k` in position (empirically
`0.5^8·√2 ≈ 0.0055` for an 8-symbol shared suffix, versus `~0.1` for a different
suffix — a >10× separation; see `tests/test_core.py::test_cgr_shared_suffix_are_close`).
We call `C_t = p_{t−1}` the **context point** for position *t*: the geometric
summary of everything the model has read before symbol *t*.

The same trajectory, binned into a grid, yields the **Frequency-CGR image** — a
fixed-length, alignment-free fingerprint of a whole sequence, here generalised to
any alphabet.

### 2.3 The cortex: prototypes that grow where the model is surprised

The model keeps a set of **prototypes** `{(x_j, n_j, u_j)}`, where `x_j ∈ [0,1]²`
is a location, `n_j ∈ ℝ^a` a tally of which symbol followed when the context
landed near `x_j`, and `u_j` a (decaying) usage counter.

**Prediction.** At context point `C_t`, take the `κ` nearest prototypes, weight
them by a Gaussian of their distance (`w_j ∝ exp(−‖C_t − x_j‖² / 2σ²)`, with
`σ` equal to the split resolution below), and form the next-symbol distribution

```
    P(s | C_t) = (1−λ) · ( Σ_j w_j n_j[s] + α ) / ( Σ_j w_j Σ_s n_j[s] + aα )
                 + λ · P_bg(s),
```

i.e. a distance-weighted, Laplace-smoothed (`α`) mixture of the local
prototypes, blended with a small background term (`λ`) so every probability is
strictly positive and surprise is always finite.

**Surprise.** The model observes the true `s_t` and pays

```
    surprise_t = −log₂ P(s_t | C_t)   [bits].
```

Summed over the sequence, this is the code length under the model — the object
FPC minimises, i.e. the MDL criterion.

**Online learning and the growth decision.** After scoring, FPC reinforces the
winning prototype (`n_win[s_t] += 1`) and nudges it toward the context
(`x_win += η (C_t − x_win)`, SOM-style). Crucially, it *grows a new prototype at
`C_t`* when it is in unfamiliar or hard territory:

```
    grow  if  nearest-distance > τ           (empty region), or
              surprise_t is abnormally high  (a region it keeps getting wrong),
    where τ = r^order.
```

The single knob `order` sets the split resolution `τ = r^order`, which — because
CGR distance halves per shared suffix symbol — corresponds to resolving roughly
`order` symbols of context. This is the model's *maximum* effective Markov order;
it uses that resolution only where surprise justifies it, and coarser resolution
elsewhere. A usage-decay + prune step caps total memory. This "grow where you are
surprised" rule is the point of contact between the CGR geometry and predictive
coding, and is, to our knowledge, the novel element of the method.

### 2.4 Four read-outs of one signal

* **Predict.** `P(s | C_t)` directly; argmax for next-base calling, sampling for
  generation, per-symbol log-loss for evaluation.
* **Detect.** The per-position surprise trace, smoothed and converted to a
  robust z-score (median/MAD) with a per-region Gaussian p-value and a Bonferroni
  correction; spikes are candidate mutations, inserts, or contamination. A
  reference brain (trained on a reference genome) flags stretches of a query it
  cannot explain.
* **Classify.** Train one cortex per class; label a query by *which brain is
  least surprised* — the minimum cross-entropy / MDL rule, which for equal priors
  is the Bayes-optimal decision for the learned generative models. Softmax over
  `−bits` gives calibrated posteriors.
* **Discover.** Between two sequences define the symmetric *excess* cross-entropy
  `d(A,B) = ½[(CE_A(B) − CE_B(B)) + (CE_B(A) − CE_A(A))]`, a Kullback–Leibler-style
  divergence of their generative statistics (0 for identical, growing with
  divergence). Feed it to classical MDS (for a 3-D embedding) or hierarchical
  clustering (for a tree).

### 2.5 The 3-D cognitive map

Because the model lives in `[0,1]²`, we visualise it directly: the floor is CGR
space, the height/colour is surprise (the **surprise landscape**), and the
prototypes float above their territory coloured by the symbol they predict. A
foreign insert appears as a bright ridge; a mastered motif as a valley. A second
view places whole organisms in 3-D by their cross-entropy distance (MDS) with a
minimum-spanning-tree "lineage skeleton". Both export to standalone, offline HTML.

---

## 3. Experiments

All data are synthetic with known ground truth, generated by `fpc/datasets.py`
with fixed seeds; every number and figure is reproduced by `benchmarks/bench.py`.
Baselines: fixed-order Markov models (`Markov-k`, Laplace-smoothed), `gzip -9` as
a model-free code-length reference, k-mer frequency vectors fed to scikit-learn's
1-NN (cosine) and logistic regression, and — for phylogeny — faithful pure-Python
re-implementations of the two standard alignment-free tools **Mash** (bottom-s
MinHash of canonical k-mers → Mash distance; Ondov et al. 2016) and **FFP**
(k-mer frequency profiles compared by Jensen–Shannon distance; Sims et al. 2009).
FPC uses a single generic setting (`order` 6–12) across all experiments; it is
never tuned per dataset.

### 3.1 Compression / adaptivity (Fig. 1)

Held-out bits/symbol on three regimes. On a **heterogeneous** genome (blocks of
different orders + a tandem repeat + a low-complexity region) FPC reaches 1.12
bits, beating gzip (1.55) and Markov-2 (1.42) but not the best fixed order here,
Markov-6 (0.96). On a **high-order, data-limited** source, note that Markov-8
(1.48) *overfits* and is beaten by Markov-6 (1.24) — evidence that the "right" k
is neither obvious nor monotone — while FPC (1.72) again beats gzip (2.24) and
Markov-2 (1.95). On the **low-order Markov oracle** case, Markov-2 wins (0.30) as
it must; FPC (0.38) stays close and beats gzip (0.62). **Takeaway:** FPC is a
solid general-purpose model that needs no order selection, not a record-setting
compressor.

![Fig 1](../output/figures/fig1_compression.png)

### 3.2 Prediction (Fig. 2)

Next-base accuracy vs. training length on an order-3 source. FPC tracks Markov-6
within ~1–3 points across three orders of magnitude of training data (FPC
0.83→0.89; Markov-6 0.86→0.90), i.e. competitive prediction from the same model
that does everything else.

![Fig 2](../output/figures/fig2_prediction.png)

### 3.3 Classification of short reads (Fig. 3)

Six species (distinct order-2 models), classifiers trained on long references,
tested on reads of decreasing length. FPC holds **100%** accuracy from 2400 bp
down to **150 bp**; the k-mer baselines are strong but slip at short lengths
(1-NN 0.97, logistic 0.94 at 150 bp). The generative "least-surprised" rule uses
the full learned model, so it degrades more gracefully as reads shorten.

![Fig 3](../output/figures/fig3_classification.png)

### 3.4 Unsupervised phylogenetics — statistically rigorous (Figs. 4, 9)

On a single example tree, FPC cross-entropy distance correlates with true
patristic distance at r = 0.94 versus r = 0.76 for 5-mer cosine (Fig. 4). But a
single tree proves nothing, so we recover **25 independent random phylogenies**
(varying divergence depth, source order, and composition) and score every
method's correlation-with-truth per tree, with bootstrap 95% CIs and paired
Wilcoxon tests (Fig. 9). The honest result: FPC (mean r = **0.926**, 95% CI
0.900–0.948) is **statistically indistinguishable from the standard tools FFP
(0.927; paired p = 0.29) and Mash (0.913; p = 0.52)**, and **significantly better
than naive 5-mer cosine** (0.780; p < 10⁻⁶). We therefore claim *parity with
strong standard tools on controlled phylogenies*, not superiority — the earlier
single-tree "win" over k-mer cosine held up under scrutiny only against that
*weak* baseline.

![Fig 4](../output/figures/fig4_phylogeny.png)

![Fig 9](../output/figures/fig8_phylo_stats.png)

### 3.5 Foreign-insert detection (Fig. 5)

Per-position ROC for detecting a 600 bp insert in an 8 kb host, for an *unrelated*
donor (easy) and a *compositional-cousin* donor that shares low-order statistics
(hard). Both FPC surprise and 8-mer novelty achieve AUC ≈ 1.0 in both regimes:
FPC matches a strong, standard detector. FPC's added value is qualitative —
calibrated per-position bits with p-values, point-mutation localisation, and a
reference-free self-detection mode — rather than a higher AUC here.

![Fig 5](../output/figures/fig5_detection_roc.png)

### 3.6 Real genomes: seven coronaviruses (`examples/real_genomes.py`)

To move beyond synthetic data we downloaded seven real coronavirus RefSeq
genomes (SARS-CoV-2, SARS-CoV-1, MERS-CoV, HCoV-OC43, HCoV-HKU1, HCoV-229E,
HCoV-NL63; ~27–31 kb each) with known taxonomy.

* **Compression.** All k-mer models sit near 2.0 bits/base on real viral DNA —
  coronavirus genomes are close to incompressible at short range (their signal
  is codon- and protein-level). FPC (2.003) still beats gzip (2.525) and the
  higher-order Markov models; it matches Markov-2 (1.95). Honest conclusion:
  little short-range redundancy exists to exploit, and FPC extracts about as
  much as any k-mer method while beating a general compressor.

* **Phylogeny (partial success).** Unsupervised FPC distance correctly pairs the
  *closely related* taxa — SARS-CoV-2 ↔ SARS-CoV-1 (Sarbecovirus), HCoV-OC43 ↔
  HCoV-HKU1 (Embecovirus), and HCoV-229E ↔ HCoV-NL63 (the two Alphacoronaviruses)
  — and the 3-D MDS constellation (Fig. 6) visibly separates the subgenera. At
  the deepest split, Alpha/Beta genus clustering purity is 0.71: composition
  alone does not fully resolve ancient divergences, a well-known limitation of
  alignment-free methods that we report rather than hide.

* **Detection (real contamination).** A coronavirus reference brain scores ~2
  bits/base everywhere, so a foreign insert produces *no absolute surprise
  spike* — the self/reference surprise-spike detector fails here, and we say so.
  The correct instrument is the model's own classification principle applied
  locally: a host brain vs. a phiX174 brain (phiX174 is the standard Illumina
  spike-in and a real contaminant), scored in sliding windows. This *relative*
  surprise cleanly separates the spliced phiX174 segment at per-window
  ROC-AUC = 1.000. The lesson — absolute surprise needs a compressible host,
  relative surprise does not — is itself a useful contribution.

![Fig 6](../output/real_phylogeny.png)

### 3.7 A larger real-genome study: 28 genomes, five families (`examples/big_study.py`)

We scaled up to 28 real viral genomes spanning five families with ground-truth
labels — Coronaviridae (7), Flaviviridae (8), Filoviridae (5), Picornaviridae
(5), Retroviridae (3) — using a length-fair 6 kb central window from each.

* **Unsupervised phylogeny (Fig. 7, Fig. 8).** The FPC cross-entropy dendrogram
  recovers **four of the five families as clean clades** (all Coronaviridae, all
  Flaviviridae, all Filoviridae, all Retroviridae group monophyletically) and
  resolves **every close relationship correctly**: SARS-CoV-2 + SARS-CoV-1 is the
  tightest pair, the four Dengue serotypes cluster, the Ebola species cluster,
  and HIV-1/HIV-2/SIV cluster. The one split — Picornaviridae — is *real biology*:
  Hepatitis A (genus *Hepatovirus*) is genuinely deeply divergent from the
  enteroviruses (Polio, Coxsackie). Family clustering purity (5 clusters), all
  methods: **FFP 0.93 > 5-mer cosine 0.86 > FPC 0.82 > Mash 0.71**. This is the
  clearest reversal from the synthetic result — on real, deeply-divergent
  families the simple composition method FFP is *better* than FPC, and we report
  it as such. (The purity metric also penalises the biologically-correct
  Picornaviridae split, which lowers every method.)

* **Family classification.** On the four families with ≥4 members, held-out
  accuracy is FPC 0.82, tying the best baseline (4-mer + 1-NN, 0.82) and beating
  4-mer + logistic regression (0.64).

**Nuanced, honest conclusion.** Across *deep, ancient* family divergences,
composition-based k-mer methods (FFP especially) are strong and **beat** FPC;
FPC is competitive but not superior. Where FPC's geometry holds parity is
*moderate / controlled* divergence (§3.4, statistically tied with FFP/Mash) and
*fine / recent* structure (the correct within-family pairings above) — together
with the unification and the 3-D map that no single-purpose baseline provides.

![Fig 7](../output/figures/fig7_big_dendrogram.png)

![Fig 8](../output/big_phylogeny.png)

### 3.8 Real proteins: the same engine on a 20-letter alphabet (`examples/protein_study.py`)

FPC's core claim is that it is alphabet-agnostic. We test it on **40 real
proteins** (fetched from NCBI) spanning four families — cytochrome c, haemoglobin
β, myoglobin, lysozyme (82–200 aa) — with a *one-line* change (`DNA → PROTEIN`).
Family classification: FPC 0.75 vs. a 3-mer + 1-NN baseline 0.85; family
clustering purity: FPC 0.525 vs. FFP 0.400 (chance 0.25). Two honest readings:
(i) the identical engine runs unchanged on the 20-symbol alphabet and beats
chance — the alphabet-agnostic claim holds on real data; (ii) on these *short*
(≈150 aa), partly-overlapping families (myoglobin and haemoglobin are both
globins), neither FPC nor the k-mer baseline excels, and FPC does not outperform
the composition baseline on classification. FPC is a *learned* model and is
data-starved when each class is only a handful of short sequences — a genuine
limitation, not a bug.

![Fig 10](../output/protein_phylogeny.png)

### 3.9 Community-standard benchmark: Robinson–Foulds on real mitogenomes (Fig. 11)

The metric reviewers expect for genome phylogeny (the AFproject standard) is the
topological **Robinson–Foulds (RF)** distance between a method's
Neighbor-Joining tree and a trusted reference tree. On **17 real mammalian
mitochondrial genomes** (16.3–17.2 kb, NCBI RefSeq) with a reference topology
from established mammalian phylogeny (Boreoeutheria = Euarchontoglires +
Laurasiatheria, sister to Afrotheria; rooted on a marsupial outgroup; resolved
where accepted, polytomous where genuinely debated), FPC's NJ tree achieves
**normalized RF = 0.429 — exactly tying the standard tools FFP and Mash (both
0.429)** and clearly beating 5-mer cosine (0.714). This is the clearest
defensible statement of FPC's phylogenetic standing: **on the community-standard
dataset and metric, it is on par with the strong standard tools**, and the
whole-model unification and interpretability come at no cost in tree accuracy
here.

![Fig 11](../output/figures/fig11_rf_benchmark.png)

---

## 4. Discussion

The experiments support a specific, honest thesis. FPC is *not* the best tool for
any single task in isolation: specialised compressors and matched Markov models
beat it on compression; FFP beats it on deep-divergence real phylogeny; a good
k-mer detector matches it on insert detection. Its contribution is **unification
with interpretability and no order selection**: one label-free, alignment-free
model produces a predictor, a compressor (better than gzip), a short-read
classifier (better than k-mer + ML at short lengths), a phylogenetic distance
(statistically on par with FFP and Mash on controlled trees; behind FFP at deep
divergence), and an anomaly detector with a calibrated per-position surprise
signal — and it draws a literal 3-D map of what it has learned. For a practitioner
staring at an unfamiliar sequence with no labels, no reference, and no idea what
*k* to pick, that single-model combination — not any one number — is the useful
part.

The "grow where surprised" rule connects two literatures — CGR geometry and
predictive coding — that are rarely used together, and gives the model a
biologically-flavoured behaviour: it *decides* where to allocate memory based on
its own prediction failures, rather than tiling space uniformly.

---

## 5. Limitations and honest caveats

* **Not SOTA on any single task.** On stationary Markov data, matched fixed-order
  models and context-tree methods (PPM, CTW) beat FPC on compression; on
  deep-divergence real phylogeny, FFP beats it; on short protein families it does
  not beat a k-mer baseline. FPC trades peak per-task performance for generality
  and interpretability.
* **No individual component is novel.** As §1.1 makes explicit, CGR, self-organising
  maps, the least-surprised-model classifier, and surprise = MDL are all prior art;
  the contribution is only their unification. Readers wanting a from-scratch
  invention will not find one here.
* **Data-hungry (learned model).** FPC needs enough sequence to learn its
  generative model; on short, few-example classes (e.g. the ~150 aa proteins of
  §3.8) simple k-mer composition wins.
* **Bounded, geometry-limited context.** Effective order is capped by `order` and
  by CGR's exponentially-decaying memory; very long-range dependencies (e.g.
  distal regulatory grammar) are not captured.
* **Self-anomaly detection has a regime.** A long, internally-consistent insert
  can be *learned* during a single pass and cease to look anomalous;
  reference-based/relative detection is the right tool there.
* **Partial community-benchmark coverage.** We report the standard Robinson–Foulds
  metric on a real mammalian-mitogenome dataset (§3.9), where FPC ties Mash and
  FFP, but a *full* submission to the **AFproject** platform (Zielezinski et al.
  2019 — all 5 task categories, 24 tools, their exact reference trees) remains
  future work, as do larger multi-taxon and read-technology studies; `fpc/fetch.py`
  makes the data collection straightforward.
* **Cost.** Frozen scoring is vectorised via a KD-tree; online learning is still
  `O(T · m)` in the prototype count `m`, which we cap by usage-pruning.

---

## 6. Reproducibility

```
pip install -e .[all]
pytest -q                       # 15 correctness tests, incl. the CGR property
python benchmarks/bench.py      # synthetic experiments + figures 1-5
python benchmarks/phylo_stats.py# 25-tree CIs + Wilcoxon vs Mash/FFP (Fig 9)
python benchmarks/rf_benchmark.py# Robinson-Foulds on real mitogenomes (Fig 11)
python examples/big_study.py    # 28 real viral genomes, 5 families (Figs 7, 8)
python examples/protein_study.py# 40 real proteins, 4 families (Fig 10)
python examples/make_visuals.py # the interactive 3-D artifacts
```

Everything is deterministic under fixed seeds. The core engine (`fpc/`) imports
only NumPy and SciPy; real data is fetched once from NCBI and cached for offline
re-runs.

---

## Data and code availability

All code, tests, benchmarks, and figure-generating scripts are open-source under
the MIT license at `https://github.com/[USERNAME]/genocortex` (to be archived on
Zenodo with a DOI at submission). The core engine depends only on NumPy and SciPy
and runs fully offline; visualisation, benchmarking, and the optional
explanation layer are opt-in extras. All synthetic data are generated
deterministically from fixed seeds by `fpc/datasets.py`; all real sequences are
public NCBI RefSeq records, fetched once by `fpc/fetch.py` and cached, with
accessions recorded in the cached FASTA headers. Every number and figure in this
paper is reproduced by the commands in §6.

## Author contributions

[AUTHOR NAME] conceived the method, implemented the software, designed and ran
the experiments, and wrote the manuscript. [Supervisor], if added, supervised the
work and revised the manuscript. *(Update to reflect the final author list.)*

## Competing interests

The authors declare no competing interests.

## Funding

[No specific funding to declare / add grant numbers if applicable.]

## Acknowledgements

The literature review and parts of the analysis were assisted by an AI research
assistant; all methods, code, results, and claims were verified by the author(s),
and the honest reporting of the method's limitations reflects that verification.

---

## References

Entries marked **✓** were located and verified against their primary sources
during an automated, adversarially-checked literature review (24 claims confirmed
3–0); the remaining foundational entries are standard references. DOIs should be
double-checked against the publisher of record before submission.

**Closest prior art / must-compare (verified):**

- ✓ B. Alipour, K. A. Hill, L. Kari. *CGRclust: Chaos Game Representation for
  twin contrastive clustering of unlabelled DNA sequences.* BMC Genomics, 2024.
  doi:10.1186/s12864-024-11135-y
- ✓ Y. Wang, Z. Yu, C. Li (et al.). *CGRWDL: alignment-free phylogeny
  reconstruction based on Chaos Game Representation weighted by a dynamical
  language model.* Frontiers in Microbiology, 2024. doi:10.3389/fmicb.2024.1339156
- ✓ O. Neme et al. *The Self-Organized Chaos Game Representation for genomic
  signatures analysis.* Learning and Nonlinear Models (Braz. Soc. Neural
  Networks), 6(2):111–120, 2008.
- ✓ R. Begleiter, R. El-Yaniv, G. Yona. *On prediction using variable order
  Markov models.* J. Artificial Intelligence Research, 22:385–421, 2004.
  (winner-take-all "least-surprised model" classifier). arXiv:1107.0051
- ✓ D. Silva, D. Pratas, A. J. Pinho. *Efficient DNA sequence compression with
  neural networks (GeCo3).* GigaScience, 9(11):giaa119, 2020.
  doi:10.1093/gigascience/giaa119
- ✓ D. Pratas, M. Hosseini, D. Silva, A. J. Pinho. *A reference-free lossless
  compression algorithm for DNA sequences (JARVIS).* Entropy, 21(11):1074, 2019.
  doi:10.3390/e21111074
- ✓ D. La Rosa et al. *Alignment-free comparison of DNA barcodes via compression
  distances (NCD/IBD).* BMC Bioinformatics, 14(Suppl 7):S4, 2013.
  doi:10.1186/1471-2105-14-S7-S4
- ✓ P. Ferragina, R. Giancarlo, V. Greco, G. Manzini, G. Valiente.
  *Compression-based classification of biological sequences and structures via
  the Universal Similarity Metric.* BMC Bioinformatics, 8:252, 2007.
  doi:10.1186/1471-2105-8-252
- ✓ (F)CGR = k-mer equivalence: J. He, L. Kari, F. Millán-Arias. *On the
  equivalence of Frequency Chaos Game Representation and k-mer frequency
  vectors.* Springer, 2025. doi:10.1007/978-3-032-15641-9_15; arXiv:2506.22172

**Benchmark and standard tools (verified):**

- ✓ A. Zielezinski et al. *Benchmarking of alignment-free sequence comparison
  methods (AFproject).* Genome Biology, 20:144, 2019.
  doi:10.1186/s13059-019-1755-7
- ✓ B. D. Ondov et al. *Mash: fast genome and metagenome distance estimation
  using MinHash.* Genome Biology, 17:132, 2016. doi:10.1186/s13059-016-0997-x
- G. E. Sims et al. *Alignment-free genome comparison with feature frequency
  profiles (FFP).* PNAS, 106(8):2677–2682, 2009. doi:10.1073/pnas.0813249106

**Foundational:**

- H. J. Jeffrey. *Chaos game representation of gene structure.* Nucleic Acids
  Research, 18(8):2163–2170, 1990.
- J. S. Almeida et al. *Analysis of genomic sequences by Chaos Game
  Representation.* Bioinformatics, 17(5):429–437, 2001.
- J. Rissanen. *Modeling by shortest data description.* Automatica, 14:465–471, 1978.
- J. G. Cleary, I. H. Witten. *Data compression using adaptive coding and partial
  string matching (PPM).* IEEE Trans. Communications, 32(4):396–402, 1984.
- F. M. J. Willems, Y. M. Shtarkov, T. J. Tjalkens. *The context-tree weighting
  method: basic properties.* IEEE Trans. Information Theory, 41(3):653–664, 1995.
- R. P. N. Rao, D. H. Ballard. *Predictive coding in the visual cortex.* Nature
  Neuroscience, 2:79–87, 1999.
- K. Friston. *The free-energy principle: a unified brain theory?* Nature Reviews
  Neuroscience, 11:127–138, 2010.
- T. Kohonen. *Self-organizing maps.* Springer, 2001.
- B. Fritzke. *A growing neural gas network learns topologies.* NeurIPS, 1995.
- S. Vinga, J. Almeida. *Alignment-free sequence comparison — a review.*
  Bioinformatics, 19(4):513–523, 2003.
