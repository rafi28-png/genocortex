"""
Correctness tests for the Fractal Predictive Cortex.

These lock in the properties the whole method relies on, so a refactor that
quietly breaks the science fails loudly instead.
"""
import numpy as np
import pytest

from fpc import DNA, PROTEIN, ChaosGameEmbedding, FractalPredictiveCortex
from fpc import tasks, datasets, io_fasta


# --------------------------------------------------------------------- #
#  alphabet                                                             #
# --------------------------------------------------------------------- #
def test_encode_decode_roundtrip():
    s = "ACGTACGTGGCA"
    assert DNA.decode(DNA.encode(s)) == s


def test_wildcards_map_safely():
    ids = DNA.encode("ACGTNNXY-")
    assert (ids[4:] == DNA.wildcard_id).all()
    assert DNA.clean_fraction(ids) == pytest.approx(4 / 9)


def test_protein_alphabet_size():
    assert PROTEIN.size == 20 and PROTEIN.full_size == 21


# --------------------------------------------------------------------- #
#  chaos game embedding -- the core geometric property                  #
# --------------------------------------------------------------------- #
def test_cgr_points_in_unit_square():
    emb = ChaosGameEmbedding(DNA)
    P = emb.embed(DNA.encode(datasets.random_sequence(2000, DNA, seed=1)))
    assert P.min() >= -1e-9 and P.max() <= 1 + 1e-9


def test_cgr_shared_suffix_are_close():
    """Two positions sharing a long suffix must land near each other -- the
    property that lets geometry stand in for context."""
    emb = ChaosGameEmbedding(DNA)
    a = DNA.encode("TTTTTTTTACGTACGT")   # shares suffix ...ACGTACGT
    b = DNA.encode("AAAACCCCACGTACGT")   # same suffix, different prefix
    pa = emb.embed(a)[-1]
    pb = emb.embed(b)[-1]
    # different suffix control
    c = emb.embed(DNA.encode("AAAACCCCTGCATGCA"))[-1]
    # Sharing a k=8 suffix forces distance <= ~(1-r)^8 * prefix-gap = 0.5^8*sqrt(2)
    # ~ 0.0055; the different-suffix control is ~20x larger. That >10x gap IS the
    # "geometry encodes recent context" property the engine depends on.
    assert np.linalg.norm(pa - pb) < 0.01
    assert np.linalg.norm(pa - c) > 0.1
    assert np.linalg.norm(pa - c) > 10 * np.linalg.norm(pa - pb)


def test_cgr_protein_generalises():
    emb = ChaosGameEmbedding(PROTEIN)
    P = emb.embed(PROTEIN.encode("MKV" * 100))
    assert P.shape == (300, 2) and np.isfinite(P).all()


# --------------------------------------------------------------------- #
#  the cortex learns                                                    #
# --------------------------------------------------------------------- #
def test_learns_structure_but_not_noise():
    struct = "ACGTACGTTTGCA" * 300
    rand = datasets.random_sequence(len(struct), DNA, seed=3)
    cs = FractalPredictiveCortex(DNA, order=8, seed=0)
    cr = FractalPredictiveCortex(DNA, order=8, seed=0)
    ts = cs.fit(DNA.encode(struct))
    tr = cr.fit(DNA.encode(rand))
    assert ts.mean_bits < 0.5            # compresses structure hard
    assert tr.mean_bits > 1.8            # random DNA stays near 2 bits
    assert ts.mean_bits < tr.mean_bits


def test_online_learning_curve_decreases():
    seq = datasets.markov_sequence(8000, DNA, order=2, concentration=0.15, seed=5)
    cx = FractalPredictiveCortex(DNA, order=8, seed=0)
    t = cx.fit(DNA.encode(seq))
    b = t.bits[np.isfinite(t.bits)]
    assert b[:300].mean() > b[-300:].mean()   # got better over the pass


def test_score_is_deterministic_and_frozen():
    seq = DNA.encode(datasets.markov_sequence(3000, DNA, seed=1))
    cx = FractalPredictiveCortex(DNA, order=7, seed=0)
    cx.fit(seq)
    n_before = cx.n_prototypes
    t1 = cx.score(seq)
    t2 = cx.score(seq)
    assert cx.n_prototypes == n_before             # scoring does not learn
    assert np.allclose(t1.bits, t2.bits, equal_nan=True)


def test_prediction_beats_chance():
    seq = datasets.markov_sequence(12000, DNA, order=2, concentration=0.12, seed=7)
    tr, te = seq[:9000], seq[9000:]
    cx = FractalPredictiveCortex(DNA, order=8, seed=0)
    cx.fit(DNA.encode(tr))
    ev = tasks.evaluate_prediction(cx, te)
    assert ev["accuracy"] > 0.5                    # chance is 0.25


# --------------------------------------------------------------------- #
#  the four abilities                                                   #
# --------------------------------------------------------------------- #
def test_classifier_separates_species():
    ds = datasets.make_species(n_species=3, per_species=5, length=2000,
                               order=2, concentration=0.2, seed=1)
    Xtr, ytr, Xte, yte = [], [], [], []
    from collections import defaultdict
    buck = defaultdict(list)
    for s, l in zip(ds.sequences, ds.labels):
        buck[l].append(s)
    for l, seqs in buck.items():
        Xtr += seqs[:3]; ytr += [l] * 3
        Xte += seqs[3:]; yte += [l] * (len(seqs) - 3)
    clf = tasks.FPCClassifier(DNA, order=6).fit(Xtr, ytr)
    acc = tasks.classification_accuracy(clf.predict(Xte), yte)
    assert acc >= 0.8


def test_distance_matrix_is_metric_like():
    seqs = [datasets.markov_sequence(1500, DNA, seed=i) for i in range(4)]
    D = tasks.distance_matrix(seqs, DNA, order=5)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)
    assert (D >= -1e-9).all()


def test_mds_embedding_shape():
    seqs = [datasets.markov_sequence(1200, DNA, seed=i) for i in range(5)]
    D = tasks.distance_matrix(seqs, DNA, order=5)
    coords = tasks.mds_embedding(D, dims=3)
    assert coords.shape == (5, 3) and np.isfinite(coords).all()


def test_detect_reference_finds_insert():
    host = datasets.markov_sequence(5000, DNA, order=2, concentration=0.15, seed=2)
    donor = datasets.markov_sequence(400, DNA, order=3, concentration=0.4, seed=99)
    chimera, (a, b) = datasets.insert_foreign(host, donor, position=2500, seed=0)
    res = tasks.detect_against_reference(host, chimera, DNA, order=8,
                                         window=40, z_threshold=3.0)
    assert any(not (an.end < a or an.start > b) for an in res.anomalies)


# --------------------------------------------------------------------- #
#  io                                                                   #
# --------------------------------------------------------------------- #
def test_fasta_roundtrip(tmp_path):
    recs = [("seq1", "ACGTACGT"), ("seq2", "GGGGCCCCAAAA")]
    p = tmp_path / "x.fasta"
    io_fasta.write_fasta(str(p), recs)
    back = io_fasta.read_fasta(str(p))
    assert len(back) == 2
    assert back[0].id == "seq1" and back[0].seq == "ACGTACGT"
    assert back[1].seq == "GGGGCCCCAAAA"
