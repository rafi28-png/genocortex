"""
Self-contained, reproducible biological-sequence generators.
============================================================

No downloads, no internet -- yet enough biological realism and, crucially,
*known ground truth* to run controlled, publishable experiments:

* ``random_sequence``     -- an incompressible baseline (max-entropy DNA).
* ``markov_sequence``     -- a fixed variable-order Markov "organism".
* ``make_species``        -- several organisms with distinct statistics, for
                             label-free classification / clustering benchmarks.
* ``evolve``              -- inject point mutations / indels at known positions,
                             the ground truth for anomaly detection.
* ``insert_foreign``      -- splice a foreign segment into a host (horizontal
                             transfer), a localized-anomaly ground truth.
* ``phylogeny``           -- evolve a genome down a known binary tree, giving
                             leaf genomes with a ground-truth topology for
                             phylogenetic-clustering evaluation.

Controlled synthetic data with ground truth is not a weakness of a methods
paper -- it is how you *prove* the method recovers what it should before you
turn it loose on messy real genomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .alphabet import Alphabet, DNA


# --------------------------------------------------------------------- #
#  primitive generators                                                 #
# --------------------------------------------------------------------- #
def random_sequence(length: int, alphabet: Alphabet = DNA,
                    gc: float | None = None, seed: int = 0) -> str:
    """A max-entropy sequence. For DNA, ``gc`` sets the G+C fraction."""
    rng = np.random.default_rng(seed)
    a = alphabet.size
    if gc is not None and alphabet is DNA:
        # A, C, G, T  ->  weight C,G by gc/2 and A,T by (1-gc)/2.
        p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    else:
        p = np.full(a, 1.0 / a)
    ids = rng.choice(a, size=length, p=p)
    return alphabet.decode(ids)


def _random_transition(a: int, order: int, concentration: float,
                       rng: np.random.Generator) -> np.ndarray:
    """A random order-``order`` transition table, shape ``(a**order, a)``.

    ``concentration`` is the Dirichlet parameter: small -> peaky/deterministic
    (very learnable, compressible); large -> flat/random (hard, high entropy).
    """
    n_ctx = a ** order
    return rng.dirichlet(np.full(a, concentration), size=n_ctx)


def markov_sequence(length: int, alphabet: Alphabet = DNA, order: int = 2,
                    concentration: float = 0.3, seed: int = 0,
                    transition: np.ndarray | None = None) -> str:
    """Generate a sequence from a fixed variable-order Markov "organism"."""
    rng = np.random.default_rng(seed)
    a = alphabet.size
    if transition is None:
        transition = _random_transition(a, order, concentration, rng)
    ctx = 0
    out = np.empty(length, dtype=np.int64)
    # Seed the first `order` symbols uniformly.
    for i in range(min(order, length)):
        out[i] = rng.integers(a)
        ctx = (ctx * a + int(out[i])) % (a ** order)
    for i in range(order, length):
        nxt = rng.choice(a, p=transition[ctx])
        out[i] = nxt
        ctx = (ctx * a + int(nxt)) % (a ** order)
    return alphabet.decode(out)


def related_transition(base: np.ndarray, strength: float, concentration: float,
                       seed: int) -> np.ndarray:
    """A transition table that is a *blend* of ``base`` and fresh randomness.

    ``strength`` in [0, 1] is how far the result drifts from ``base``. Small
    strength -> a close "cousin" (shares most low-order composition, differs in
    the fine structure) -- the hard case where raw k-mer novelty is blind but a
    predictive model still feels the shift.
    """
    rng = np.random.default_rng(seed)
    a = base.shape[1]
    noise = rng.dirichlet(np.full(a, concentration), size=base.shape[0])
    mixed = (1.0 - strength) * base + strength * noise
    return mixed / mixed.sum(axis=1, keepdims=True)


def species_transitions(n_species: int, alphabet: Alphabet = DNA, order: int = 2,
                        concentration: float = 0.25, seed: int = 0) -> list:
    """Fixed transition models, one per species (reusable across samples)."""
    rng = np.random.default_rng(seed)
    return [_random_transition(alphabet.size, order, concentration, rng)
            for _ in range(n_species)]


def heterogeneous_genome(alphabet: Alphabet = DNA, seed: int = 0,
                         block: int = 1200) -> str:
    """A genome no single fixed order models well.

    It stitches together blocks of very different character: a low-order region,
    a high-order region, a tandem-repeat (satellite) region, a low-complexity
    homopolymer-ish region, and a near-random region. A fixed-k Markov model
    must compromise -- too small a k underfits the high-order block, too large a
    k overfits everywhere else -- whereas an adaptive model can spend resolution
    only where it pays off. This is the honest setting where adaptivity wins.
    """
    rng = np.random.default_rng(seed)
    a = alphabet.size
    parts = []
    # 1. low-order structure
    parts.append(markov_sequence(block, alphabet, order=1, concentration=0.08,
                                 seed=int(rng.integers(1 << 30))))
    # 2. high-order structure
    parts.append(markov_sequence(block, alphabet, order=4, concentration=0.25,
                                 seed=int(rng.integers(1 << 30))))
    # 3. tandem repeat (a satellite): a random unit repeated
    unit_len = int(rng.integers(6, 16))
    unit = alphabet.decode(rng.integers(a, size=unit_len))
    reps = block // unit_len + 1
    parts.append((unit * reps)[:block])
    # 4. low-complexity: biased toward one or two symbols
    p = np.full(a, 0.03); p[rng.integers(a)] = 1.0; p /= p.sum()
    parts.append(alphabet.decode(rng.choice(a, size=block, p=p)))
    # 5. near-random
    parts.append(random_sequence(block, alphabet, seed=int(rng.integers(1 << 30))))
    return "".join(parts)


# --------------------------------------------------------------------- #
#  classification / clustering benchmark                                #
# --------------------------------------------------------------------- #
@dataclass
class LabeledSet:
    sequences: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    alphabet: Alphabet = DNA

    def __len__(self):  # pragma: no cover
        return len(self.sequences)


def make_species(n_species: int = 4, per_species: int = 6, length: int = 3000,
                 alphabet: Alphabet = DNA, order: int = 2,
                 concentration: float = 0.25, seed: int = 0) -> LabeledSet:
    """Several "species", each a distinct Markov organism, many samples each.

    Same-species samples share a transition model (differing only by RNG draw);
    different species have different models. Ground-truth labels come free, so
    you can score classification accuracy and clustering purity honestly.
    """
    rng = np.random.default_rng(seed)
    a = alphabet.size
    ds = LabeledSet(alphabet=alphabet)
    for sp in range(n_species):
        trans = _random_transition(a, order, concentration, rng)
        label = f"species_{sp}"
        for k in range(per_species):
            sub = int(rng.integers(0, 2**31 - 1))
            seq = markov_sequence(length, alphabet, order, seed=sub, transition=trans)
            ds.sequences.append(seq)
            ds.labels.append(label)
    return ds


# --------------------------------------------------------------------- #
#  mutation / anomaly benchmark                                         #
# --------------------------------------------------------------------- #
@dataclass
class MutationTruth:
    sequence: str                 # the mutated sequence
    substitutions: list           # positions (in mutated coords) of point subs
    inserted: list                # (start, end) spans that were inserted
    deleted: list                 # positions (in mutated coords) next to deletions


def evolve(seq: str, sub_rate: float = 0.0, ins_rate: float = 0.0,
           del_rate: float = 0.0, alphabet: Alphabet = DNA,
           seed: int = 0) -> MutationTruth:
    """Apply point substitutions / insertions / deletions at random sites.

    Returns the mutated sequence *and* the ground-truth positions of every
    change (in the coordinates of the mutated sequence), for scoring a detector.
    """
    rng = np.random.default_rng(seed)
    a = alphabet.size
    ids = list(alphabet.encode(seq))
    out: list[int] = []
    subs: list[int] = []
    dels: list[int] = []
    ins: list[tuple] = []
    for sym in ids:
        # deletion
        if rng.random() < del_rate:
            dels.append(len(out))
            continue
        # substitution
        if rng.random() < sub_rate and sym < a:
            choices = [x for x in range(a) if x != sym]
            sym = int(rng.choice(choices))
            subs.append(len(out))
        out.append(int(sym))
        # insertion after this symbol
        if rng.random() < ins_rate:
            start = len(out)
            k = 1 + int(rng.integers(0, 3))
            for _ in range(k):
                out.append(int(rng.integers(a)))
            ins.append((start, len(out) - 1))
    return MutationTruth(alphabet.decode(out), subs, ins, dels)


def insert_foreign(host: str, donor: str, position: int | None = None,
                   seed: int = 0) -> tuple[str, tuple]:
    """Splice ``donor`` into ``host`` (horizontal-transfer style).

    Returns the chimeric sequence and the ``(start, end)`` span of the insert
    in the new coordinates -- the ground-truth region a detector should flag.
    """
    rng = np.random.default_rng(seed)
    if position is None:
        position = int(rng.integers(0, len(host)))
    chimera = host[:position] + donor + host[position:]
    return chimera, (position, position + len(donor) - 1)


# --------------------------------------------------------------------- #
#  phylogeny benchmark                                                  #
# --------------------------------------------------------------------- #
@dataclass
class Phylogeny:
    leaves: dict                  # name -> sequence
    truth_distance: dict          # (name_i, name_j) -> patristic distance
    newick: str                   # ground-truth topology


def phylogeny(root_length: int = 4000, depth: int = 3, branch_sub: float = 0.03,
              alphabet: Alphabet = DNA, order: int = 2,
              concentration: float = 0.25, seed: int = 0) -> Phylogeny:
    """Evolve a root genome down a full binary tree of the given ``depth``.

    Each edge applies ``branch_sub`` substitutions; leaves are extant taxa.
    Returns the leaf genomes, the ground-truth patristic distances, and a
    Newick string -- everything needed to score a clustering / tree against the
    real topology.
    """
    rng = np.random.default_rng(seed)
    root = markov_sequence(root_length, alphabet, order, concentration,
                           seed=int(rng.integers(0, 2**31 - 1)))

    leaves: dict = {}
    dist_to_root_path: dict = {}  # name -> list of node ids along its path

    def recurse(seq: str, name: str, level: int, path: list):
        if level == depth:
            leaves[name] = seq
            dist_to_root_path[name] = path
            return
        for b in (0, 1):
            mut = evolve(seq, sub_rate=branch_sub, alphabet=alphabet,
                         seed=int(rng.integers(0, 2**31 - 1))).sequence
            recurse(mut, name + str(b), level + 1, path + [name + str(b)])

    recurse(root, "L", 0, [])

    # Patristic distance = number of edges to the most-recent common ancestor,
    # summed on both sides = 2 * (depth - shared_prefix_length_of_names).
    names = list(leaves)
    truth: dict = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            shared = 0
            for a_, b_ in zip(ni, nj):
                if a_ == b_:
                    shared += 1
                else:
                    break
            d = 2 * (len(ni) - shared)
            truth[(ni, nj)] = d
            truth[(nj, ni)] = d

    def to_newick(prefix: str, level: int) -> str:
        if level == depth:
            return prefix
        left = to_newick(prefix + "0", level + 1)
        right = to_newick(prefix + "1", level + 1)
        return f"({left},{right})"

    newick = to_newick("L", 0) + ";"
    return Phylogeny(leaves=leaves, truth_distance=truth, newick=newick)
