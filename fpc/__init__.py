"""
GenoCortex - the Fractal Predictive Cortex (FPC)
================================================

A training-free, alignment-free, self-organizing "brain" for biological
sequences. It maps any symbolic sequence (DNA, RNA, protein -- it is
alphabet-agnostic) into a continuous *fractal mind-space* via Chaos Game
Representation, grows its own internal predictive model online, and minimizes
a single quantity -- *surprise* (the negative log-probability of what comes
next) -- to simultaneously:

    * predict   : what symbol / region comes next,
    * detect    : mutations, horizontal gene transfer & pathogenicity islands,
    * classify  : "which brain is least surprised" (a strand-aware generative,
                  minimum-description-length classifier),
    * discover  : Neighbor-Joining phylogenetics & 3-D surprise geometry.

The engine depends only on NumPy and SciPy (with an optional ANSI C PEP 3118
spatial-grid kernel ``fpc._fast`` for megabase-per-second throughput). No API.
No pretrained model. No internet required.

License: MIT (see LICENSE).
"""

from .alphabet import Alphabet, DNA, RNA, PROTEIN, BUILTIN_ALPHABETS, get_alphabet
from .cgr import ChaosGameEmbedding
from .cortex import FractalPredictiveCortex, HAS_FAST_KERNEL
from .surprise import SurpriseTrack, Anomaly, benjamini_hochberg
from . import tasks, datasets, io_fasta, io_seq, phylo, islands

__all__ = [
    "Alphabet",
    "DNA",
    "RNA",
    "PROTEIN",
    "BUILTIN_ALPHABETS",
    "get_alphabet",
    "ChaosGameEmbedding",
    "FractalPredictiveCortex",
    "HAS_FAST_KERNEL",
    "SurpriseTrack",
    "Anomaly",
    "benjamini_hochberg",
    "tasks",
    "datasets",
    "io_fasta",
    "io_seq",
    "phylo",
    "islands",
]

__version__ = "1.0.0"
