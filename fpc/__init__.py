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
    * detect    : mutations & anomalies as localized spikes of surprise,
    * classify  : "which brain is least surprised" (a generative,
                  minimum-description-length classifier),
    * discover  : hidden structure, via the geometry of the surprise landscape.

The engine depends only on NumPy and SciPy. No API. No pretrained model.
No internet. It runs offline, forever.

License: MIT (see LICENSE).
"""

from .alphabet import Alphabet, DNA, RNA, PROTEIN
from .cgr import ChaosGameEmbedding
from .cortex import FractalPredictiveCortex
from .surprise import SurpriseTrack, Anomaly
from . import tasks, datasets, io_fasta

__all__ = [
    "Alphabet",
    "DNA",
    "RNA",
    "PROTEIN",
    "ChaosGameEmbedding",
    "FractalPredictiveCortex",
    "SurpriseTrack",
    "Anomaly",
    "tasks",
    "datasets",
    "io_fasta",
]

__version__ = "0.1.0"
