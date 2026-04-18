"""Forensic-linguistics toolkit — authorship verification, LR-framed reporting, and related
methods aimed at forensic research and evidentiary-standard output.

Every method here is designed around forensic assumptions: small known/questioned document
sets, open-set rather than closed-set decisions, and reporting suitable for journals such as
*International Journal of Speech, Language and the Law*, *Language and Law*, and *Applied
Linguistics*.
"""

from tamga.forensic.char_ngrams import CategorizedCharNgramExtractor, classify_ngram
from tamga.forensic.distortion import distort_corpus, distort_text
from tamga.forensic.lr import (
    CalibratedScorer,
    log_lr_from_probs,
    log_lr_from_probs_with_priors,
)
from tamga.forensic.metrics import brier, cllr, ece, tippett
from tamga.forensic.verify import GeneralImpostors

__all__ = [
    "CalibratedScorer",
    "CategorizedCharNgramExtractor",
    "GeneralImpostors",
    "brier",
    "classify_ngram",
    "cllr",
    "distort_corpus",
    "distort_text",
    "ece",
    "log_lr_from_probs",
    "log_lr_from_probs_with_priors",
    "tippett",
]
