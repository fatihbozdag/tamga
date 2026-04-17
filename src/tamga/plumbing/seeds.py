"""Per-method RNG derivation.

`numpy.random.Generator` is created from a derived seed combining the study-level seed with the
method id. This gives each method an independent, reproducible random stream — reordering methods
in a study config does not affect any individual method's output.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(study_seed: int, method_id: str) -> int:
    """Return a deterministic uint32 seed from a study seed and a method id.

    Implemented via sha256 of `f"{study_seed}:{method_id}"`; takes the first 4 bytes as an
    unsigned 32-bit integer, which is what `numpy.random.default_rng` expects.
    """
    digest = hashlib.sha256(f"{study_seed}:{method_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def derive_rng(study_seed: int, method_id: str) -> np.random.Generator:
    """Return a seeded numpy Generator for a given (study_seed, method_id)."""
    return np.random.default_rng(derive_seed(study_seed, method_id))
