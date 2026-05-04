"""Tests for deterministic per-method seed derivation."""

import numpy as np

from bitig.plumbing.seeds import derive_rng, derive_seed


def test_derive_seed_is_deterministic():
    assert derive_seed(42, "burrows") == derive_seed(42, "burrows")


def test_derive_seed_differs_by_method_id():
    assert derive_seed(42, "burrows") != derive_seed(42, "consensus")


def test_derive_seed_differs_by_study_seed():
    assert derive_seed(42, "burrows") != derive_seed(43, "burrows")


def test_derive_seed_is_in_uint32_range():
    for seed in (0, 1, 42, 123456, 2**31 - 1):
        for method in ("a", "b", "consensus-run-0"):
            derived = derive_seed(seed, method)
            assert 0 <= derived < 2**32


def test_derive_rng_returns_numpy_generator():
    rng = derive_rng(42, "burrows")
    assert isinstance(rng, np.random.Generator)


def test_derive_rng_produces_reproducible_draws():
    r1 = derive_rng(42, "burrows").integers(0, 100, size=10)
    r2 = derive_rng(42, "burrows").integers(0, 100, size=10)
    assert np.array_equal(r1, r2)


def test_derive_rng_produces_different_draws_for_different_methods():
    r1 = derive_rng(42, "burrows").integers(0, 10**6, size=10)
    r2 = derive_rng(42, "consensus").integers(0, 10**6, size=10)
    assert not np.array_equal(r1, r2)
