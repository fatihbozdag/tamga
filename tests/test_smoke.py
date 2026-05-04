"""Smoke tests — the absolute minimum that must work."""

import bitig


def test_package_imports():
    assert hasattr(bitig, "__version__")


def test_version_is_string():
    assert isinstance(bitig.__version__, str)
    assert len(bitig.__version__) > 0
