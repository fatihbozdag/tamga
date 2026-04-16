"""Smoke tests — the absolute minimum that must work."""

import tamga


def test_package_imports():
    assert hasattr(tamga, "__version__")


def test_version_is_string():
    assert isinstance(tamga.__version__, str)
    assert len(tamga.__version__) > 0
