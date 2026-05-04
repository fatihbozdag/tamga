"""Tests for the logger helper."""

import logging

from bitig.plumbing.logging import get_logger, set_verbosity


def test_get_logger_returns_logger():
    log = get_logger("bitig.test")
    assert isinstance(log, logging.Logger)
    assert log.name == "bitig.test"


def test_get_logger_is_idempotent():
    a = get_logger("bitig.test")
    b = get_logger("bitig.test")
    assert a is b


def test_set_verbosity_changes_level():
    set_verbosity("DEBUG")
    log = get_logger("bitig.test")
    assert log.isEnabledFor(logging.DEBUG)

    set_verbosity("WARNING")
    assert not log.isEnabledFor(logging.DEBUG)
    assert log.isEnabledFor(logging.WARNING)
