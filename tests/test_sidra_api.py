from __future__ import annotations

from brepi.sources.sidra.api import ValueStatus, classify_value


def test_classify_value_preserves_two_published_zero_semantics():
    assert classify_value("-") == ("-", 0.0, ValueStatus.ABSOLUTE_ZERO)
    assert classify_value("0") == ("0", 0.0, ValueStatus.ZERO_OR_ROUNDED)


def test_classify_value_never_turns_missing_or_suppressed_into_zero():
    assert classify_value("...")[1:] == (None, ValueStatus.NOT_AVAILABLE)
    assert classify_value("X")[1:] == (None, ValueStatus.SUPPRESSED)
