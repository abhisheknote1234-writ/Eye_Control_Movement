"""
tests/test_annotator.py
-----------------------
Unit tests for Annotator (programmatic path, no GUI).
"""
import numpy as np
import pytest

from eye_control.annotator import Annotator


@pytest.fixture
def simple_signal():
    """100-sample ramp signal."""
    return np.linspace(0, 1, 100)


def test_empty_on_start(simple_signal):
    ann = Annotator(simple_signal)
    assert ann.get_annotations() == []


def test_valid_annotation(simple_signal):
    ann = Annotator(simple_signal)
    result = ann.add_annotation_programmatic(10, 50, 90)
    assert result is True
    recs = ann.get_annotations()
    assert len(recs) == 1
    assert recs[0] == {"start": 10, "peak": 50, "end": 90}


def test_multiple_annotations(simple_signal):
    ann = Annotator(simple_signal)
    ann.add_annotation_programmatic(5, 20, 35)
    ann.add_annotation_programmatic(50, 60, 80)
    recs = ann.get_annotations()
    assert len(recs) == 2


def test_invalid_order_start_after_end(simple_signal):
    """start > end should be rejected."""
    ann = Annotator(simple_signal)
    result = ann.add_annotation_programmatic(80, 50, 10)
    assert result is False
    assert ann.get_annotations() == []


def test_invalid_peak_outside_range(simple_signal):
    """peak > end should be rejected (start ≤ peak ≤ end rule)."""
    ann = Annotator(simple_signal)
    result = ann.add_annotation_programmatic(10, 90, 50)
    assert result is False


def test_zero_length_segment(simple_signal):
    """start == end should be rejected."""
    ann = Annotator(simple_signal)
    result = ann.add_annotation_programmatic(30, 30, 30)
    assert result is False


def test_clear_annotations(simple_signal):
    ann = Annotator(simple_signal)
    ann.add_annotation_programmatic(0, 10, 20)
    ann.clear_annotations()
    assert ann.get_annotations() == []


def test_indices_clamped_to_signal_bounds(simple_signal):
    """Indices beyond signal length should be clamped, not raise."""
    ann = Annotator(simple_signal)
    # end > len(signal) - 1 → clamped to 99
    result = ann.add_annotation_programmatic(0, 50, 500)
    assert result is True
    recs = ann.get_annotations()
    assert recs[0]["end"] == 99   # clamped
