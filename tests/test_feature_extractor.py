"""
tests/test_feature_extractor.py
--------------------------------
Unit tests for FeatureExtractor.
"""
import numpy as np
import pytest

from eye_control.feature_extractor import FeatureExtractor


@pytest.fixture
def fe():
    return FeatureExtractor(sample_rate=250, compute_energy=True)


@pytest.fixture
def spike():
    """Synthetic single Gaussian spike, 50 samples."""
    t = np.linspace(-3, 3, 50)
    return np.exp(-0.5 * t ** 2)


def test_extract_returns_dict(fe, spike):
    signal = np.zeros(200)
    signal[75:125] = spike
    ann = {"start": 75, "peak": 100, "end": 124}
    feats = fe.extract(signal, ann)
    assert feats is not None
    for key in ("slope_up", "slope_down", "amplitude", "gap", "energy"):
        assert key in feats


def test_slope_up_positive(fe, spike):
    """Rising edge → slope_up should be positive."""
    feats = fe.extract_from_segment(spike)
    assert feats is not None
    assert feats["slope_up"] > 0


def test_slope_down_negative(fe, spike):
    """Falling edge → slope_down should be negative."""
    feats = fe.extract_from_segment(spike)
    assert feats is not None
    assert feats["slope_down"] < 0


def test_amplitude(fe):
    seg = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    feats = fe.extract_from_segment(seg)
    assert feats is not None
    assert feats["amplitude"] == pytest.approx(1.0)


def test_gap(fe):
    seg = np.arange(10, dtype=float)
    feats = fe.extract_from_segment(seg)
    assert feats is not None
    assert feats["gap"] == pytest.approx(9.0)   # 10 samples → gap = 9


def test_energy(fe):
    seg = np.ones(5)
    feats = fe.extract_from_segment(seg)
    assert feats is not None
    assert feats["energy"] == pytest.approx(5.0)


def test_too_short_returns_none(fe):
    assert fe.extract_from_segment(np.array([1.0, 2.0])) is None


def test_no_energy_when_disabled():
    fe_no_e = FeatureExtractor(compute_energy=False)
    seg = np.array([0.0, 1.0, 0.5, 0.0, -0.2, 0.1])
    feats = fe_no_e.extract_from_segment(seg)
    assert feats is not None
    assert "energy" not in feats


def test_missing_annotation_key(fe):
    signal = np.ones(100)
    result = fe.extract(signal, {"start": 10})  # missing "end"
    assert result is None
