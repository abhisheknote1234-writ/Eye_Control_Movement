"""
tests/test_pattern_model.py
----------------------------
Unit tests for PatternModel.
"""
import json
import numpy as np
import pytest

from eye_control.signal_processor import SignalProcessor
from eye_control.annotator import Annotator
from eye_control.pattern_model import PatternModel


@pytest.fixture
def signal_and_annotations():
    """Simulated signal with known spike positions annotated."""
    sig = SignalProcessor.simulate_signal(
        duration=5.0, sample_rate=250, n_spikes=4, seed=42
    )
    # Auto-annotate via scipy peaks
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(sig, height=0.5, distance=30)
    except ImportError:
        peaks = np.array([125, 375, 625, 875])

    annotations = []
    hw = 20
    for p in peaks[:4]:
        p = int(p)
        annotations.append({
            "start": max(0, p - hw),
            "peak": p,
            "end": min(len(sig) - 1, p + hw),
        })
    return sig, annotations


def test_train_succeeds(signal_and_annotations):
    sig, anns = signal_and_annotations
    model = PatternModel()
    ok = model.train(sig, anns)
    assert ok is True
    assert model.feature_medians is not None
    assert model.avg_template is not None
    assert model.n_training_samples == len(anns)


def test_train_with_no_annotations():
    model = PatternModel()
    sig = np.ones(100)
    result = model.train(sig, [])
    assert result is False


def test_features_match_trained_values(signal_and_annotations):
    sig, anns = signal_and_annotations
    model = PatternModel(tolerance=0.5)
    model.train(sig, anns)
    # The median features themselves should match (within tolerance)
    assert model.features_match(model.feature_medians) is True


def test_features_out_of_tolerance(signal_and_annotations):
    sig, anns = signal_and_annotations
    model = PatternModel(tolerance=0.30)
    model.train(sig, anns)
    # Wildly different features should NOT match
    bad = {k: v * 100 for k, v in model.feature_medians.items()}
    assert model.features_match(bad) is False


def test_template_similarity_self(signal_and_annotations):
    sig, anns = signal_and_annotations
    model = PatternModel()
    model.train(sig, anns)
    assert model.avg_template is not None
    # Average template vs itself should give near-perfect correlation
    sim = model.template_similarity(model.avg_template)
    assert sim > 0.95


def test_template_similarity_random_returns_float(signal_and_annotations):
    sig, anns = signal_and_annotations
    model = PatternModel()
    model.train(sig, anns)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 41)
    sim = model.template_similarity(noise)
    assert isinstance(sim, float)


def test_save_load_json(signal_and_annotations, tmp_path):
    sig, anns = signal_and_annotations
    model = PatternModel()
    model.train(sig, anns)
    path = tmp_path / "model.json"
    model.save_json(path)

    loaded = PatternModel()
    loaded.load_json(path)

    assert loaded.feature_medians == pytest.approx(model.feature_medians, rel=1e-6)
    np.testing.assert_array_almost_equal(loaded.avg_template, model.avg_template)


def test_save_load_pickle(signal_and_annotations, tmp_path):
    sig, anns = signal_and_annotations
    model = PatternModel()
    model.train(sig, anns)
    path = tmp_path / "model.pkl"
    model.save_pickle(path)

    loaded = PatternModel()
    loaded.load_pickle(path)

    assert loaded.feature_medians == pytest.approx(model.feature_medians, rel=1e-6)
    np.testing.assert_array_almost_equal(loaded.avg_template, model.avg_template)


def test_recalibrate(signal_and_annotations):
    sig, anns = signal_and_annotations
    model = PatternModel()
    model.train(sig, anns[:2])  # train on first 2

    old_medians = dict(model.feature_medians)
    # Recalibrate with a new annotation
    ok = model.recalibrate(sig, anns[2], weight=0.5)
    assert ok is True
    # Medians should have shifted slightly
    assert model.feature_medians != old_medians
