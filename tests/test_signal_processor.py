"""
tests/test_signal_processor.py
--------------------------------
Unit tests for SignalProcessor.
"""
import numpy as np
import pytest

from eye_control.signal_processor import SignalProcessor


def test_push_and_get_buffer():
    sp = SignalProcessor(sample_rate=100, buffer_size=10)
    sp.push(1.0)
    sp.push(2.0)
    buf = sp.get_buffer()
    assert len(buf) == 2
    assert buf[0] == pytest.approx(1.0)
    assert buf[1] == pytest.approx(2.0)


def test_buffer_rolls_over():
    sp = SignalProcessor(sample_rate=100, buffer_size=5)
    for i in range(8):
        sp.push(float(i))
    buf = sp.get_buffer()
    assert len(buf) == 5
    # oldest 3 dropped, last 5 remain: 3,4,5,6,7
    assert list(buf) == pytest.approx([3.0, 4.0, 5.0, 6.0, 7.0])


def test_push_many():
    sp = SignalProcessor(sample_rate=50, buffer_size=100)
    sp.push_many([10.0, 20.0, 30.0])
    assert len(sp.get_buffer()) == 3


def test_clear_buffer():
    sp = SignalProcessor()
    sp.push_many([1.0, 2.0, 3.0])
    sp.clear_buffer()
    assert len(sp.get_buffer()) == 0


def test_simulate_signal_shape():
    sig = SignalProcessor.simulate_signal(duration=2.0, sample_rate=250, n_spikes=3, seed=0)
    assert sig.shape == (500,)
    assert sig.dtype == np.float64


def test_simulate_signal_has_spikes():
    sig = SignalProcessor.simulate_signal(duration=5.0, sample_rate=250, n_spikes=5, seed=1)
    # At least one value above 0.5 (the spike amplitude is 1.0)
    assert np.max(sig) > 0.5


def test_simulate_double_blink():
    sig = SignalProcessor.simulate_double_blink_signal(
        duration=5.0, sample_rate=250, n_double_blinks=2, n_single_blinks=2, seed=3
    )
    assert len(sig) == 1250
    assert np.max(sig) > 0.5


def test_stream_from_array():
    sp = SignalProcessor(sample_rate=50)
    data = np.arange(100, dtype=float)
    chunks = list(sp.stream_from_array(data, chunk_size=10, realtime=False))
    # 10 chunks of 10 samples each
    assert len(chunks) == 10
    assert len(chunks[0]) == 10
    # Buffer should contain all 100 samples
    assert len(sp.get_buffer()) == 100


def test_load_from_npy(tmp_path):
    arr = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "sig.npy"
    np.save(str(p), arr)
    loaded = SignalProcessor.load_from_file(p)
    np.testing.assert_array_almost_equal(loaded, arr)


def test_load_from_csv(tmp_path):
    p = tmp_path / "sig.csv"
    p.write_text("1.0\n2.0\n3.0\n")
    loaded = SignalProcessor.load_from_file(p)
    np.testing.assert_array_almost_equal(loaded, [1.0, 2.0, 3.0])
