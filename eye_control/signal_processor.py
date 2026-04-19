"""
signal_processor.py
-------------------
SignalProcessor handles signal acquisition, buffering, and optional simulation.

Supports:
  - Simulated signal generation (Gaussian-shaped spikes + noise)
  - File-based loading (single-column CSV / plain text / NumPy .npy)
  - Configurable sampling rate (default 250 Hz)
  - Fixed-size rolling buffer for streaming use
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Generator, Optional, Sequence

import numpy as np


class SignalProcessor:
    """Acquire, buffer, and optionally simulate a 1-D signal.

    Parameters
    ----------
    sample_rate : int
        Sampling frequency in Hz.  Default is 250 Hz.
    buffer_size : int
        Number of most-recent samples kept in the rolling buffer.
        Default is 5 × sample_rate (5 seconds of data).
    """

    def __init__(self, sample_rate: int = 250, buffer_size: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size if buffer_size is not None else sample_rate * 5
        self._buffer: collections.deque[float] = collections.deque(maxlen=self.buffer_size)

    # ------------------------------------------------------------------
    # Buffer access
    # ------------------------------------------------------------------

    def push(self, sample: float) -> None:
        """Append one sample to the rolling buffer."""
        self._buffer.append(float(sample))

    def push_many(self, samples: Sequence[float]) -> None:
        """Append multiple samples to the rolling buffer."""
        for s in samples:
            self._buffer.append(float(s))

    def get_buffer(self) -> np.ndarray:
        """Return the current rolling buffer as a NumPy array (oldest→newest)."""
        return np.array(self._buffer, dtype=np.float64)

    def clear_buffer(self) -> None:
        """Empty the rolling buffer."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def simulate_signal(
        duration: float = 10.0,
        sample_rate: int = 250,
        n_spikes: int = 6,
        noise_std: float = 0.05,
        spike_amplitude: float = 1.0,
        spike_width_s: float = 0.08,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Generate a synthetic signal with Gaussian-shaped spikes.

        Parameters
        ----------
        duration : float
            Total signal length in seconds.
        sample_rate : int
            Samples per second.
        n_spikes : int
            Number of spike events to embed.
        noise_std : float
            Standard deviation of additive Gaussian noise.
        spike_amplitude : float
            Peak height of each spike.
        spike_width_s : float
            Approximate half-width of each spike in seconds.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        rng = np.random.default_rng(seed)
        n = int(duration * sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)
        signal = rng.normal(0.0, noise_std, size=n)

        # Place spikes at random positions (avoid edges)
        margin = int(0.1 * sample_rate)
        positions = rng.integers(margin, n - margin, size=n_spikes)
        sigma = spike_width_s * sample_rate

        for pos in positions:
            spike = spike_amplitude * np.exp(-0.5 * ((np.arange(n) - pos) / sigma) ** 2)
            signal += spike

        return signal

    @staticmethod
    def simulate_double_blink_signal(
        duration: float = 10.0,
        sample_rate: int = 250,
        n_double_blinks: int = 2,
        n_single_blinks: int = 3,
        noise_std: float = 0.05,
        spike_amplitude: float = 1.0,
        spike_width_s: float = 0.08,
        double_gap_s: float = 0.25,
        seed: Optional[int] = 7,
    ) -> np.ndarray:
        """Generate a synthetic signal with both single and double blinks.

        Double blinks are two spikes spaced ``double_gap_s`` seconds apart.
        """
        rng = np.random.default_rng(seed)
        n = int(duration * sample_rate)
        signal = rng.normal(0.0, noise_std, size=n)

        margin = int(0.15 * sample_rate)
        sigma = spike_width_s * sample_rate

        def _add_spike(arr: np.ndarray, center: int) -> None:
            arr += spike_amplitude * np.exp(
                -0.5 * ((np.arange(len(arr)) - center) / sigma) ** 2
            )

        # Single blinks
        for _ in range(n_single_blinks):
            pos = rng.integers(margin, n - margin)
            _add_spike(signal, pos)

        # Double blinks
        gap = int(double_gap_s * sample_rate)
        for _ in range(n_double_blinks):
            pos = rng.integers(margin, n - margin - gap)
            _add_spike(signal, pos)
            _add_spike(signal, pos + gap)

        return signal

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_file(path: str | Path) -> np.ndarray:
        """Load a 1-D signal from a file.

        Supported formats
        -----------------
        * ``.npy`` — NumPy binary array
        * ``.csv`` / ``.txt`` — plain single-column numeric text

        Returns
        -------
        np.ndarray, shape (n_samples,), dtype float64
        """
        path = Path(path)
        if path.suffix == ".npy":
            arr = np.load(str(path))
        else:
            arr = np.loadtxt(str(path), delimiter=",", dtype=np.float64)

        arr = arr.flatten().astype(np.float64)
        return arr

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    def stream_from_array(
        self,
        signal: np.ndarray,
        chunk_size: int = 10,
        realtime: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Yield successive chunks from a pre-loaded array, optionally with
        real-time pacing so chunks arrive at the correct sample rate.

        Parameters
        ----------
        signal : np.ndarray
            Full signal to stream.
        chunk_size : int
            Number of samples per yielded chunk.
        realtime : bool
            If True, sleep between chunks to match ``self.sample_rate``.

        Yields
        ------
        np.ndarray
            Chunk of *chunk_size* (or fewer for the last chunk) samples.
        """
        delay = chunk_size / self.sample_rate if realtime else 0.0
        for start in range(0, len(signal), chunk_size):
            chunk = signal[start : start + chunk_size]
            self.push_many(chunk)
            yield chunk
            if realtime and delay > 0:
                time.sleep(delay)
