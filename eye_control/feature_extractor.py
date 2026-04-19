"""
feature_extractor.py
--------------------
FeatureExtractor computes numerical features for a signal segment
bounded by an annotation record.

Features
--------
slope_up   : max first-order derivative (steepest upward edge)
slope_down : min first-order derivative (steepest downward edge)
amplitude  : max(segment) - min(segment)
gap        : end_index - start_index  (in samples)
energy     : sum of squared samples (optional, controlled by compute_energy parameter)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class FeatureExtractor:
    """Compute features for annotated signal segments.

    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz (stored for reference; not used in computation).
    compute_energy : bool
        Whether to include ``energy`` in the returned feature dict.
        Default True.
    """

    def __init__(self, sample_rate: int = 250, compute_energy: bool = True) -> None:
        self.sample_rate = sample_rate
        self.compute_energy = compute_energy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        signal: np.ndarray,
        annotation: Dict[str, int],
    ) -> Optional[Dict[str, float]]:
        """Extract features from one annotated segment.

        Parameters
        ----------
        signal : np.ndarray
            Full 1-D signal array.
        annotation : dict
            ``{"start": int, "peak": int, "end": int}``

        Returns
        -------
        dict or None
            Feature dictionary, or None if the segment is too short.
        """
        try:
            start = int(annotation["start"])
            end = int(annotation["end"])
        except (KeyError, TypeError, ValueError):
            return None

        segment = signal[start : end + 1]
        if len(segment) < 3:
            # Need at least 3 points for a useful derivative
            return None

        return self._compute_features(segment)

    def extract_from_segment(self, segment: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract features directly from a raw segment array.

        Parameters
        ----------
        segment : np.ndarray
            1-D signal slice.

        Returns
        -------
        dict or None
        """
        segment = np.asarray(segment, dtype=np.float64)
        if len(segment) < 3:
            return None
        return self._compute_features(segment)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Core feature computation from a 1-D segment."""
        diff = np.diff(segment)
        slope_up = float(np.max(diff))
        slope_down = float(np.min(diff))
        amplitude = float(np.max(segment) - np.min(segment))
        gap = float(len(segment) - 1)   # end - start in samples

        features: Dict[str, float] = {
            "slope_up": slope_up,
            "slope_down": slope_down,
            "amplitude": amplitude,
            "gap": gap,
        }

        if self.compute_energy:
            features["energy"] = float(np.sum(segment ** 2))

        return features
