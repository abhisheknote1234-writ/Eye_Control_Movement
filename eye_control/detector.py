"""
detector.py
-----------
Detector scans a signal window for candidate events and validates them
against a trained PatternModel.

Detection algorithm
-------------------
1. Compute first derivative of the incoming window.
2. Find positions where derivative exceeds ``slope_up_thresh`` (rising edges).
3. For each rising edge, scan forward for a falling edge (derivative
   below ``slope_down_thresh``) within ``max_gap`` samples.
4. Extract the segment (with a small pad).
5. Compute features and check against model tolerance.
6. Compute template similarity and check against ``similarity_thresh``.
7. If both pass → emit a detection event.

Double-blink heuristic
-----------------------
If two detections occur within ``double_blink_gap_s`` seconds, the second
event is re-labelled as ``DOUBLE_BLINK``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .feature_extractor import FeatureExtractor
from .pattern_model import PatternModel


class Detection:
    """Lightweight result object for one detected event.

    Attributes
    ----------
    label : str
        ``"BLINK"`` or ``"DOUBLE_BLINK"`` (extensible).
    start, peak, end : int
        Sample indices within the analysed window.
    similarity : float
        Template correlation score [−1, 1].
    features : dict
        Computed feature values.
    """

    __slots__ = ("label", "start", "peak", "end", "similarity", "features")

    def __init__(
        self,
        label: str,
        start: int,
        peak: int,
        end: int,
        similarity: float,
        features: Dict[str, float],
    ) -> None:
        self.label = label
        self.start = start
        self.peak = peak
        self.end = end
        self.similarity = similarity
        self.features = features

    def __repr__(self) -> str:
        return (
            f"Detection(label={self.label!r}, start={self.start}, "
            f"peak={self.peak}, end={self.end}, sim={self.similarity:.3f})"
        )


class Detector:
    """Slope-based candidate detector with model-guided validation.

    Parameters
    ----------
    model : PatternModel
        Trained pattern model used for feature matching and template
        similarity checks.
    sample_rate : int
        Sampling rate in Hz (default 250).
    slope_up_thresh : float or None
        Minimum derivative value to count as a rising edge.
        If None, 50% of the model's ``slope_up`` median is used.
    slope_down_thresh : float or None
        Maximum derivative value to count as a falling edge.
        If None, 50% of the model's ``slope_down`` median is used.
    max_gap : int
        Maximum samples between rising and falling edge (default 100).
    similarity_thresh : float
        Minimum Pearson correlation vs average template (default 0.70).
    double_blink_gap_s : float
        If two blinks occur within this many seconds, the second is
        labelled ``DOUBLE_BLINK`` (default 0.50 s).
    pad : int
        Extra samples added on each side of a detected segment (default 5).
    """

    def __init__(
        self,
        model: PatternModel,
        sample_rate: int = 250,
        slope_up_thresh: Optional[float] = None,
        slope_down_thresh: Optional[float] = None,
        max_gap: int = 100,
        similarity_thresh: float = 0.70,
        double_blink_gap_s: float = 0.50,
        pad: int = 5,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.max_gap = max_gap
        self.similarity_thresh = similarity_thresh
        self._double_blink_gap = int(double_blink_gap_s * sample_rate)
        self.pad = pad

        self._fe = FeatureExtractor(sample_rate=sample_rate)

        # Thresholds — auto-set from model when None
        self._slope_up_thresh = slope_up_thresh
        self._slope_down_thresh = slope_down_thresh

        # Refractory period: minimum samples between accepted detections
        self._refractory = int(0.05 * sample_rate)  # 50 ms

        # History for double-blink logic
        self._last_detection_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    @property
    def slope_up_thresh(self) -> float:
        if self._slope_up_thresh is not None:
            return self._slope_up_thresh
        if (
            self.model.feature_medians is not None
            and "slope_up" in self.model.feature_medians
        ):
            # Use 50 % of the learned median so noise oscillations don't
            # trigger false candidates.
            return max(0.02, 0.50 * self.model.feature_medians["slope_up"])
        return 0.02

    @property
    def slope_down_thresh(self) -> float:
        if self._slope_down_thresh is not None:
            return self._slope_down_thresh
        if (
            self.model.feature_medians is not None
            and "slope_down" in self.model.feature_medians
        ):
            return min(-0.02, 0.50 * self.model.feature_medians["slope_down"])
        return -0.02

    @property
    def _expected_half_width(self) -> int:
        """Half-width (in samples) of the segment window around a detected peak."""
        if (
            self.model.feature_medians is not None
            and "gap" in self.model.feature_medians
        ):
            return max(5, int(self.model.feature_medians["gap"] / 2))
        return 20

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_window(
        self,
        window: np.ndarray,
        offset: int = 0,
    ) -> List[Detection]:
        """Scan one signal window and return a list of detections.

        Parameters
        ----------
        window : np.ndarray
            1-D signal chunk to analyse.
        offset : int
            Sample index of ``window[0]`` within the full signal (used so
            returned indices are absolute).

        Returns
        -------
        list of Detection
        """
        if len(window) < 4:
            return []
        if self.model.feature_medians is None:
            return []

        diff = np.diff(window.astype(np.float64))
        detections: List[Detection] = []
        last_accepted: int = -self._refractory - 1  # sample index of last accepted peak
        half = self._expected_half_width
        n = len(window)

        i = 0
        while i < len(diff) - 1:
            # Find rising edge
            if diff[i] >= self.slope_up_thresh:
                # Search for falling edge within max_gap
                found = False
                for j in range(i + 1, min(i + self.max_gap, len(diff))):
                    if diff[j] <= self.slope_down_thresh:
                        # Peak is the highest point between rising and falling edge
                        peak_rel = int(np.argmax(window[i : j + 2])) + i

                        # Refractory check
                        if peak_rel - last_accepted < self._refractory:
                            i = j + 1
                            found = True
                            break

                        # Extract a peak-centred segment of expected width.
                        # This aligns the extracted shape with the trained templates
                        # regardless of where exactly the slope edges were found.
                        start_rel = max(0, peak_rel - half)
                        end_rel = min(n - 1, peak_rel + half)
                        segment = window[start_rel : end_rel + 1]

                        feats = self._fe.extract_from_segment(segment)

                        if feats is not None and self.model.features_match(feats):
                            sim = self.model.template_similarity(segment)
                            if sim >= self.similarity_thresh:
                                abs_peak = offset + peak_rel
                                label = self._assign_label(abs_peak)
                                det = Detection(
                                    label=label,
                                    start=offset + start_rel,
                                    peak=abs_peak,
                                    end=offset + end_rel,
                                    similarity=sim,
                                    features=feats,
                                )
                                detections.append(det)
                                last_accepted = peak_rel
                                self._last_detection_idx = abs_peak

                        i = j + 1
                        found = True
                        break

                if not found:
                    i += 1
            else:
                i += 1

        return detections

    def stream_detect(
        self,
        signal: np.ndarray,
        window_size: int = 500,
        step: int = 250,
        callback: Optional[Callable[[Detection], None]] = None,
    ) -> List[Detection]:
        """Slide a window over a full signal and collect all detections.

        Parameters
        ----------
        signal : np.ndarray
            Full signal to scan.
        window_size : int
            Width of the sliding window in samples.
        step : int
            Hop between consecutive windows.
        callback : callable or None
            Optional function called for every detection as it occurs.

        Returns
        -------
        list of Detection
            All detections found, in order.
        """
        all_detections: List[Detection] = []
        seen_peaks: set = set()   # deduplicate by absolute peak index

        for start in range(0, len(signal), step):
            window = signal[start : start + window_size]
            dets = self.process_window(window, offset=start)
            for d in dets:
                if d.peak not in seen_peaks:
                    seen_peaks.add(d.peak)
                    all_detections.append(d)
                    if callback is not None:
                        callback(d)

        return all_detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_label(self, peak_idx: int) -> str:
        """Return ``DOUBLE_BLINK`` if a blink occurred recently, else ``BLINK``."""
        if (
            self._last_detection_idx is not None
            and 0 < (peak_idx - self._last_detection_idx) <= self._double_blink_gap
        ):
            return "DOUBLE_BLINK"
        return "BLINK"
