"""
pattern_model.py
----------------
PatternModel stores learned signal-pattern parameters and templates.

Learning strategy (semi-supervised)
-------------------------------------
* Features are aggregated across all annotated segments.
* Robust centrals are computed via **median** to handle outliers.
* Tolerance bands default to ±30 % of the median value.
* Waveform templates are stored normalised and averaged.

Save / Load
-----------
* ``save_json(path)``  / ``load_json(path)``  — human-readable JSON
* ``save_pickle(path)``/ ``load_pickle(path)`` — full fidelity (includes numpy arrays)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .feature_extractor import FeatureExtractor

# Default template length (in samples) for normalised resampling
_DEFAULT_TEMPLATE_LEN = 50


class PatternModel:
    """Semi-supervised pattern model.

    Parameters
    ----------
    tolerance : float
        Fractional tolerance around median feature values (default 0.30 = ±30 %).
    template_len : int
        All waveform templates are resampled to this length before averaging.
    feature_extractor : FeatureExtractor or None
        If None, a default one is created internally.
    """

    def __init__(
        self,
        tolerance: float = 0.30,
        template_len: int = _DEFAULT_TEMPLATE_LEN,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.tolerance = tolerance
        self.template_len = template_len
        self._fe = feature_extractor or FeatureExtractor()

        # Learned model parameters (None = not trained yet)
        self.feature_medians: Optional[Dict[str, float]] = None
        self.feature_lower: Optional[Dict[str, float]] = None
        self.feature_upper: Optional[Dict[str, float]] = None

        # Template store
        self._raw_templates: List[np.ndarray] = []
        self.avg_template: Optional[np.ndarray] = None   # length == template_len
        self._n_templates: int = 0

        # Track number of training samples
        self.n_training_samples: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        signal: np.ndarray,
        annotations: List[Dict[str, int]],
    ) -> bool:
        """Learn model parameters from annotated segments.

        Parameters
        ----------
        signal : np.ndarray
            Full 1-D signal.
        annotations : list of dict
            Each: ``{"start": int, "peak": int, "end": int}``.

        Returns
        -------
        bool
            True if training succeeded (at least one valid segment found).
        """
        if len(annotations) == 0:
            return False

        all_features: List[Dict[str, float]] = []
        templates: List[np.ndarray] = []

        for ann in annotations:
            feats = self._fe.extract(signal, ann)
            if feats is None:
                continue
            all_features.append(feats)

            seg = signal[ann["start"] : ann["end"] + 1]
            if len(seg) >= 2:
                templates.append(self._normalize(seg))

        if not all_features:
            return False

        self._fit_features(all_features)
        self._fit_templates(templates)
        self.n_training_samples = len(all_features)
        return True

    # ------------------------------------------------------------------
    # Adaptive recalibration
    # ------------------------------------------------------------------

    def recalibrate(
        self,
        signal: np.ndarray,
        new_annotation: Dict[str, int],
        weight: float = 0.2,
    ) -> bool:
        """Incrementally update model with a newly confirmed detection.

        Uses an exponential moving average blend:
            new_median = (1 - weight) * old + weight * new_value

        Parameters
        ----------
        signal : np.ndarray
            Full current signal buffer.
        new_annotation : dict
            ``{"start": int, "peak": int, "end": int}``
        weight : float
            Blending weight for the new sample (0 < weight < 1).

        Returns
        -------
        bool
            True if recalibration succeeded.
        """
        feats = self._fe.extract(signal, new_annotation)
        if feats is None:
            return False
        if self.feature_medians is None:
            # First training — delegate
            return self.train(signal, [new_annotation])

        # Blend feature medians
        for key, val in feats.items():
            if key in self.feature_medians:
                old = self.feature_medians[key]
                blended = (1.0 - weight) * old + weight * val
                self.feature_medians[key] = blended
        self._recompute_bounds()

        # Update template store
        seg = signal[new_annotation["start"] : new_annotation["end"] + 1]
        if len(seg) >= 2:
            norm_seg = self._normalize(seg)
            self._raw_templates.append(norm_seg)
            self._n_templates += 1
            self._rebuild_avg_template()

        return True

    # ------------------------------------------------------------------
    # Feature matching
    # ------------------------------------------------------------------

    def features_match(self, features: Dict[str, float]) -> bool:
        """Check whether ``features`` fall within the learned tolerance band.

        Returns False if model is not yet trained or if *any* shared feature
        is out of range.
        """
        if self.feature_lower is None or self.feature_upper is None:
            return False

        for key in ("slope_up", "slope_down", "amplitude", "gap"):
            if key not in features:
                continue
            lo = self.feature_lower.get(key)
            hi = self.feature_upper.get(key)
            if lo is None or hi is None:
                continue
            val = features[key]
            if not (lo <= val <= hi):
                return False
        return True

    # ------------------------------------------------------------------
    # Template similarity
    # ------------------------------------------------------------------

    def template_similarity(self, segment: np.ndarray) -> float:
        """Return Pearson correlation between ``segment`` and the average
        template (both resampled to ``self.template_len``).

        Returns 0.0 if no template has been learned yet.
        """
        if self.avg_template is None:
            return 0.0

        candidate = self._resample(self._normalize(segment), self.template_len)
        reference = self.avg_template

        # Pearson correlation
        c = np.corrcoef(candidate, reference)
        r = float(c[0, 1])
        # corrcoef can return NaN for flat signals
        return 0.0 if np.isnan(r) else r

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save_json(self, path: str | Path) -> None:
        """Save model parameters (no numpy arrays) to a JSON file."""
        data: Dict[str, Any] = {
            "tolerance": self.tolerance,
            "template_len": self.template_len,
            "n_training_samples": self.n_training_samples,
            "feature_medians": self.feature_medians,
            "feature_lower": self.feature_lower,
            "feature_upper": self.feature_upper,
            "avg_template": (
                self.avg_template.tolist() if self.avg_template is not None else None
            ),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_json(self, path: str | Path) -> None:
        """Load model parameters from a JSON file."""
        data = json.loads(Path(path).read_text())
        self.tolerance = data.get("tolerance", self.tolerance)
        self.template_len = data.get("template_len", self.template_len)
        self.n_training_samples = data.get("n_training_samples", 0)
        self.feature_medians = data.get("feature_medians")
        self.feature_lower = data.get("feature_lower")
        self.feature_upper = data.get("feature_upper")
        avg = data.get("avg_template")
        self.avg_template = np.array(avg, dtype=np.float64) if avg is not None else None

    def save_pickle(self, path: str | Path) -> None:
        """Serialize entire model (including raw templates) to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_pickle(self, path: str | Path) -> None:
        """Restore model state from a pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_features(self, all_features: List[Dict[str, float]]) -> None:
        """Compute per-feature medians and tolerance bounds."""
        keys = list(all_features[0].keys())
        medians: Dict[str, float] = {}
        for k in keys:
            vals = [f[k] for f in all_features if k in f]
            medians[k] = float(np.median(vals))
        self.feature_medians = medians
        self._recompute_bounds()

    def _recompute_bounds(self) -> None:
        """Recompute lower/upper from current medians and tolerance."""
        if self.feature_medians is None:
            return
        lower: Dict[str, float] = {}
        upper: Dict[str, float] = {}
        for k, m in self.feature_medians.items():
            span = abs(m) * self.tolerance
            lower[k] = m - span
            upper[k] = m + span
        self.feature_lower = lower
        self.feature_upper = upper

    def _fit_templates(self, templates: List[np.ndarray]) -> None:
        """Resample + average templates."""
        if not templates:
            return
        self._raw_templates = list(templates)
        self._n_templates = len(templates)
        self._rebuild_avg_template()

    def _rebuild_avg_template(self) -> None:
        """Rebuild average template from raw template list."""
        if not self._raw_templates:
            return
        resampled = np.array(
            [self._resample(t, self.template_len) for t in self._raw_templates]
        )
        avg = np.mean(resampled, axis=0)
        # Normalise average template
        norm = np.linalg.norm(avg)
        self.avg_template = avg / norm if norm > 0 else avg

    @staticmethod
    def _normalize(segment: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-L2-norm normalisation."""
        s = segment - np.mean(segment)
        n = np.linalg.norm(s)
        return s / n if n > 0 else s

    @staticmethod
    def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Linearly resample ``arr`` to exactly ``target_len`` points."""
        if len(arr) == target_len:
            return arr.copy()
        src_idx = np.linspace(0, len(arr) - 1, target_len)
        return np.interp(src_idx, np.arange(len(arr)), arr)
