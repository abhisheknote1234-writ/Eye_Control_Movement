"""
eye_control — semi-supervised signal pattern detection system.

Modules
-------
signal_processor  : SignalProcessor  — buffer, simulated/file input
annotator         : Annotator        — matplotlib click annotation
feature_extractor : FeatureExtractor — per-segment feature computation
pattern_model     : PatternModel     — median-based model + template store
detector          : Detector         — slope-based real-time detection
command_mapper    : CommandMapper    — label → action mapping
"""

from .signal_processor import SignalProcessor
from .annotator import Annotator
from .feature_extractor import FeatureExtractor
from .pattern_model import PatternModel
from .detector import Detector
from .command_mapper import CommandMapper

__all__ = [
    "SignalProcessor",
    "Annotator",
    "FeatureExtractor",
    "PatternModel",
    "Detector",
    "CommandMapper",
]
