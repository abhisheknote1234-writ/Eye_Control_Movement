#!/usr/bin/env python3
"""
demo.py
-------
End-to-end demonstration of the eye_control signal-pattern detection system.

What this script does
---------------------
1. Generates (or loads) a synthetic signal with embedded blink-like spikes.
2. Opens a matplotlib window so you can annotate example blinks by clicking.
3. Trains the PatternModel from your annotations.
4. Runs streaming detection over the whole signal.
5. Prints every detected event and its mapped command (BLINK → LEFT, etc.).

Quick start
-----------
    python demo.py                    # uses simulated signal
    python demo.py --file signal.npy  # loads a NumPy .npy file
    python demo.py --no-gui           # skip annotation, use pre-computed annotations

Controls during annotation
--------------------------
  • Click 1 on a spike  → marks START  (green dashed line)
  • Click 2 on a spike  → marks PEAK   (orange dashed line)
  • Click 3 on a spike  → marks END    (red dashed line)
  Repeat for as many blinks as you want, then close the window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Optional matplotlib — imported lazily so --no-gui works without a display
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("TkAgg")          # prefer interactive TkAgg for the demo
except Exception:
    try:
        import matplotlib
        matplotlib.use("Qt5Agg")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Local package
# ---------------------------------------------------------------------------
from eye_control import (
    SignalProcessor,
    Annotator,
    FeatureExtractor,
    PatternModel,
    Detector,
    CommandMapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_demo_annotations(signal: np.ndarray, sample_rate: int) -> List[Dict[str, int]]:
    """Return a handful of hard-coded annotations for the simulated signal
    when --no-gui is requested.  Works only with the fixed seed=42 signal.
    """
    # Use scipy to find the actual peaks in the simulated signal
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, height=0.5, distance=int(0.2 * sample_rate))
    except ImportError:
        # Fallback: evenly spaced guesses
        peaks = np.linspace(50, len(signal) - 50, 4, dtype=int)

    annotations = []
    half_width = int(0.06 * sample_rate)  # ~60 ms each side
    for p in peaks[:4]:                   # use at most 4
        start = max(0, p - half_width)
        end = min(len(signal) - 1, p + half_width)
        annotations.append({"start": int(start), "peak": int(p), "end": int(end)})
    return annotations


def _print_separator(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    sample_rate = args.sample_rate

    # -----------------------------------------------------------------------
    # 1. Signal acquisition
    # -----------------------------------------------------------------------
    _print_separator("SIGNAL ACQUISITION")
    sp = SignalProcessor(sample_rate=sample_rate)

    if args.file:
        print(f"Loading signal from {args.file} …")
        signal = SignalProcessor.load_from_file(args.file)
        print(f"  Loaded {len(signal)} samples.")
    else:
        print(f"Generating simulated signal ({args.duration} s @ {sample_rate} Hz) …")
        signal = SignalProcessor.simulate_signal(
            duration=args.duration,
            sample_rate=sample_rate,
            n_spikes=args.n_spikes,
            seed=42,
        )
        print(f"  Generated {len(signal)} samples.")

    sp.push_many(signal)

    # -----------------------------------------------------------------------
    # 2. Annotation
    # -----------------------------------------------------------------------
    _print_separator("ANNOTATION")

    if args.no_gui:
        print("--no-gui: using auto-generated annotations.")
        annotations = _make_demo_annotations(signal, sample_rate)
        print(f"  {len(annotations)} annotations created automatically.")
        for i, a in enumerate(annotations):
            print(f"  #{i}: start={a['start']}, peak={a['peak']}, end={a['end']}")
    else:
        print(
            "Annotation window opening …\n"
            "  Click START (green) → PEAK (orange) → END (red) for each blink.\n"
            "  Repeat for multiple blinks, then close the window."
        )
        annotator = Annotator(signal, sample_rate=sample_rate)
        annotator.plot_and_annotate(block=True)
        annotations = annotator.get_annotations()

        if not annotations:
            print(
                "\n[WARNING] No annotations were made.\n"
                "  Falling back to auto-generated annotations for the demo."
            )
            annotations = _make_demo_annotations(signal, sample_rate)
        else:
            print(f"\n  {len(annotations)} annotation(s) collected:")
            for i, a in enumerate(annotations):
                print(f"    #{i}: start={a['start']}, peak={a['peak']}, end={a['end']}")

    if not annotations:
        print("[ERROR] Could not obtain any annotations. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Feature extraction (show what was learned per segment)
    # -----------------------------------------------------------------------
    _print_separator("FEATURE EXTRACTION")
    fe = FeatureExtractor(sample_rate=sample_rate, compute_energy=True)

    print("Per-annotation features:")
    for i, ann in enumerate(annotations):
        feats = fe.extract(signal, ann)
        if feats:
            print(
                f"  #{i}: slope_up={feats['slope_up']:.4f}  "
                f"slope_down={feats['slope_down']:.4f}  "
                f"amplitude={feats['amplitude']:.4f}  "
                f"gap={feats['gap']:.0f} samples  "
                f"energy={feats.get('energy', float('nan')):.4f}"
            )
        else:
            print(f"  #{i}: segment too short — skipped.")

    # -----------------------------------------------------------------------
    # 4. Model training
    # -----------------------------------------------------------------------
    _print_separator("MODEL TRAINING")
    model = PatternModel(
        tolerance=args.tolerance,
        template_len=50,
        feature_extractor=fe,
    )

    ok = model.train(signal, annotations)
    if not ok:
        print("[ERROR] Model training failed — no valid segments. Exiting.")
        sys.exit(1)

    print(f"  Training succeeded on {model.n_training_samples} segment(s).")
    print(f"  Tolerance: ±{model.tolerance * 100:.0f}%")
    print("  Learned feature medians:")
    for k, v in (model.feature_medians or {}).items():
        lo = (model.feature_lower or {}).get(k, float("nan"))
        hi = (model.feature_upper or {}).get(k, float("nan"))
        print(f"    {k:>12}: {v:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(
        f"  Average template shape: {model.avg_template.shape if model.avg_template is not None else 'None'}"
    )

    # Optionally save model
    if args.save_model:
        model.save_json(args.save_model)
        print(f"  Model saved → {args.save_model}")

    # -----------------------------------------------------------------------
    # 5. Real-time streaming detection + command mapping
    # -----------------------------------------------------------------------
    _print_separator("STREAMING DETECTION")

    detector = Detector(
        model=model,
        sample_rate=sample_rate,
        similarity_thresh=args.similarity_thresh,
        double_blink_gap_s=0.50,
    )
    mapper = CommandMapper()   # BLINK→LEFT, DOUBLE_BLINK→RIGHT

    all_detections = []

    def on_detection(det) -> None:
        all_detections.append(det)
        action = mapper.emit(det.label)
        print(
            f"  ▶ {det.label:14s}  peak@{det.peak:5d}  "
            f"sim={det.similarity:.3f}  →  {action}"
        )

    print("Scanning signal …")
    detector.stream_detect(
        signal,
        window_size=min(500, len(signal) // 2),
        step=min(250, len(signal) // 4),
        callback=on_detection,
    )

    _print_separator()
    print(f"Total detections: {len(all_detections)}")
    blinks = sum(1 for d in all_detections if d.label == "BLINK")
    double_blinks = sum(1 for d in all_detections if d.label == "DOUBLE_BLINK")
    print(f"  BLINK:        {blinks}")
    print(f"  DOUBLE_BLINK: {double_blinks}")

    # -----------------------------------------------------------------------
    # 6. Optional: plot detections overlay
    # -----------------------------------------------------------------------
    if not args.no_gui and not args.no_plot:
        _plot_detections(signal, sample_rate, annotations, all_detections)

    _print_separator("DONE")
    print("Demo complete.")


def _plot_detections(signal, sample_rate, annotations, detections) -> None:
    """Show signal with annotation marks and detected events."""
    try:
        import matplotlib.pyplot as plt

        t = np.arange(len(signal)) / sample_rate
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, signal, color="steelblue", linewidth=0.8, label="signal", zorder=1)

        # Annotation ground truth
        for ann in annotations:
            for role, color, marker in [
                ("start", "green", "v"),
                ("peak", "orange", "^"),
                ("end", "red", "v"),
            ]:
                idx = ann[role]
                ax.plot(
                    t[idx], signal[idx], marker=marker, color=color,
                    markersize=8, zorder=3,
                )

        # Detections
        for det in detections:
            color = "magenta" if det.label == "BLINK" else "cyan"
            ax.axvline(x=t[det.peak], color=color, linewidth=1.2, alpha=0.6, zorder=2)
            ax.annotate(
                det.label,
                xy=(t[det.peak], signal[det.peak]),
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=7,
                color=color,
                ha="center",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Signal with annotations (▼start/end, △peak) and detections (|lines)")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"[WARNING] Could not display detections plot: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eye-control signal-pattern detection demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", type=str, default=None, help="Path to a signal file (.npy/.csv/.txt).")
    p.add_argument("--duration", type=float, default=10.0, help="Simulated signal duration (seconds).")
    p.add_argument("--sample-rate", type=int, default=250, help="Sampling rate in Hz.")
    p.add_argument("--n-spikes", type=int, default=6, help="Number of spikes in simulated signal.")
    p.add_argument("--tolerance", type=float, default=0.30, help="Feature tolerance (0–1).")
    p.add_argument("--similarity-thresh", type=float, default=0.70, help="Template similarity threshold.")
    p.add_argument("--save-model", type=str, default=None, help="Save trained model to JSON file.")
    p.add_argument("--load-model", type=str, default=None, help="Load model from JSON file.")
    p.add_argument("--no-gui", action="store_true", help="Skip interactive annotation window.")
    p.add_argument("--no-plot", action="store_true", help="Skip detection overlay plot.")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_demo(args)
