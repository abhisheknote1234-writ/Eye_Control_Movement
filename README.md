# Eye Control Movement — Signal Pattern Detection System

A modular, semi-supervised Python system for detecting and classifying signal-based patterns (eye blinks, EMG spikes, etc.) using only lightweight dependencies: **NumPy**, **SciPy**, and **Matplotlib**.

---

## Features

| # | Feature | Notes |
|---|---------|-------|
| 1 | Signal acquisition | Simulated Gaussian-spike signals or file-based (`.npy`/`.csv`) |
| 2 | Configurable sample rate | Default 250 Hz; rolling buffer |
| 3 | Interactive annotation | Matplotlib click-based: START → PEAK → END per event |
| 4 | Feature extraction | Slope up/down, amplitude, gap, energy |
| 5 | Semi-supervised model | Median-based learning, ±30 % tolerance bands |
| 6 | Template learning | Normalised average waveform template |
| 7 | Real-time detection | Slope + template-similarity gating |
| 8 | Command mapping | `BLINK → LEFT`, `DOUBLE_BLINK → RIGHT` (customisable) |
| 9 | Adaptive recalibration | Incremental model update with new detections |
| 10 | Save / load model | JSON (human-readable) or pickle (full fidelity) |

---

## Project layout

```
Eye_Control_Movement/
├── eye_control/                  # Core package
│   ├── __init__.py
│   ├── signal_processor.py       # SignalProcessor
│   ├── annotator.py              # Annotator
│   ├── feature_extractor.py      # FeatureExtractor
│   ├── pattern_model.py          # PatternModel
│   ├── detector.py               # Detector
│   └── command_mapper.py         # CommandMapper
├── tests/                        # pytest test suite (51 tests)
│   ├── test_signal_processor.py
│   ├── test_annotator.py
│   ├── test_feature_extractor.py
│   ├── test_pattern_model.py
│   ├── test_detector.py
│   └── test_command_mapper.py
├── demo.py                       # End-to-end runnable demo
└── README.md
```

---

## Requirements

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Quick start

### 1 — Headless demo (no display needed)

```bash
python demo.py --no-gui --no-plot
```

This generates a synthetic signal, auto-annotates it, trains the model, runs streaming detection, and prints detected commands.

### 2 — Interactive demo (with GUI annotation)

```bash
python demo.py
```

An annotation window opens:

1. **Click 1** on a spike → marks **START** (green line)
2. **Click 2** on a spike → marks **PEAK** (orange line)
3. **Click 3** on a spike → marks **END** (red line)
4. Repeat for as many blinks as you like, then **close the window**.

The system trains on your annotations, runs streaming detection over the full signal, and prints mapped commands (`BLINK → LEFT`, `DOUBLE_BLINK → RIGHT`).

A second plot then overlays the detections on the signal.

### 3 — Load your own signal

```bash
# NumPy binary
python demo.py --file my_signal.npy

# CSV (single column)
python demo.py --file my_signal.csv
```

### 4 — Save / reload the trained model

```bash
# Save model after training
python demo.py --save-model model.json
```

### 5 — Full option reference

```
python demo.py --help

options:
  --file FILE              Path to a signal file (.npy/.csv/.txt)
  --duration DURATION      Simulated signal duration in seconds (default: 10.0)
  --sample-rate RATE       Sampling rate in Hz (default: 250)
  --n-spikes N             Number of spikes in simulated signal (default: 6)
  --tolerance TOL          Feature tolerance 0–1 (default: 0.30)
  --similarity-thresh T    Template similarity threshold (default: 0.70)
  --save-model FILE        Save trained model to JSON file
  --no-gui                 Skip interactive annotation window
  --no-plot                Skip detection overlay plot
```

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 51 tests should pass.

---

## Architecture overview

```
SignalProcessor  ──►  rolling buffer, simulated/file signal, streaming helper
      │
      ▼
  Annotator       ──►  matplotlib click → {start, peak, end} records
      │
      ▼
FeatureExtractor  ──►  slope_up, slope_down, amplitude, gap, energy
      │
      ▼
 PatternModel     ──►  median-based learning, tolerance bands, avg template,
      │                save/load JSON/pickle, adaptive recalibration
      ▼
   Detector       ──►  slope-based candidate scan + peak-centred extraction
      │                + feature match + template similarity gating
      ▼
CommandMapper     ──►  BLINK → LEFT | DOUBLE_BLINK → RIGHT (customisable)
```

---

## License

MIT
