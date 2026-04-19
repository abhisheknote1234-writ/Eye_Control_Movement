"""
annotator.py
------------
Annotator provides interactive matplotlib-based annotation.

Usage flow
----------
1. Call ``Annotator(signal, sample_rate)`` to create the annotator.
2. Call ``annotator.plot_and_annotate()`` which opens an interactive figure.
3. Click on the waveform three times per event:
       click 1 → marks *start*
       click 2 → marks *peak*
       click 3 → marks *end*   → annotation record is saved
4. After closing the window (or calling ``annotator.finish()``), retrieve
   records via ``annotator.get_annotations()``.

Each annotation record is a dict::

    {"start": int, "peak": int, "end": int}

Points are stored as sample indices (rounded to nearest integer x-value).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # use non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

# Colours for the three click roles
_ROLE_COLORS = {"start": "green", "peak": "orange", "end": "red"}
_ROLES = ["start", "peak", "end"]


class Annotator:
    """Interactive waveform annotator using matplotlib mouse clicks.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array to annotate.
    sample_rate : int
        Sampling frequency in Hz (used for x-axis labelling only).
    """

    def __init__(self, signal: np.ndarray, sample_rate: int = 250) -> None:
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for Annotator.")

        self.signal = np.asarray(signal, dtype=np.float64)
        self.sample_rate = sample_rate

        self._annotations: List[Dict[str, int]] = []
        self._current: Dict[str, Optional[int]] = {"start": None, "peak": None, "end": None}
        self._role_index: int = 0  # which click role is next: 0→start, 1→peak, 2→end

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._markers: List[plt.Artist] = []   # keep references to remove later if needed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_and_annotate(self, block: bool = True) -> None:
        """Open the annotation window and (optionally) block until closed.

        Parameters
        ----------
        block : bool
            If True (default), the call blocks until the figure window is
            closed by the user.  Set False for programmatic / test use.
        """
        self._setup_figure()
        if block:
            plt.show(block=True)

    def get_annotations(self) -> List[Dict[str, int]]:
        """Return list of completed annotation records.

        Each record: ``{"start": int, "peak": int, "end": int}``
        """
        return list(self._annotations)

    def clear_annotations(self) -> None:
        """Remove all stored annotations and reset click state."""
        self._annotations.clear()
        self._current = {"start": None, "peak": None, "end": None}
        self._role_index = 0
        if self._ax is not None:
            for m in self._markers:
                try:
                    m.remove()
                except Exception:
                    pass
            self._markers.clear()
            if self._fig is not None:
                self._fig.canvas.draw_idle()

    def add_annotation_programmatic(
        self,
        start: int,
        peak: int,
        end: int,
    ) -> bool:
        """Add an annotation without mouse interaction (useful for testing).

        Returns True if the annotation was valid and stored, False otherwise.
        """
        return self._try_store_annotation(start, peak, end)

    def finish(self) -> None:
        """Close the matplotlib figure if it is open."""
        if self._fig is not None:
            plt.close(self._fig)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_figure(self) -> None:
        """Create the matplotlib figure and wire up the click callback."""
        self._fig, self._ax = plt.subplots(figsize=(14, 4))
        t = np.arange(len(self.signal)) / self.sample_rate
        self._ax.plot(t, self.signal, color="steelblue", linewidth=0.8, label="signal")
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Amplitude")
        self._ax.set_title(
            "Click to annotate — role order: START (green) → PEAK (orange) → END (red)\n"
            "Close window when done."
        )

        # Legend for roles
        legend_patches = [
            Line2D([0], [0], marker="v", color="w", markerfacecolor=c, markersize=10, label=r)
            for r, c in _ROLE_COLORS.items()
        ]
        self._ax.legend(handles=legend_patches, loc="upper right")

        self._cid = self._fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event) -> None:
        """Handle a mouse click on the axes."""
        if event.inaxes is not self._ax:
            return
        if event.xdata is None:
            return

        # Convert x (time in seconds) → nearest sample index
        idx = int(round(event.xdata * self.sample_rate))
        idx = max(0, min(idx, len(self.signal) - 1))

        role = _ROLES[self._role_index]
        self._current[role] = idx

        # Draw a marker
        color = _ROLE_COLORS[role]
        marker = self._ax.axvline(
            x=event.xdata, color=color, linestyle="--", linewidth=1.0, alpha=0.7
        )
        dot = self._ax.plot(
            event.xdata,
            self.signal[idx],
            marker="v",
            color=color,
            markersize=9,
            zorder=5,
        )[0]
        self._markers.extend([marker, dot])
        self._fig.canvas.draw_idle()

        self._role_index += 1

        if self._role_index == 3:
            # All three roles collected — try to store
            s = self._current["start"]
            p = self._current["peak"]
            e = self._current["end"]
            ok = self._try_store_annotation(s, p, e)
            if not ok:
                # Invalid order — remove the last 3 markers and reset
                for _ in range(3):
                    if self._markers:
                        try:
                            self._markers.pop().remove()
                        except Exception:
                            pass
                print(
                    f"[Annotator] Invalid annotation order (start={s}, peak={p}, end={e}). "
                    "Please redo this annotation: start must come before peak and end."
                )
            else:
                n = len(self._annotations)
                print(f"[Annotator] Annotation #{n} saved: start={s}, peak={p}, end={e}")

            # Reset for next annotation
            self._current = {"start": None, "peak": None, "end": None}
            self._role_index = 0
            if self._fig is not None:
                self._fig.canvas.draw_idle()

    def _try_store_annotation(
        self,
        start: Optional[int],
        peak: Optional[int],
        end: Optional[int],
    ) -> bool:
        """Validate and store one annotation record.

        Validation rules
        ----------------
        * All three indices must be integers.
        * ``start <= peak <= end``
        * ``start < end`` (non-zero length segment)

        Returns True on success, False if validation fails.
        """
        if start is None or peak is None or end is None:
            return False
        try:
            start, peak, end = int(start), int(peak), int(end)
        except (TypeError, ValueError):
            return False

        n = len(self.signal)
        # Clamp to valid range
        start = max(0, min(start, n - 1))
        peak = max(0, min(peak, n - 1))
        end = max(0, min(end, n - 1))

        if not (start <= peak <= end and start < end):
            return False

        self._annotations.append({"start": start, "peak": peak, "end": end})
        return True
