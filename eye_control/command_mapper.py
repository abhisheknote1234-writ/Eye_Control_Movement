"""
command_mapper.py
-----------------
CommandMapper maps detected pattern labels to user-defined actions and
optionally triggers a callback.

Default mappings
----------------
    BLINK        → LEFT
    DOUBLE_BLINK → RIGHT
"""

from __future__ import annotations

from typing import Callable, Dict, Optional


class CommandMapper:
    """Map detection labels to action strings and emit commands.

    Parameters
    ----------
    mapping : dict or None
        ``{label: action}`` mapping.  If None, the default mapping is used.
    on_command : callable or None
        Optional callback invoked as ``on_command(label, action)`` whenever
        a command is emitted.  The default behaviour is to print to stdout.
    """

    _DEFAULT_MAPPING: Dict[str, str] = {
        "BLINK": "LEFT",
        "DOUBLE_BLINK": "RIGHT",
    }

    def __init__(
        self,
        mapping: Optional[Dict[str, str]] = None,
        on_command: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.mapping: Dict[str, str] = dict(
            mapping if mapping is not None else self._DEFAULT_MAPPING
        )
        self._on_command = on_command or self._default_print

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(self, label: str) -> Optional[str]:
        """Look up ``label`` and trigger the mapped action.

        Parameters
        ----------
        label : str
            Detection label (e.g. ``"BLINK"``).

        Returns
        -------
        str or None
            The mapped action string, or None if the label is not mapped.
        """
        action = self.mapping.get(label)
        if action is not None:
            self._on_command(label, action)
        return action

    def add_mapping(self, label: str, action: str) -> None:
        """Register or update a label → action mapping."""
        self.mapping[label] = action

    def remove_mapping(self, label: str) -> bool:
        """Remove a mapping.  Returns True if it existed."""
        if label in self.mapping:
            del self.mapping[label]
            return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_print(label: str, action: str) -> None:
        print(f"[CommandMapper] {label} → {action}")
