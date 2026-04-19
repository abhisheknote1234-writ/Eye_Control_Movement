"""
tests/test_command_mapper.py
-----------------------------
Unit tests for CommandMapper.
"""
import pytest

from eye_control.command_mapper import CommandMapper


def test_default_mapping_blink():
    mapper = CommandMapper()
    result = mapper.emit("BLINK")
    assert result == "LEFT"


def test_default_mapping_double_blink():
    mapper = CommandMapper()
    result = mapper.emit("DOUBLE_BLINK")
    assert result == "RIGHT"


def test_unknown_label_returns_none():
    mapper = CommandMapper()
    result = mapper.emit("UNKNOWN_PATTERN")
    assert result is None


def test_custom_mapping():
    mapper = CommandMapper(mapping={"WINK": "UP", "SQUINT": "DOWN"})
    assert mapper.emit("WINK") == "UP"
    assert mapper.emit("SQUINT") == "DOWN"
    assert mapper.emit("BLINK") is None   # not in custom map


def test_add_mapping():
    mapper = CommandMapper()
    mapper.add_mapping("TRIPLE_BLINK", "SELECT")
    assert mapper.emit("TRIPLE_BLINK") == "SELECT"


def test_remove_mapping():
    mapper = CommandMapper()
    removed = mapper.remove_mapping("BLINK")
    assert removed is True
    assert mapper.emit("BLINK") is None


def test_remove_nonexistent_mapping():
    mapper = CommandMapper()
    removed = mapper.remove_mapping("NONEXISTENT")
    assert removed is False


def test_callback_is_called():
    received = []

    def cb(label, action):
        received.append((label, action))

    mapper = CommandMapper(on_command=cb)
    mapper.emit("BLINK")
    assert received == [("BLINK", "LEFT")]


def test_callback_not_called_for_unknown():
    received = []

    def cb(label, action):
        received.append((label, action))

    mapper = CommandMapper(on_command=cb)
    mapper.emit("MYSTERY")
    assert received == []
