"""Check rendered terminal contents, including wrapping, resize and teardown.

Run independently of hardware and collection dependencies:
uv run --no-project --with pytest --with pyte python -m pytest \
    tests/utils/test_lerobot_progress_terminal.py
"""

import io
import json
import os

import pytest

from utils.lerobot_progress import RolloutProgress

pyte = pytest.importorskip("pyte")


class Terminal(io.StringIO):
    def __init__(self, screen):
        super().__init__()
        self.parser = pyte.Stream(screen)

    def isatty(self):
        return True

    def fileno(self):
        return 123

    def write(self, text):
        self.parser.feed(text)
        return super().write(text)


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    screen = pyte.Screen(57, 8)
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda fd: os.terminal_size((screen.columns, screen.lines)),
    )
    (tmp_path / "session.json").write_text(json.dumps({"rollout_index": 49, "pending": 3, "phase": "recording"}))
    (tmp_path / "worker.json").write_text(json.dumps({"episode": "46", "state": "processing"}))
    stream = Terminal(screen)
    return screen, stream, RolloutProgress(tmp_path, stream=stream)


def test_bottom_status_survives_wrapped_logs_and_progress_output(terminal):
    screen, stream, display = terminal
    display.refresh()
    expected = screen.display[-1]
    expected_event = screen.display[-2]
    assert "#049" in expected
    for index in range(100):
        # Model/saver logs wrap in narrow panes; video progress rewrites lines.
        stream.write(f"\033[32mlog {index}: " + "x" * 160 + "\033[0m\r\n")
        stream.write("encoding 10%\r\033[2Kencoding 90%\r\n")
        assert screen.display[-1] == expected
        assert screen.display[-2] == expected_event
        display.refresh()
        assert screen.display[-1] == expected
    assert any("encoding 90%" in row for row in screen.display[:-1])


def test_done_and_discard_stay_visible_through_idle_refreshes(terminal):
    screen, stream, display = terminal
    (display.directory / "session.json").write_text(
        json.dumps(
            {
                "phase": "idle",
                "last_rollout_index": 49,
                "pending": 0,
                "event": {"at": 2, "message": "episode_049 discarded"},
            }
        )
    )
    (display.directory / "worker.json").write_text(
        json.dumps(
            {
                "episode": "48",
                "state": "finished",
            }
        )
    )
    for _ in range(3):
        display.refresh()
        assert "#048 done" in screen.display[-1]
        assert "#049 idle" in screen.display[-1]
        assert "discarded" in screen.display[-2]
        stream.write("background log\r\n" * 10)


def test_refresh_restores_log_cursor_and_color(terminal):
    screen, stream, display = terminal
    display.refresh()
    stream.write("\033[32mpartial log ")
    before = (screen.cursor.x, screen.cursor.y, screen.cursor.attrs)
    display.refresh()
    assert (screen.cursor.x, screen.cursor.y, screen.cursor.attrs) == before
    stream.write("continued\033[0m\r\n")
    assert any("partial log continued" in row for row in screen.display[:-1])


def test_resize_and_external_margin_reset(terminal):
    screen, stream, display = terminal
    display.refresh()
    for columns, lines in [(120, 20), (40, 6), (57, 8)]:
        screen.resize(lines=lines, columns=columns)
        # Other terminal writers can reset the scrolling region.
        stream.write("\033[r")
        display.refresh()
        assert "#049" in screen.display[-1] and "#046" in screen.display[-1]
        expected = screen.display[-1]
        stream.write("more logs\r\n" * 100)
        assert screen.display[-1] == expected


def test_tiny_terminal_can_recover_and_close_leaves_normal_scrolling(terminal):
    screen, stream, display = terminal
    display.refresh()
    screen.resize(lines=2, columns=57)
    display.refresh()
    screen.resize(lines=8, columns=57)
    display.refresh()
    assert "#049" in screen.display[-1]
    display.close()
    assert screen.display[-1].strip() == ""
    stream.write("shell prompt")
    assert screen.display[-1].strip() == "shell prompt"
    stream.write("\r\n" + "ordinary output\r\n" * 20)
    assert all("#049" not in row for row in screen.display)


def test_terminal_size_failure_recovers_next_refresh(terminal, monkeypatch):
    screen, stream, display = terminal
    original = os.get_terminal_size

    def failing_size(fd):
        raise OSError("resize in progress")

    monkeypatch.setattr(os, "get_terminal_size", failing_size)
    display.refresh()
    assert not stream.getvalue()
    monkeypatch.setattr(os, "get_terminal_size", original)
    display.refresh()
    assert "#049" in screen.display[-1]
