"""Pinned terminal row, non-TTY output and cross-generation rollout numbering."""

import io
import os
from unittest.mock import Mock

from utils.lerobot_online.protocol import atomic_json
from utils.lerobot_progress import (
    RolloutProgress,
    episode_label,
    run_rollout_history,
    status_line,
    event_line,
)


def test_status_line_and_history_override():
    state = {"rollout_index": 49, "pending": 3, "phase": "recording"}
    worker = {"episode": "46", "state": "processing"}
    text = status_line(state, worker)
    assert "rollout #049 recording" in text
    assert "lerobot #046 processing" in text
    assert "queued: 2" in text
    assert "rollout #--" in status_line(state, worker, stopping=True)
    assert episode_label({"rollout_index": 40, "collection_session_id": "old"}) == "40"
    assert episode_label({"rollout_index": 0}, rollout_index=5) == "5"


def test_run_history_counts_unique_retained_rollouts_in_time_order(tmp_path):
    """Old sessions can repeat index 0; value prefixes are not extra rollouts."""

    def record(stage, uid, start, **changes):
        manifest = {
            "episode_uid": uid,
            "start_ns": start,
            "rollout_index": 0,
            "result": "success",
            "run_name": "run-a",
        }
        manifest.update(changes)
        atomic_json(tmp_path / stage / uid / "manifest.json", manifest)

    record("ready", "first", 10)
    record("ready", "second", 20, result="failure", outputs={"value": [1, 2, 3]})
    record("capture", "second", 20, normal_end=True)
    record(
        "capture",
        "queued",
        30,
        normal_end=True,
        run_name=None,
        config={"run_name": "run-a"},
    )
    record("building", "queued", 30)
    record("capture", "unfinished", 40, normal_end=False)
    record("capture", "discarded", 45, normal_end=True, result="discard")
    record("ready", "foreign", 50, run_name="run-b")
    assert run_rollout_history(tmp_path, "run-a") == {
        "first": 0,
        "second": 1,
        "queued": 2,
    }
    assert run_rollout_history(tmp_path, "new-run") == {}


def test_deleted_last_episode_does_not_reserve_a_history_number(tmp_path):
    import shutil

    for index in range(3):
        atomic_json(
            tmp_path / "ready" / str(index) / "manifest.json",
            {
                "episode_uid": str(index),
                "start_ns": index,
                "run_name": "run",
                "result": "success",
            },
        )
    shutil.rmtree(tmp_path / "ready" / "2")
    assert len(run_rollout_history(tmp_path, "run")) == 2


def test_redirected_output_has_no_escape_codes_and_only_changes(tmp_path):
    stream = io.StringIO()
    display = RolloutProgress(tmp_path, stream=stream)
    atomic_json(tmp_path / "session.json", {"rollout_index": 0})
    display.refresh()
    display.refresh()
    assert len(stream.getvalue().splitlines()) == 2
    assert "\033" not in stream.getvalue()
    atomic_json(tmp_path / "session.json", {"rollout_index": 40})
    display.refresh()
    assert "rollout #040" in stream.getvalue()
    display.close()


def test_terminal_reserves_bottom_row_and_restores_scroll_region(tmp_path, monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

        def fileno(self):
            return 123

    monkeypatch.setenv("TERM", "xterm-256color")
    size = os.terminal_size((120, 30))
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: size)
    stream = Terminal()
    display = RolloutProgress(tmp_path, stream=stream)
    display.refresh()
    assert "\033[1;28r" in stream.getvalue()
    assert "\033[30;1H\033[0m\033[2K" in stream.getvalue()
    assert stream.getvalue().endswith("\0338")
    size = os.terminal_size((80, 20))
    display.refresh()
    assert "\033[1;18r" in stream.getvalue()
    display.close()
    assert stream.getvalue().endswith("\033[r\033[20;1H\n")


def test_narrow_terminal_keeps_both_episode_numbers():
    state = {"rollout_index": 49, "pending": 3, "phase": "recording"}
    worker = {"episode": "46", "state": "processing"}
    text = status_line(state, worker, width=56)
    assert "#049" in text and "#046" in text
    assert "recording" in text and "busy" in text and "Q:2" in text


def test_transient_terminal_error_does_not_stop_refresh_thread(tmp_path):
    display = RolloutProgress(tmp_path, stream=io.StringIO())
    display._stop = Mock(wait=Mock(side_effect=[False, False, True]))
    display.refresh = Mock(side_effect=[OSError("terminal resizing"), None])
    display._run()
    assert display.refresh.call_count == 2


def test_non_object_status_is_treated_as_unavailable(tmp_path):
    (tmp_path / "session.json").write_text("null")
    (tmp_path / "worker.json").write_text("[]")
    stream = io.StringIO()
    RolloutProgress(tmp_path, stream=stream).refresh()
    assert "rollout #--" in stream.getvalue()


def test_completed_numbers_remain_and_finished_worker_is_not_subtracted():
    state = {"phase": "idle", "last_rollout_index": 40, "pending": 1}
    worker = {"episode": "39", "state": "finished", "started_at": 100, "ended_at": 248}
    text = status_line(state, worker)
    assert "rollout #040 idle" in text
    assert "lerobot #039 done 02:28" in text
    assert "queued: 1" in text
    assert "last done" not in text.lower()
    assert "#039 done" in status_line(state, worker, stopping=True)


def test_failures_and_queue_wait_are_visible_without_hiding_episode():
    state = {
        "phase": "wait conversion",
        "last_rollout_index": 40,
        "pending": 2,
        "pending_uids": ["new1", "new2"],
    }
    worker = {"episode": "39", "episode_uid": "old", "state": "processing"}
    assert "queued: 2" in status_line(state, worker)  # Worker snapshot is stale.
    worker.update(state="failed")
    text = status_line(state, worker)
    assert "#039 FAILED" in text and "wait_conversion" in text
    assert "homing" not in status_line(state, worker, controller={"phase": "homing"})
    state.update(phase="idle")
    assert "wait_policy" in status_line(state, worker, controller={"phase": "wait_policy"})


def test_newest_event_is_retained_and_errors_survive_later_completions():
    state = {"event": {"at": 2, "message": "episode_040 discarded"}}
    worker = {"event": {"at": 1, "message": "convert success #039"}}
    assert "discarded" in event_line(state, worker)
    assert "S success / F failure / D discard" in event_line(state, worker)
    worker["event"]["at"] = 3
    assert "convert success" in event_line(state, worker)
    state["errors"] = ["episode_038 failed"]
    assert "episode_038 failed" in event_line(state, worker)


def test_redirected_output_does_not_repeat_for_elapsed_time(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "session.json",
        {
            "rollout_index": 1,
            "phase": "recording",
            "started_at": 100,
        },
    )
    stream = io.StringIO()
    display = RolloutProgress(tmp_path, stream=stream)
    monkeypatch.setattr("utils.lerobot_progress.time.time", lambda: 105)
    display.refresh()
    initial = stream.getvalue()
    monkeypatch.setattr("utils.lerobot_progress.time.time", lambda: 150)
    display.refresh()
    assert stream.getvalue() == initial


def test_transient_status_read_failure_keeps_last_valid_numbers(tmp_path):
    atomic_json(tmp_path / "session.json", {"phase": "idle", "last_rollout_index": 9})
    atomic_json(tmp_path / "worker.json", {"episode": "8", "state": "finished"})
    stream = io.StringIO()
    display = RolloutProgress(tmp_path, stream=stream)
    display.refresh()
    initial = stream.getvalue()
    (tmp_path / "worker.json").write_text("{")
    (tmp_path / "session.json").unlink()
    display.refresh()
    assert stream.getvalue() == initial
    assert "#008 done" in initial and "#009 idle" in initial
