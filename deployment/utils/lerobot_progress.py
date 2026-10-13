"""Per-invocation recorder progress and a reserved terminal status row."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path


def read_status(path):
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def run_rollout_history(directory, run_name):
    """Index retained rollouts by capture time, deduplicating cache/ready copies.

    Each raw rollout counts once, regardless of its action/value output count.
    Scan capture before ready so a concurrent conversion's atomic promotion
    cannot make its episode disappear from both sides of the scan.
    """
    episodes = {}
    numbered_indices = {}
    for stage in ("capture", "building", "ready"):
        for path in (Path(directory) / stage).glob("*/manifest.json"):
            manifest = read_status(path)
            source_run = manifest.get("run_name") or manifest.get("config", {}).get("run_name")
            if source_run != run_name or manifest.get("result") not in {
                "success",
                "failure",
            }:
                continue
            if stage == "capture" and not manifest.get("normal_end"):
                continue
            uid = manifest.get("episode_uid")
            start = manifest.get("start_ns")
            if isinstance(uid, str) and type(start) is int:
                episodes[uid] = start
                index = manifest.get("rollout_index")
                if manifest.get("episode_name") and type(index) is int and index >= 0:
                    numbered_indices[uid] = index
    return {
        uid: numbered_indices.get(uid, index)
        for index, uid in enumerate(sorted(episodes, key=lambda uid: (episodes[uid], uid)))
    }


def episode_label(manifest, rollout_index=None):
    index = manifest.get("rollout_index") if rollout_index is None else rollout_index
    label = str(index) if index is not None else manifest.get("episode_uid", "unknown")[:8]
    return label


def progress_event(message, level="success"):
    """Emit one lifecycle log and retain the same event for the status panel."""
    from utils.logger_config import logger

    color, symbol = {
        "success": ("green", "✓"),
        "warning": ("yellow", "–"),
        "error": ("red", "✗"),
    }[level]
    logger.opt(colors=True).log(
        {"success": "INFO", "warning": "WARNING", "error": "ERROR"}[level],
        f"<{color}>{symbol} {{}}</{color}>",
        message,
    )
    return {"at": time.time_ns(), "message": f"{symbol} {message}", "level": level}


def elapsed(start, end=None):
    seconds = max(0, int((time.time() if end is None else end) - start))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def status_line(state, worker, *, stopping=False, width=None, controller=None, timers=True):
    active = state.get("rollout_index")
    phase = state.get("phase", "starting").replace(" ", "_")
    if phase == "recording" and active is None:
        phase = "idle"
    if active is None and phase == "idle" and controller:
        phase = controller.get("phase", phase)
    if stopping or phase in {"draining", "finished", "failed"}:
        active = None
    index = active if active is not None else state.get("last_rollout_index")
    rollout = f"{int(index):03d}" if index is not None else "--"
    index = worker.get("episode")
    converted = f"{int(index):03d}" if str(index).isdigit() else (index or "--")
    working = worker.get("state") == "processing"
    pending_worker = working and ("pending_uids" not in state or worker.get("episode_uid") in state["pending_uids"])
    queued = max(0, state.get("pending", 0) - int(pending_worker))
    conversion = {
        "processing": worker.get("stage", "processing"),
        "finished": "done",
        "failed": "FAILED",
    }.get(worker.get("state"), "idle")
    if stopping and phase not in {"finished", "failed"}:
        phase = "draining"
    record_time = (
        f" {elapsed(state['started_at'])}" if timers and active is not None and state.get("started_at") else ""
    )
    convert_time = (
        f" {elapsed(worker['started_at'], worker.get('ended_at'))}" if timers and worker.get("started_at") else ""
    )
    full = (
        f"rollout #{rollout} {phase}{record_time} | lerobot #{converted} {conversion}{convert_time} | queued: {queued}"
    )
    if width is None or len(full) <= width:
        return full
    conversion = "busy" if working else conversion
    compact = f"rollout #{rollout} {phase} | lerobot #{converted} {conversion} | Q:{queued}"
    if len(compact) <= width:
        return compact
    return f"REC #{rollout} {phase} | CONV #{converted} {conversion} Q:{queued}"[:width]


def event_line(state, worker):
    events = [state.get("event", {}), worker.get("event", {})]
    if state.get("errors"):
        return "✗ Conversion error; source retained | " + " ".join(str(state["errors"][-1]).split())
    event = max(events, key=lambda e: e.get("at", 0))
    message = event.get("message", "Ready")
    return message + " | S success / F failure / D discard"


class RolloutProgress:
    """Reserve two bottom rows for events and progress, with logs above them.

    Each update is a single write with cursor save/restore. Redirected output gets
    plain status lines only on changes, never terminal escape sequences.
    """

    def __init__(self, directory, *, stream=None):
        self.directory = Path(directory)
        self.stream = stream if stream is not None else sys.stderr
        self.tty = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.stopping = False
        self._stop = threading.Event()
        self._thread = None
        self._size = None
        self._last = None
        self.controller = {}
        self._state = {}
        self._worker = {}

    def set_phase(self, phase):
        # The inference owner controls this field, independently of saver status.
        self.controller = {"phase": phase}

    def start(self):
        self.refresh()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rollout-status")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(0.2):
            try:
                self.refresh()
            except (OSError, ValueError):
                # A transient terminal failure must not permanently stop updates.
                continue

    def refresh(self):
        # Keep the last valid snapshot through transient file/read failures.
        self._state = read_status(self.directory / "session.json") or self._state
        self._worker = read_status(self.directory / "worker.json") or self._worker
        state, worker = self._state, self._worker
        text = status_line(
            state,
            worker,
            stopping=self.stopping,
            controller=self.controller,
            timers=self.tty,
        )
        event = event_line(state, worker)
        if not self.tty:
            snapshot = (text, event)
            if snapshot != self._last:
                self.stream.write(event + "\n" + text + "\n")
                self.stream.flush()
            self._last = snapshot
            return
        try:
            size = os.get_terminal_size(self.stream.fileno())
        except OSError:
            return
        if size.lines < 3 or size.columns < 2:
            if self._size is not None:
                self.stream.write("\033[r")
                self.stream.flush()
                self._size = None
            return
        setup = ""
        reserved = 2 if size.lines >= 4 else 1
        bottom = size.lines - reserved
        if size != self._size:
            setup = f"\033[1;{bottom}r\033[{bottom};1H\n"
            self._size = size
        text = status_line(
            state,
            worker,
            stopping=self.stopping,
            width=size.columns - 1,
            controller=self.controller,
        )
        event_color = (
            "31" if state.get("errors") or worker.get("state") == "failed" else "32" if event.startswith("✓") else "33"
        )
        event_row = (
            (f"\033[{size.lines - 1};1H\033[0m\033[2K\033[{event_color}m" + event[: size.columns - 1])
            if reserved == 2
            else ""
        )
        # Reassert margins in case another terminal writer reset them. Save before
        # setting margins (DECSTBM homes the cursor), then restore logs' position
        # and colors. Emit the whole frame in one write, including resize setup.
        self.stream.write(
            setup
            + f"\0337\033[1;{bottom}r"
            + event_row
            + f"\033[{size.lines};1H\033[0m\033[2K\033[7m"
            + text
            + "\033[0m\0338"
        )
        self.stream.flush()
        self._last = text

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        try:
            self.refresh()
        except (OSError, ValueError):
            pass
        if self.tty and self._size is not None:
            try:
                # Leave the final status in scrollback and a clean line for logs
                # and the shell prompt, with normal full-screen scrolling restored.
                self.stream.write(f"\033[r\033[{self._size.lines};1H\n")
                self.stream.flush()
            except (OSError, ValueError):
                pass
