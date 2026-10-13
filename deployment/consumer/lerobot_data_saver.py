#!/usr/bin/env python3
"""Collect SHM streams into paired offline LeRobot datasets; no models or recorded data."""

from __future__ import annotations

import select
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from consumer.base import BaseConsumer
from utils.collection_signals import CollectionSignals
from utils.lerobot_recording import RecordingSession, check_runtime, validate_config
from utils.logger_config import logger


def resolved_consumer_config(cfg):
    # Resolve only the sections we use. Other groups contain hydra:runtime fields.
    return validate_config(OmegaConf.to_container(cfg.consumer, resolve=True))


class LeRobotCollectConsumer(CollectionSignals, BaseConsumer):
    def __init__(self, config, signal_files, halt_and_home, *, session=None):
        self.summary_shm = None
        self.config = validate_config(config)
        required = (
            "episode_signal_pattern",
            "robot_home_command_pattern",
            "robot_teleop_start_command_pattern",
            "robot_teleop_discard_command_pattern",
            "robot_halt_command_pattern",
            "robot_halt_reply_pattern",
        )
        missing = [key for key in required if not signal_files.get(key)]
        if missing:
            raise ValueError(f"Missing runtime signals: {missing}")
        super().__init__(self.config["capture"]["summary_shm"])
        self.session = session
        self.halt_and_home_cfg = halt_and_home
        for key in required:
            setattr(self, key, signal_files[key])
        self._episode_start_signal_path = self.episode_signal_pattern.format(signal="start")
        self._episode_end_signal_path = self.episode_signal_pattern.format(signal="end")
        self._discard_keyboard_pressed = False
        self.keyboard_listener = None
        self.episode_running = False
        self.stop_event = threading.Event()
        self.poll_stats = {"cycles": 0, "max_cycle_s": 0.0, "overruns": 0}

    def stop(self):
        self.running = False
        self.stop_event.set()

    def _signal_handler(self, signum, frame):
        logger.info(f"Collect shutdown requested by signal {signum}")
        self.stop()

    def _read_keyboard_line(self):
        if not sys.stdin.isatty():
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip().lower()
        return None

    def _wait_start(self):
        if self.config["capture"]["enable_button_control"]:
            logger.info("Ready. Press the leader button to start recording.")
            self._wait_for_episode_start_signal()
        else:
            logger.info("Press Enter to start collection")
            while self.running and self._read_keyboard_line() is None:
                self.stop_event.wait(0.01)

    def _signal_and_wait_for_robot_halt(self):
        # Same handshake as the old saver, but shutdown can interrupt the wait.
        pairs = []
        for device in self.halt_and_home_cfg:
            fields = {
                "device_name": device["device_name"],
                "device_id": device["device_id"],
            }
            reply = Path(self.robot_halt_reply_pattern.format(**fields))
            reply.unlink(missing_ok=True)
            command = Path(self.robot_halt_command_pattern.format(**fields))
            command.parent.mkdir(parents=True, exist_ok=True)
            command.touch()
            pairs.append(reply)
        while self.running and not all(reply.exists() for reply in pairs):
            self.stop_event.wait(0.1)
        for reply in pairs:
            reply.unlink(missing_ok=True)

    def collect_once(self):
        if self.session.active is None:
            return
        start = time.monotonic()
        stats = self.poll_stats
        if stats["cycles"] == 0:
            self._last_poll_start = None
            stats.update(
                interval_count=0,
                interval_total_s=0.0,
                max_interval_s=0.0,
                intervals_over_1_5x=0,
                max_read_s={},
                max_enqueue_s={},
            )
        if self._last_poll_start is not None:
            gap = start - self._last_poll_start
            stats["interval_count"] += 1
            stats["interval_total_s"] += gap
            stats["max_interval_s"] = max(stats["max_interval_s"], gap)
            stats["intervals_over_1_5x"] += int(gap > 1.5 * self.config["capture"]["poll_interval"])
        self._last_poll_start = start
        for device in self.config["inputs"].values():
            read_start = time.monotonic()
            result = self._read_device_data(device)
            read_end = time.monotonic()
            stats["max_read_s"][device] = max(stats["max_read_s"].get(device, 0.0), read_end - read_start)
            if result is not None:
                timestamp, data = result
                self.session.active.record(device, timestamp, data)
                stats["max_enqueue_s"][device] = max(
                    stats["max_enqueue_s"].get(device, 0.0),
                    time.monotonic() - read_end,
                )
        elapsed = time.monotonic() - start
        self.poll_stats["cycles"] += 1
        self.poll_stats["max_cycle_s"] = max(self.poll_stats["max_cycle_s"], elapsed)
        if elapsed > self.config["capture"]["poll_interval"]:
            self.poll_stats["overruns"] += 1

    def _record_episode(self):
        cache = self.session.start_episode()
        self._discard_keyboard_pressed = False
        self.episode_running = True
        self._clear_episode_signal("end")
        self.poll_stats = {"cycles": 0, "max_cycle_s": 0.0, "overruns": 0}
        self._signal_i2rt_teleop_start()
        self.stop_event.wait(self.config["capture"]["teleop_start_wait_s"])
        logger.info(f"🟢 Recording {cache.path.name}; press D to discard, or the leader button to finish.")
        discarded, ended = False, False
        interval = self.config["capture"]["poll_interval"]
        deadline = time.monotonic()
        try:
            while self.running:
                self.collect_once()
                line = self._read_keyboard_line()
                discarded = self._discard_keyboard_pressed or line == "d"
                ended = self._check_episode_end_signal()
                if not self.config["capture"]["enable_button_control"] and line == "":
                    ended = True
                if discarded or ended:
                    break
                deadline += interval
                self.stop_event.wait(max(0, deadline - time.monotonic()))
                if time.monotonic() - deadline > interval:
                    deadline = time.monotonic()
        finally:
            self.episode_running = False
            cache.manifest["polling"] = self.poll_stats
            if discarded:
                self._signal_i2rt_teleop_discard()
            self.session.end_episode(discard=discarded, interrupted=not ended and not discarded)
        if self.running:
            self._signal_and_wait_for_robot_halt()
        if self.running and self._signal_i2rt_home():
            self.stop_event.wait(self.config["capture"]["episode_home_wait_s"])

    def run(self):
        self.running = True
        try:
            if self.session is None:
                self.session = RecordingSession(self.config)
            while self.running and not self.connect():
                self.stop_event.wait(1)
            if not self.running:
                return
            missing = set(self.config["inputs"].values()) - self.devices.keys()
            if missing:
                raise ValueError(f"Configured streams missing from SHM: {sorted(missing)}")
            self._setup_keyboard_listener()
            if self._signal_i2rt_home():
                self.stop_event.wait(self.config["capture"]["initial_home_wait_s"])
            while self.running:
                while self.running and not self.session.can_start():
                    if not getattr(self, "_backlog_reported", False):
                        logger.info("⏳ Saving backlog is full. Waiting for an episode to finish saving...")
                        self._backlog_reported = True
                    self.stop_event.wait(0.1)
                self._backlog_reported = False
                if not self.running:
                    break
                self._wait_start()
                if self.running:
                    self._record_episode()
        finally:
            self.stop()
            self._stop_keyboard_listener()
            self.disconnect()
            if self.session is not None:
                self.session.close()


@hydra.main(config_path="../configs", config_name="collect", version_base="1.3")
def main(cfg: DictConfig):
    config = resolved_consumer_config(cfg)
    check_runtime()
    if cfg.get("finalize_pending", False):
        session = RecordingSession(config)
        try:
            session.finalize_pending()
        finally:
            session.close()
        if session.errors:
            raise RuntimeError("Some episodes failed; inspect their retained manifests")
        return
    signals = OmegaConf.to_container(cfg.runtime.signal_files, resolve=True)
    halt_and_home = OmegaConf.to_container(cfg.get("halt_and_home", []), resolve=True)
    LeRobotCollectConsumer(config, signals, halt_and_home).run()


if __name__ == "__main__":
    if "--finalize-pending" in sys.argv:
        sys.argv.remove("--finalize-pending")
        sys.argv.append("+finalize_pending=true")
    main()
