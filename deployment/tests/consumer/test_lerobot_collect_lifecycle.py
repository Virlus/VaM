"""Mocked SHM and temporary signal files; no physical robot commands."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_lerobot_recording import Executor

from consumer.lerobot_data_saver import LeRobotCollectConsumer
from utils.collection_signals import CollectionSignals
from utils.lerobot_recording import RecordingSession


@pytest.fixture
def collector(collect_config, tmp_path, monkeypatch):
    signals = {
        "episode_signal_pattern": str(tmp_path / "episode_{signal}"),
        "robot_home_command_pattern": str(tmp_path / "{device_name}_{device_id}_home"),
        "robot_teleop_start_command_pattern": str(tmp_path / "{device_name}_{device_id}_start"),
        "robot_teleop_discard_command_pattern": str(tmp_path / "{device_name}_{device_id}_discard"),
        "robot_halt_command_pattern": str(tmp_path / "{device_name}_{device_id}_halt"),
        "robot_halt_reply_pattern": str(tmp_path / "{device_name}_{device_id}_reply"),
    }
    session = RecordingSession(collect_config, executor=Executor())
    obj = LeRobotCollectConsumer(collect_config, signals, [], session=session)
    obj.running = True
    obj.devices = {
        key: {
            "name": key,
            "type": "I2RTRobot" if "Robot" in key else "OpenCVCameraDevice",
        }
        for key in collect_config["inputs"].values()
    }
    monkeypatch.setattr(obj, "_read_keyboard_line", lambda: None)
    yield obj
    if not session.lock.closed:
        session.close()


@pytest.mark.parametrize("result", ["success", "discard", "stale_reconnect", "shutdown"])
def test_record_lifecycle(collector, monkeypatch, tmp_path, result):
    """Only explicit discard removes data; a retired reconnect signal has no effect."""
    reads = []

    def read(device):
        reads.append(device)
        if len(reads) == len(collector.config["inputs"]):
            if result == "success":
                Path(collector._episode_end_signal_path).touch()
            elif result == "discard":
                collector._discard_keyboard_pressed = True
            elif result == "stale_reconnect":
                (tmp_path / "reconnect").touch()
                Path(collector._episode_end_signal_path).touch()
            else:
                collector.stop()
        return 123456, (np.zeros((16, 16, 3), np.uint8) if "Camera" in device else np.zeros(14))

    monkeypatch.setattr(collector, "_read_device_data", read)
    collector._record_episode()
    assert len(reads) == 5
    jobs = collector.session.executor.jobs
    paths = list(collector.session.root.glob("episode_*_temp"))
    if result in {"success", "stale_reconnect"}:
        assert len(jobs) == 1
        assert paths
        import json

        metadata = json.loads((paths[0] / "manifest.json").read_text())
        assert metadata["success"] is True and metadata["polling"]["cycles"] == 1
    elif result == "discard":
        assert not jobs and not paths
    else:
        assert not jobs and paths
    assert not collector.episode_running


def test_record_exception_is_not_success(collector, monkeypatch):
    def fail(*args):
        raise RuntimeError("read failed")

    monkeypatch.setattr(collector, "_read_device_data", fail)
    with pytest.raises(RuntimeError):
        collector._record_episode()
    assert not collector.session.executor.jobs
    assert collector.session.active is None


def test_collect_empty_and_poll_metrics(collector, monkeypatch):
    collector.collect_once()
    collector.session.start_episode()
    monkeypatch.setattr(collector, "_read_device_data", lambda key: None)
    collector.config["capture"]["poll_interval"] = 1e-12
    collector.collect_once()
    assert collector.poll_stats["cycles"] == 1
    assert collector.poll_stats["overruns"] == 1


def test_poll_gap_includes_time_outside_collect_once(collector, monkeypatch):
    """Scheduling delays must appear even when each SHM read is fast."""
    collector.session.start_episode()
    clock = [1.0]
    monkeypatch.setattr("consumer.lerobot_data_saver.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(collector, "_read_device_data", lambda key: None)
    collector.collect_once()
    clock[0] += 0.02
    collector.collect_once()
    assert collector.poll_stats["overruns"] == 0
    assert collector.poll_stats["intervals_over_1_5x"] == 1
    assert collector.poll_stats["max_interval_s"] == pytest.approx(0.02)


def test_signals_and_identity(collector):
    assert collector._parse_robot_identity("I2RTRobot_2_control_flag", {}) == (
        "I2RTRobot",
        "2",
    )
    assert collector._parse_robot_identity("unknown", {"id": 3}) == ("I2RTRobot", "3")
    assert collector._parse_robot_identity("unknown", {}) is None
    assert collector._signal_i2rt_home() == 2
    assert collector._signal_i2rt_teleop_start() == 2
    assert collector._signal_i2rt_teleop_discard() == 2
    Path(collector._episode_start_signal_path).touch()
    collector._wait_start()
    assert not Path(collector._episode_start_signal_path).exists()
    assert not collector._check_episode_end_signal()
    Path(collector._episode_end_signal_path).touch()
    assert collector._check_episode_end_signal()
    collector._clear_episode_signal("end")
    collector._signal_handler(15, None)
    assert not collector.running
    collector.devices = {}
    assert collector._signal_i2rt_home() == 0


def test_halt_handshake(collector, monkeypatch):
    collector.halt_and_home_cfg = [{"device_name": "I2RTRobot", "device_id": 0}]
    reply = Path(collector.robot_halt_reply_pattern.format(device_name="I2RTRobot", device_id=0))
    monkeypatch.setattr(collector.stop_event, "wait", lambda timeout: reply.touch())
    collector._signal_and_wait_for_robot_halt()
    assert not reply.exists()
    assert Path(collector.robot_halt_command_pattern.format(device_name="I2RTRobot", device_id=0)).exists()
    # The extracted legacy mixin preserves the same handshake behavior.
    reply.touch()
    CollectionSignals._signal_and_wait_for_robot_halt(collector)
    assert not reply.exists()


def test_keyboard_listener_and_enter(collector, monkeypatch):
    calls = []

    class Listener:
        def __init__(self, on_press):
            calls.append(on_press)

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    monkeypatch.setitem(
        sys.modules,
        "pynput",
        SimpleNamespace(keyboard=SimpleNamespace(Listener=Listener)),
    )
    collector._setup_keyboard_listener()
    collector.episode_running = True
    calls[0](SimpleNamespace(char="d"))
    assert collector._discard_keyboard_pressed
    collector._stop_keyboard_listener()
    collector._stop_keyboard_listener()
    assert calls[-1] == "stop"
    collector.config["capture"]["enable_button_control"] = False
    monkeypatch.setattr(collector, "_read_keyboard_line", lambda: "")
    collector._wait_start()
    monkeypatch.setitem(sys.modules, "pynput", None)
    collector._setup_keyboard_listener()
    assert collector.keyboard_listener is None


def test_run_connect_and_shutdown(collector, monkeypatch):
    monkeypatch.setattr(collector, "connect", lambda: True)
    monkeypatch.setattr(collector, "_setup_keyboard_listener", lambda: None)
    starts = []

    def record():
        starts.append(1)
        collector.stop()

    monkeypatch.setattr(collector, "_record_episode", record)
    monkeypatch.setattr(collector, "_wait_start", lambda: None)
    collector.run()
    assert starts == [1]
    assert collector.session.lock.closed


def test_missing_devices(collector, monkeypatch):
    collector.devices = {}
    monkeypatch.setattr(collector, "connect", lambda: True)
    with pytest.raises(ValueError, match="missing from SHM"):
        collector.run()


def test_missing_signals(collect_config):
    with pytest.raises(ValueError, match="Missing runtime signals"):
        LeRobotCollectConsumer(collect_config, {}, [])
