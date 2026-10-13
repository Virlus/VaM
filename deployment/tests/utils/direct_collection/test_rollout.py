"""Connector adapter acknowledgements and policy identity without a model."""

import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils import lerobot_rollout as rollout
from utils.lerobot_online.protocol import atomic_json, read_json


@pytest.fixture(autouse=True)
def no_terminal(monkeypatch):
    monkeypatch.setattr(rollout, "RolloutProgress", Mock())


def connector(config, policy):
    return SimpleNamespace(
        infer_rollout_config={
            "format": "lerobot",
            "lerobot": config,
            "output_dir": config["output_dir"],
            "task_name": config["task_name"],
            "ready_timeout": 0.1,
            "signal_timeout": 0.1,
        },
        signal_files=config["signal_files"],
        enable_value_prediction=True,
        value_shm=SimpleNamespace(created=False, create=Mock(), get_config=lambda: {"shm": "real_value"}),
        action_chunk_shm=SimpleNamespace(create=Mock()),
        _get_action_chunk_shape=lambda: (10, 14),
        rollout_python_path="python",
        current_policy_info=policy,
        policy_params={"ckpt_dir": "/ckpt"},
        _initial_policy_info=lambda: policy,
        infer_rollout_process=Mock(poll=Mock(return_value=None)),
    )


def test_start_saver_and_policy_signals(config, policy, monkeypatch):
    target = connector(config, policy)
    target.infer_rollout_config["excluded_control_flag_values"] = [0.5]

    def launch(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        Path(config["signal_files"]["rollout_ready"]).write_text("ready")
        return target.infer_rollout_process

    monkeypatch.setattr(rollout.subprocess, "Popen", launch)
    rollout.start_saver(target)
    saved = read_json(Path(config["output_dir"]) / ".runtime/saver.json")
    assert saved["inputs"]["value_prediction"] == "real_value"
    assert saved["excluded_control_flag_values"] == [0.5]
    assert saved["collection_session_id"]
    assert target._infer_rollout_started
    request = Mock()
    monkeypatch.setattr(rollout, "request", request)
    target._episode_count = 40
    rollout.start_recording(target)
    rollout.start_recording(target)
    assert request.call_count == 1
    assert request.call_args.args[2]["policy"] == policy
    assert request.call_args.args[2]["rollout_index"] == 40
    Path(target.signal_files["episode_result"]).write_text("failure")
    rollout.stop_recording(target)
    assert request.call_args.args[2]["result"] == "failure"
    rollout.stop_recording(target)
    assert request.call_count == 2
    target.current_policy_info = None
    target._episode_count = 41
    rollout.start_recording(target)
    assert request.call_args.args[2]["rollout_index"] == 41
    rollout.stop_recording(target)
    assert request.call_args.args[2]["result"] == "interrupted"


def test_restart_continues_after_retained_history(config, policy, monkeypatch):
    """The counter includes a queued conversion and survives policy changes."""
    target = connector(config, policy)
    root = Path(config["output_dir"])
    for index, stage in enumerate(("ready", "capture")):
        atomic_json(
            root / stage / str(index) / "manifest.json",
            {
                "episode_uid": str(index),
                "start_ns": index,
                "config": {"run_name": config["run_name"]},
                "normal_end": True,
                "result": "success",
                "rollout_index": 0,
            },
        )

    def launch(*args, **kwargs):
        Path(config["signal_files"]["rollout_ready"]).write_text("ready")
        return target.infer_rollout_process

    monkeypatch.setattr(rollout.subprocess, "Popen", launch)
    rollout.start_saver(target)
    assert target._episode_count == 2
    saved = read_json(root / ".runtime/saver.json")
    assert saved["history_rollout_indices"] == {"0": 0, "1": 1}
    request = Mock()
    monkeypatch.setattr(rollout, "request", request)
    rollout.start_recording(target)
    assert request.call_args.args[2]["rollout_index"] == 2
    target._lerobot_recording = False
    target._episode_count += 1
    target.current_policy_info = {**policy, "policy_id": 999}
    rollout.start_recording(target)
    assert request.call_args.args[2]["rollout_index"] == 3


@pytest.mark.parametrize("failure", ["exit", "timeout", "no_value"])
def test_startup_failure(config, policy, monkeypatch, failure):
    target = connector(config, policy)
    process = target.infer_rollout_process
    target.infer_rollout_config["ready_timeout"] = 0.005
    if failure == "exit":
        process.poll.return_value = 1
    elif failure == "no_value":
        target.enable_value_prediction = False
    monkeypatch.setattr(rollout.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises((RuntimeError, TimeoutError, ValueError)):
        rollout.start_saver(target)
    if failure == "timeout":
        process.send_signal.assert_called_once_with(signal.SIGINT)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_shutdown_waits_past_timeout_and_repeated_interrupt(config, policy):
    """A backlog outlives the legacy timeout and cannot be killed by Ctrl+C."""
    target = connector(config, policy)
    target.infer_rollout_config["shutdown_timeout"] = 0.001
    process = target.infer_rollout_process
    process.wait.side_effect = [
        subprocess.TimeoutExpired("saver", 0.25),
        KeyboardInterrupt(),
        0,
    ]
    process.returncode = 0
    rollout.drain_saver(target)
    assert process.wait.call_count == 3
    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    assert target.infer_rollout_process is None


@pytest.mark.parametrize("response", ["success", "error", "exit", "timeout"])
def test_ack_request(config, policy, monkeypatch, response):
    target = connector(config, policy)
    target.infer_rollout_config["signal_timeout"] = 0.005

    def send(path, request):
        atomic_json(path, request)
        atomic_json(
            str(path) + ".ack",
            {
                "request_id": (request["request_id"] if response != "timeout" else "old-id"),
                "state": "recording",
                "error": "queue failed" if response == "error" else None,
            },
        )

    monkeypatch.setattr(rollout, "atomic_json", send)
    if response == "exit":
        target.infer_rollout_process.poll.return_value = 1
    if response == "success":
        assert rollout.request(target, "start_recording", {})["state"] == "recording"
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            rollout.request(target, "start_recording", {})
