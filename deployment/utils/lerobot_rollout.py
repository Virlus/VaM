"""Small adapter from PolicyConnector's existing signals to the direct saver."""

from __future__ import annotations

import copy
import signal
import subprocess
import time
import uuid
from pathlib import Path

from utils.lerobot_episode_names import next_episode_index
from utils.lerobot_online.protocol import atomic_json, read_json
from utils.lerobot_progress import RolloutProgress, run_rollout_history
from utils.logger_config import logger


def enabled(connector):
    return connector.infer_rollout_config.get("format", "lerobot") == "lerobot"


def start_saver(connector):
    from omegaconf import OmegaConf

    rollout = connector.infer_rollout_config
    config = rollout["lerobot"]
    config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else copy.deepcopy(config)
    config.update(
        output_dir=rollout["output_dir"],
        task_name=rollout["task_name"],
        signal_files=dict(connector.signal_files),
        excluded_control_flag_values=list(rollout.get("excluded_control_flag_values", [])),
    )
    session_id = uuid.uuid4().hex
    config["collection_session_id"] = session_id
    config["progress_dir"] = str(Path(config["output_dir"]) / ".runtime" / session_id)
    history = run_rollout_history(config["output_dir"], config["run_name"])
    config["history_rollout_indices"] = history
    connector._episode_count = next_episode_index(config["output_dir"], config["run_name"])
    logger.info(
        f"LeRobot run {config['run_name']}: {len(history)} retained rollouts; next rollout #{connector._episode_count}"
    )
    if not connector.enable_value_prediction:
        raise ValueError("Direct online LeRobot collection requires real value prediction")
    if not connector.value_shm.created:
        connector.value_shm.create((1,))
    config["inputs"]["value_prediction"] = connector.value_shm.get_config()["shm"]
    connector.action_chunk_shm.create(connector._get_action_chunk_shape())
    path = Path(config["output_dir"]) / ".runtime" / "saver.json"
    atomic_json(path, config)
    for name in (
        "start_recording",
        "stop_recording",
        "rollout_ready",
        "episode_result",
    ):
        signal = Path(connector.signal_files[name])
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.unlink(missing_ok=True)
        Path(str(signal) + ".ack").unlink(missing_ok=True)
    script = Path(__file__).resolve().parents[1] / "consumer" / "async_infer_data_saver_lerobot.py"
    process = subprocess.Popen(
        [connector.rollout_python_path, str(script), "--config", str(path)],
        cwd=script.parent.parent,
        start_new_session=True,
    )
    connector.infer_rollout_process = process
    connector._lerobot_progress = RolloutProgress(config["progress_dir"]).start()
    ready = Path(connector.signal_files["rollout_ready"])
    deadline = time.monotonic() + float(rollout.get("ready_timeout", 60))
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"LeRobot saver exited with status {process.returncode}")
            if ready.exists() and ready.read_text().strip() == "ready":
                connector._infer_rollout_started = True
                connector._lerobot_recording = False
                return
            time.sleep(0.05)
        raise TimeoutError("LeRobot saver did not connect to required streams")
    except BaseException:
        drain_saver(connector)
        raise


def drain_saver(connector):
    """Wait without a conversion timeout; subsequent Ctrl+C cannot kill the queue."""
    connector._lerobot_shutting_down = True
    progress = getattr(connector, "_lerobot_progress", None)
    if progress is not None:
        progress.stopping = True
    process = connector.infer_rollout_process
    try:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            logger.info("Rollout stopped; waiting for all queued LeRobot conversions")
            while True:
                try:
                    process.wait(timeout=0.25)
                    break
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    continue
            if process.returncode:
                logger.error(f"LeRobot saver exited with code {process.returncode}; inspect retained caches")
            else:
                logger.info("All queued LeRobot conversions finished")
            connector._lerobot_saver_exit_code = process.returncode
    finally:
        connector.infer_rollout_process = None
        connector._infer_rollout_started = False
        if progress is not None:
            progress.close()
            connector._lerobot_progress = None


def request(connector, name, payload):
    path = Path(connector.signal_files[name])
    ack_path = Path(str(path) + ".ack")
    request_id = str(uuid.uuid4())
    atomic_json(path, {"request_id": request_id, **payload})
    deadline = time.monotonic() + float(connector.infer_rollout_config.get("signal_timeout", 300))
    while time.monotonic() < deadline:
        if connector.infer_rollout_process.poll() is not None:
            raise RuntimeError("LeRobot saver stopped; inference must not continue without recording")
        if ack_path.exists():
            ack = read_json(ack_path)
            if ack["request_id"] == request_id:
                if ack.get("error"):
                    raise RuntimeError(f"LeRobot recorder rejected {name}: {ack['error']}")
                return ack
        time.sleep(0.01)
    raise TimeoutError(f"LeRobot recorder did not acknowledge {name}")


def start_recording(connector):
    if getattr(connector, "_lerobot_recording", False):
        return
    progress = getattr(connector, "_lerobot_progress", None)
    if progress is not None:
        progress.set_phase("starting")
    info = connector.current_policy_info or connector._initial_policy_info()
    policy = {name: info[name] for name in ("policy_id", "checkpoint_id", "checkpoint_index")}
    policy["checkpoint_dir"] = str(info.get("checkpoint_dir") or connector.policy_params["ckpt_dir"])
    ack = request(
        connector,
        "start_recording",
        {
            "policy": policy,
            "start_ns": time.time_ns(),
            "rollout_index": getattr(connector, "_episode_count", 0),
        },
    )
    if isinstance(ack, dict) and type(ack.get("rollout_index")) is int:
        connector._episode_count = ack["rollout_index"]
    connector._lerobot_recording = True
    if progress is not None:
        progress.set_phase("idle")


def stop_recording(connector):
    if not getattr(connector, "_lerobot_recording", False):
        return
    result_file = Path(connector.signal_files["episode_result"])
    result = result_file.read_text().strip() if result_file.exists() else "interrupted"
    request(connector, "stop_recording", {"result": result, "stop_ns": time.time_ns()})
    result_file.unlink(missing_ok=True)
    connector._lerobot_recording = False
