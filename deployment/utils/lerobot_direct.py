"""Direct online ALOHA recording: arrays -> final LeRobot, without recorded data readers.

Only unfinished episodes use the existing disk-backed image cache. The published
directory contains complete action/value datasets and small provenance sidecars.
"""

from __future__ import annotations

import copy
import multiprocessing
import shutil
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from utils.lerobot_episode_names import (
    bundle_directory_name,
    episode_name,
    next_episode_index,
)
from utils.lerobot_frames import OnlineFrames
from utils.lerobot_online.bundle import inventory, validate_repo
from utils.lerobot_online.protocol import (
    atomic_json,
    exclusive_lock,
    fsync_directory,
    read_json,
)
from utils.lerobot_online.writer import write_role
from utils.lerobot_progress import episode_label, progress_event
from utils.lerobot_recording import EpisodeCache, load_cached_streams
from utils.logger_config import logger

STREAMS = {
    "img_ego",
    "img_left",
    "img_right",
    "state",
    "policy_action",
    "human_action",
    "control_flag",
    "value_prediction",
}
CAMERAS = {
    "base_image": "img_ego",
    "left_wrist_image": "img_left",
    "right_wrist_image": "img_right",
}
REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"


def validate_config(config):
    cfg = copy.deepcopy(config)
    # Existing sealed recordings without this field retain their original recipe.
    # New rollout presets explicitly select action_stride=3.
    cfg.setdefault("action_stride", 1)
    # Record the same default used by dense-camera GAE for cloud registration.
    cfg["advantage"].setdefault("action_fps", 100)
    if set(cfg["inputs"]) != STREAMS or len(set(cfg["inputs"].values())) != len(STREAMS):
        raise ValueError("Direct recorder requires eight distinct camera/state/control/value streams")
    if not str(cfg.get("task_name", "")).strip():
        raise ValueError("A task is required")
    if cfg["advantage"].get("mode") != "cloud_p70":
        raise ValueError("Direct recorder preserves cloud_p70; it does not invent local thresholds")
    if cfg["advantage"].get("horizon", 1) != 1:
        raise ValueError("Direct recorder currently supports GAE horizon=1")
    for key in ("window_size", "fps", "action_stride", "value_stride"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("alignment_tolerance_ms",):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("gamma", "gae_lambda"):
        if not 0 < cfg["advantage"][key] <= 1:
            raise ValueError(f"Invalid advantage {key}")
    if cfg["image_color_order"] not in {"bgr", "rgb"}:
        raise ValueError("image_color_order must be bgr or rgb")
    if cfg["capture"]["queue_size"] < 1 or cfg["capture"]["max_pending_episodes"] < 1:
        raise ValueError("Capture capacities must be positive")
    return cfg


def writer_contract(config, policy):
    """Writer dependency/identity checks only; no cloud contract or fabricated hashes."""
    return {
        "run_name": config["run_name"],
        **policy,
        "value_checkpoint_id": policy["checkpoint_id"],
        "lerobot_revision": REVISION,
        "codebase_version": "v2.1",
        "advantage": config["advantage"],
    }


def feature_schema(config, data):
    features = {}
    for output, name in CAMERAS.items():
        spec = {
            "dtype": "video",
            "shape": list(data[name].shape[1:]),
            "names": ["height", "width", "channels"],
        }
        features[output] = spec
        features["next_" + output] = copy.deepcopy(spec)
    for name, shape in {
        "state": [14],
        "next_state": [14],
        "actions": [config["window_size"], 14],
        "next_actions": [config["window_size"], 14],
        "reward": [config["window_size"]],
        **{
            key: [1]
            for key in (
                "success",
                "terminate",
                "value",
                "value_prediction",
                "next_value_prediction",
                "data_source",
                "control_flag",
                "policy_id",
                "checkpoint_index",
                "source_action_index",
            )
        },
    }.items():
        features[name] = {
            "dtype": "int64" if name == "source_action_index" else "float32",
            "shape": shape,
            "names": None,
        }
    return features


def finalize_direct_episode(path, *, progress=None):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if not manifest.get("normal_end") or manifest.get("result") not in {
        "success",
        "failure",
    }:
        raise ValueError("Only sealed, explicitly ended episodes may be finalized")
    cfg = validate_config(manifest["config"])
    root = Path(cfg["output_dir"])
    ready = root / "ready" / bundle_directory_name(manifest)
    building = root / "building" / manifest["episode_uid"]
    if ready.exists():
        existing = validate_direct_episode(ready)
        if existing["episode_uid"] != manifest["episode_uid"]:
            raise ValueError(f"Episode directory already belongs to another rollout: {ready}")
        return str(ready)
    if any(manifest["dropped"].values()) or any(v["backwards"] for v in manifest["observed_timestamp_gaps"].values()):
        raise ValueError("Recorded streams contain dropped samples or clock regression")
    data, _ = load_cached_streams(path, manifest)
    if progress:
        progress("aligning")
    keyed_data, timestamps = {}, {}
    for name, stream_name in cfg["inputs"].items():
        stream = manifest["streams"].get(stream_name)
        if stream is None and name in {"policy_action", "human_action"}:
            keyed_data[name], timestamps[name] = (
                np.empty((0, 14)),
                np.empty(0, dtype=np.int64),
            )
            continue
        if stream is None:
            raise ValueError(f"Missing captured stream {name}")
        keyed_data[name] = data[stream_name]
        timestamps[name] = np.fromfile(path / "cache" / f"{stream['file']}.timestamps", dtype="<i8")
    frames = OnlineFrames(
        keyed_data,
        timestamps,
        cfg,
        manifest["policy"],
        manifest["result"],
        manifest["start_ns"],
        manifest["stop_ns"],
    )
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    features = feature_schema(cfg, keyed_data)
    writer_config = {
        "features": features,
        "robot_type": "aloha_i2rt",
        "conversion": {"fps": cfg["fps"]},
        "writer": cfg["writer"],
    }
    contract = writer_contract(cfg, manifest["policy"])
    outputs = {}
    for role in ("action", "value"):
        writer_config["features"] = copy.deepcopy(features)
        if role == "action":
            writer_config["features"]["advantage_raw"] = {
                "dtype": "float64",
                "shape": [1],
                "names": None,
            }
            episodes = [frames.frames("action", frames.end)]
        else:
            episodes = [frames.frames("value", prefix["end_index"], i) for i, prefix in enumerate(frames.prefixes)]
        outputs[role] = write_role(
            building / role,
            episodes,
            writer_config,
            role,
            contract,
            raw_advantage=True,
            progress=progress,
        )
    if progress:
        progress("publishing")
    atomic_json(building / "episode_policy.json", manifest["policy"])
    atomic_json(
        building / "alignment.json",
        {
            "action": frames.audit["action"],
            "value": frames.audit["value"],
            "value_prefixes": frames.prefixes,
            "selected_commands": [
                {
                    "timestamp_ns": ts,
                    "stream": name,
                    "source_row": row,
                    "control_flag": flag,
                }
                for ts, name, row, flag in frames.action_provenance
            ],
            "excluded_intervals_ns": frames.intervals,
            "compacted_intervals_ns": frames.compacted_intervals,
        },
    )
    output = {
        "schema_version": "lerobot_direct_episode/v1",
        "episode_uid": manifest["episode_uid"],
        "run_name": cfg["run_name"],
        "task": cfg["task_name"],
        "result": manifest["result"],
        "policy": manifest["policy"],
        "start_ns": manifest["start_ns"],
        "stop_ns": manifest["stop_ns"],
        "recipe": {
            k: cfg[k]
            for k in (
                "window_size",
                "fps",
                "action_stride",
                "value_stride",
                "alignment_tolerance_ms",
                "latency_ms",
                "advantage",
            )
        },
        "sampling": frames.sampling,
        "outputs": outputs,
        "codebase_version": "v2.1",
        "lerobot_revision": REVISION,
        "files": inventory(building),
        "advantage_status": "cloud_p70_pending",
        "sample_semantics": "lerobot_camera_v1",
    }
    for key in ("rollout_index", "collection_session_id", "episode_name"):
        if key in manifest:
            output[key] = manifest[key]
    atomic_json(building / "manifest.json", output)
    for file in building.rglob("*"):
        if file.is_file():
            with file.open("rb") as handle:
                import os

                os.fsync(handle.fileno())
    ready.parent.mkdir(parents=True, exist_ok=True)
    building.rename(ready)
    fsync_directory(ready.parent)
    # Only the successful output is published; failed caches remain retryable.
    shutil.rmtree(path)
    return str(ready)


def validate_direct_episode(path, *, decode=False):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema_version") != "lerobot_direct_episode/v1" or manifest.get("result") not in {
        "success",
        "failure",
    }:
        raise ValueError("Invalid direct LeRobot episode manifest")
    if manifest["files"] != inventory(path) or manifest["policy"] != read_json(path / "episode_policy.json"):
        raise ValueError("Direct LeRobot episode content/policy changed")
    if set(manifest["outputs"]) != {"action", "value"}:
        raise ValueError("Both action and value outputs are required")
    if decode:
        cfg = {**manifest["recipe"], "run_name": manifest["run_name"]}
        contract = writer_contract(cfg, manifest["policy"])
        for role in ("action", "value"):
            validate_repo(
                path / role,
                manifest["outputs"][role]["episode_lengths"],
                role,
                contract,
                raw_advantage=True,
            )
    return manifest


def ignore_worker_interrupts():
    """Terminal interrupts stop acquisition, never a sealed episode's writer."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def finalize_with_progress(path, progress_dir, rollout_index=None):
    """Report when work actually starts, not when a Future enters the pool queue."""
    manifest = read_json(Path(path) / "manifest.json")
    status = {
        "episode": episode_label(manifest, rollout_index),
        "episode_uid": manifest["episode_uid"],
        "state": "processing",
        "stage": "loading",
        "started_at": time.time(),
    }
    status_path = Path(progress_dir) / "worker.json" if progress_dir else None
    display_name = manifest.get("episode_name", f"rollout #{status['episode']}")

    def report(stage):
        status["stage"] = stage
        if status_path:
            atomic_json(status_path, status)

    report("loading")
    try:
        result = finalize_direct_episode(path, progress=report)
    except BaseException as exc:
        status.update(state="failed", ended_at=time.time(), error=str(exc))
        status["event"] = progress_event(
            f"LeRobot convert failed: {display_name} | source retained: {path}",
            "error",
        )
        if status_path:
            atomic_json(status_path, status)
        raise
    status.update(state="finished", ended_at=time.time())
    status["event"] = progress_event(
        f"LeRobot convert success: {display_name} | {status['ended_at'] - status['started_at']:.1f}s | ready: {result}"
    )
    if status_path:
        atomic_json(status_path, status)
    return result


class DirectSession:
    """Reuse the existing raw cache; save at most one episode at a time in a child process."""

    def __init__(self, config, *, executor=None):
        self.config = validate_config(config)
        self.root = Path(config["output_dir"])
        (self.root / "capture").mkdir(parents=True, exist_ok=True)
        self.lock = exclusive_lock(self.root / ".direct_recorder.lock")
        self.lock.__enter__()
        self._next_episode_index = next_episode_index(self.root, self.config["run_name"])
        self.active = None
        self.executor = executor
        self.futures = {}
        self.errors = []
        self.completed = 0
        self.phase = "idle"
        self.last_rollout_index = None
        self.last_result = None
        self.event = {}
        self._last_status = None
        self.recovery = [
            p.parent for p in (self.root / "capture").glob("*/manifest.json") if read_json(p).get("normal_end")
        ]
        # A new terminal session starts with the latest completed rollout in this run.
        directory = self.config.get("progress_dir")
        if directory and not (Path(directory) / "worker.json").exists():
            history = self.config.get("history_rollout_indices", {})
            latest = None
            for path in (self.root / "ready").glob("*/manifest.json"):
                item = read_json(path)
                if item.get("run_name") != self.config["run_name"]:
                    continue
                index = item.get("rollout_index") if item.get("episode_name") else history.get(item.get("episode_uid"))
                if type(index) is int and (latest is None or index > latest[0]):
                    latest = (index, item)
            if latest is not None:
                self.last_rollout_index = latest[0]
                self.last_result = latest[1].get("result")
                atomic_json(
                    Path(directory) / "worker.json",
                    {
                        "episode": str(latest[0]),
                        "state": "finished",
                        "episode_uid": latest[1]["episode_uid"],
                    },
                )
        self.publish_status()

    def publish_status(self):
        directory = self.config.get("progress_dir")
        if not directory:
            return
        status = {
            "phase": self.phase,
            "rollout_index": (self.active.manifest.get("rollout_index") if self.active else None),
            "last_rollout_index": self.last_rollout_index,
            "last_result": self.last_result,
            "started_at": (self.active.manifest["start_ns"] / 1e9 if self.active else None),
            "event": self.event,
            "pending": len(self.futures) + len(self.recovery),
            "pending_uids": [Path(p).name.removesuffix("_temp") for p in self.futures]
            + [p.name.removesuffix("_temp") for p in self.recovery],
            "completed": self.completed,
            "errors": list(self.errors),
        }
        if status != self._last_status:
            atomic_json(Path(directory) / "session.json", status)
            self._last_status = status

    def reap(self):
        for path, future in list(self.futures.items()):
            if future.done():
                try:
                    future.result()
                    self.completed += 1
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"{path}: {exc}")
                    logger.exception(f"LeRobot save failed; source retained at {path}")
                del self.futures[path]
        while self.recovery and len(self.futures) < self.config["capture"]["max_pending_episodes"]:
            self.submit(self.recovery.pop(0))
        self.publish_status()

    def submit(self, path):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=ignore_worker_interrupts,
            )
        uid = read_json(Path(path) / "manifest.json")["episode_uid"]
        self.futures[str(path)] = self.executor.submit(
            finalize_with_progress,
            str(path),
            self.config.get("progress_dir"),
            self.config.get("history_rollout_indices", {}).get(uid),
        )
        self.publish_status()

    def start(self, policy, start_ns, rollout_index=None):
        self.reap()
        if self.errors:
            raise RuntimeError(f"Fix the retained failed episode before continuing: {self.errors[-1]}")
        if self.active is not None:
            raise RuntimeError("An episode is already active")
        if self.recovery or len(self.futures) >= self.config["capture"]["max_pending_episodes"]:
            self.phase = "wait conversion"
            self.publish_status()
            return False
        if shutil.disk_usage(self.root).free < self.config["capture"].get("min_free_bytes", 2 * 1024**3):
            raise RuntimeError("Insufficient disk space for another LeRobot episode")
        uid = str(uuid.uuid4())
        index = max(
            self._next_episode_index,
            rollout_index if rollout_index is not None else self._next_episode_index,
        )
        name = episode_name(index)
        self.active = EpisodeCache(self.root / "capture" / f"{uid}_temp", self.config)
        self.active.manifest.update(
            episode_uid=uid,
            policy=dict(policy),
            start_ns=int(start_ns),
            rollout_index=index,
            episode_name=name,
            collection_session_id=self.config.get("collection_session_id"),
        )
        atomic_json(self.active.path / "manifest.json", self.active.manifest)
        self._next_episode_index = index + 1
        self.phase = "recording"
        self.publish_status()
        return True

    def stop(self, result, stop_ns):
        if self.active is None:
            return
        cache, self.active = self.active, None
        self.last_rollout_index = cache.manifest["rollout_index"]
        self.last_result = result
        if self.phase != "draining":
            self.phase = "idle"
        name = cache.manifest["episode_name"]
        if result == "discard":
            cache.close(discard=True)
            # Reuse the reservation only after its source was removed. Successful
            # and interrupted caches keep their numbers to prevent collisions.
            if "rollout_index" in cache.manifest:
                self._next_episode_index = cache.manifest["rollout_index"]
            self.event = progress_event(f"{name} discarded | no conversion", "warning")
            self.publish_status()
            return
        valid = result in {"success", "failure"}
        cache.close(normal_end=valid)
        cache.manifest.update(
            result=result,
            success=result == "success" if valid else None,
            stop_ns=int(stop_ns),
        )
        atomic_json(cache.path / "manifest.json", cache.manifest)
        if valid and cache.manifest["normal_end"]:
            self.event = progress_event(f"Episode record stopped: {name} | result={result.upper()} | queued")
            self.submit(cache.path)
        else:
            self.event = progress_event(f"{name} interrupted | unfinished source retained", "warning")
        self.publish_status()

    def close(self):
        try:
            self.phase = "draining"
            self.stop("interrupted", time.time_ns())
            self.publish_status()
            while self.recovery or self.futures:
                self.reap()
                time.sleep(0.05)
            if self.executor is not None:
                self.executor.shutdown(wait=True)
            self.reap()
            self.phase = "failed" if self.errors else "finished"
            self.publish_status()
        finally:
            self.lock.__exit__(None, None, None)
