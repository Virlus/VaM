"""Disk-backed collect episodes and recoverable paired LeRobot v2.1 output.

No hardware, keyboard, model or recorded data SDK imports. Each output uses an independent
PNG/video writer; raw images are disk-backed during episode construction.
"""

from __future__ import annotations

import copy
import fcntl
import json
import multiprocessing
import os
import queue
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np

from utils.lerobot_conversion_logging import quiet_lerobot_writer
from utils.lerobot_images import legacy_jpeg_rgb
from utils.logger_config import logger

TRAINING_ROOT = Path(__file__).resolve().parents[2] / "training"


def check_runtime():
    """The pinned v2.1 reader predates datasets 4's Column API."""
    if int(version("datasets").split(".")[0]) >= 4:
        raise RuntimeError(
            "LeRobot v2.1 requires datasets<4 for round-trip reading. "
            "Install requirements-lerobot-collect.txt in the recording environment."
        )


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_config(config):
    """Fail before connecting hardware if the supported offline recipe is invalid."""
    cfg = copy.deepcopy(config)
    cfg.setdefault("start_index", 0)
    if type(cfg["start_index"]) is not int or cfg["start_index"] < 0:
        raise ValueError("start_index must be a non-negative integer")
    if cfg.get("type") != "lerobot_collect":
        raise ValueError("consumer.type must be lerobot_collect")
    if not str(cfg.get("task_name", "")).strip():
        raise ValueError("consumer.task_name must not be empty")
    c = cfg["conversion"]
    if c["data_mode"] != "offline" or c["reward_range"] != "binary" or c["split_by_control_flag"]:
        raise ValueError("Collect v1 requires offline binary rewards without control splitting")
    if int(c["window_size"]) < 1 or int(c["fps"]) < 1:
        raise ValueError("window_size and fps must be positive")
    if not np.isfinite(c["max_time_diff_ms"]) or c["max_time_diff_ms"] < 0:
        raise ValueError("Invalid alignment tolerance")
    expected_inputs = {
        "img_ego",
        "img_left",
        "img_right",
        "leader_joint_pos",
        "follower_joint_pos",
    }
    if set(cfg["inputs"]) != expected_inputs or len(set(cfg["inputs"].values())) != 5:
        raise ValueError("Five distinct camera/leader/follower streams are required")
    if set(cfg["outputs"]) != {"action", "value"}:
        raise ValueError("Exactly action and value outputs are required")
    for output in cfg["outputs"].values():
        if (
            output["stride_basis"] != "action_index"
            or type(output["frame_stride"]) is not int
            or output["frame_stride"] < 1
        ):
            raise ValueError("Output strides must be positive and use action_index")
    capture = cfg["capture"]
    if not np.isfinite(capture["poll_interval"]) or capture["poll_interval"] <= 0:
        raise ValueError("poll_interval must be positive")
    capture.setdefault("conversion_workers", 1)
    for key in ("queue_size", "max_pending_episodes", "conversion_workers"):
        if type(capture[key]) is not int or capture[key] < 1:
            raise ValueError(f"capture.{key} must be a positive integer")
    if not all(np.isfinite(v) for v in c["latency_s"].values()):
        raise ValueError("Latency must be finite")
    features = cfg["features"]
    expected = {
        "base_image",
        "left_wrist_image",
        "right_wrist_image",
        "state",
        "actions",
        "next_base_image",
        "next_left_wrist_image",
        "next_right_wrist_image",
        "next_state",
        "next_actions",
        "reward",
        "value",
        "success",
        "data_source",
        "terminate",
    }
    if set(features) != expected:
        raise ValueError("Features must match offline ALOHA, without synthetic predictions")
    for name in ("actions", "next_actions"):
        if list(features[name]["shape"]) != [c["window_size"], 14]:
            raise ValueError(f"{name} must have shape [window_size, 14]")
    if list(features["reward"]["shape"]) != [c["window_size"]]:
        raise ValueError("reward shape must match window_size")
    for name in ("state", "next_state"):
        if list(features[name]["shape"]) != [14]:
            raise ValueError(f"{name} must have 14 channels")
    for name, feature in features.items():
        if "image" in name:
            if feature["dtype"] != "video" or len(feature["shape"]) != 3 or feature["shape"][-1] != 3:
                raise ValueError(f"{name} must be an RGB video")
        elif feature["dtype"] != "float32":
            raise ValueError(f"{name} must use float32")
    for camera in ("base_image", "left_wrist_image", "right_wrist_image"):
        if features[camera] != features[f"next_{camera}"]:
            raise ValueError(f"Current/next schema differs for {camera}")
    writer = cfg["writer"]
    if writer["image_writer_processes"] < 0 or writer["image_writer_threads"] < 1:
        raise ValueError("Invalid image writer concurrency")
    if writer["video"]["vcodec"] not in {"h264", "hevc", "libsvtav1"}:
        raise ValueError("Unsupported video codec")
    cfg["output_dir"] = str(Path(cfg["output_dir"]).expanduser().resolve())
    if cfg.get("publication"):
        from utils.lerobot_collect_publish import validate_publication

        validate_publication(cfg["publication"])
    return cfg


class EpisodeCache:
    """One episode with bounded asynchronous raw-array spooling and timestamp dedup."""

    def __init__(self, directory, config):
        self.path = Path(directory)
        self.path.mkdir()
        (self.path / "cache").mkdir()
        self.config = config
        self.last_timestamp = {}
        self.timestamp_gaps = {}
        self.accepted = dict.fromkeys(config["inputs"].values(), 0)
        self.dropped = dict.fromkeys(config["inputs"].values(), 0)
        self.streams = {}
        self.error = None
        self.peak_queue = 0
        self.queue = queue.Queue(maxsize=config["capture"]["queue_size"])
        self.closed = False
        self.manifest = {
            "schema_version": "collect_lerobot/v1",
            "source_episode_id": str(uuid.uuid4()),
            "status": "recording",
            "normal_end": False,
            "task": config["task_name"],
            "config": config,
            "started_at_ns": time.time_ns(),
        }
        atomic_json(self.path / "manifest.json", self.manifest)
        self.thread = threading.Thread(target=self._write_loop, name="collect-cache", daemon=True)
        self.thread.start()

    def record(self, device, timestamp_ns, array):
        if self.closed:
            raise RuntimeError("Episode cache is closed")
        if device not in self.accepted:
            raise KeyError(device)
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= 0 or self.last_timestamp.get(device) == timestamp_ns:
            return False
        previous = self.last_timestamp.get(device)
        if previous is not None:
            gap = timestamp_ns - previous
            stats = self.timestamp_gaps.setdefault(device, {"count": 0, "total_ns": 0, "max_ns": 0, "backwards": 0})
            stats["count"] += 1
            stats["total_ns"] += gap
            stats["max_ns"] = max(stats["max_ns"], gap)
            stats["backwards"] += int(gap < 0)
        self.last_timestamp[device] = timestamp_ns
        task = (device, timestamp_ns, np.array(array, copy=True))
        try:
            self.queue.put_nowait(task)
        except queue.Full:
            try:
                self.queue.put(task, timeout=0.01)
            except queue.Full:
                self.dropped[device] += 1
                logger.warning(f"Collect cache full: dropping {device} at {timestamp_ns}")
                return False
        self.accepted[device] += 1
        self.peak_queue = max(self.peak_queue, self.queue.qsize())
        return True

    def _write_loop(self):
        handles = {}
        try:
            while True:
                task = self.queue.get()
                try:
                    if task is None:
                        break
                    if self.error is not None:
                        continue
                    device, timestamp, array = task
                    if device not in handles:
                        name = f"stream_{len(handles):02d}"
                        handles[device] = (
                            (self.path / "cache" / f"{name}.bin").open("wb"),
                            (self.path / "cache" / f"{name}.timestamps").open("wb"),
                        )
                        self.streams[device] = {
                            "file": name,
                            "dtype": array.dtype.str,
                            "shape": list(array.shape),
                            "count": 0,
                            "first_timestamp_ns": timestamp,
                        }
                    stream = self.streams[device]
                    if list(array.shape) != stream["shape"] or array.dtype.str != stream["dtype"]:
                        raise ValueError(f"Shape/dtype changed during episode: {device}")
                    handles[device][0].write(array.tobytes(order="C"))
                    handles[device][1].write(np.asarray(timestamp, dtype="<i8").tobytes())
                    stream["count"] += 1
                    stream["last_timestamp_ns"] = timestamp
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)
                    logger.exception("Collect cache write failed")
                finally:
                    self.queue.task_done()
        finally:
            for pair in handles.values():
                for handle in pair:
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except OSError as exc:
                        self.error = str(exc)
                    finally:
                        handle.close()

    def close(self, *, normal_end=False, discard=False):
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.thread.join()
        if discard:
            shutil.rmtree(self.path)
            return
        self.manifest.update(
            {
                "normal_end": bool(normal_end and self.error is None),
                "success": True if normal_end and self.error is None else None,
                "status": ("sealed" if normal_end and self.error is None else "interrupted"),
                "ended_at_ns": time.time_ns(),
                "streams": self.streams,
                "accepted": self.accepted,
                "dropped": self.dropped,
                "peak_queue": self.peak_queue,
                "observed_timestamp_gaps": self.timestamp_gaps,
                "error": self.error,
            }
        )
        atomic_json(self.path / "manifest.json", self.manifest)


def load_cached_streams(path, manifest):
    data, timestamps = {}, {}
    for key, stream in manifest["streams"].items():
        file = Path(path) / "cache" / stream["file"]
        shape = (stream["count"], *stream["shape"])
        dtype = np.dtype(stream["dtype"])
        if file.with_suffix(".bin").stat().st_size != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"Incomplete cache for {key}")
        array = np.memmap(file.with_suffix(".bin"), mode="r", dtype=dtype, shape=shape)
        data[key] = array if len(stream["shape"]) == 3 else np.asarray(array, dtype=np.float32)
        ts = np.fromfile(file.with_suffix(".timestamps"), dtype="<i8")
        if len(ts) != stream["count"]:
            raise ValueError(f"Incomplete timestamps for {key}")
        timestamps[key] = ts.astype(np.float64) / 1e9
    return data, timestamps


def _builder_module():
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.append(str(TRAINING_ROOT))
    from preprocess_data.utils.aloha_frame_builder import AlohaFrameBuilder

    return AlohaFrameBuilder


@quiet_lerobot_writer
def write_lerobot_output(root, frames, indices, config):
    """Write one independent dataset; no global monkey patches to LeRobot."""
    check_runtime()
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.video_utils import encode_video_frames

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Expected LeRobot v2.1, got {CODEBASE_VERSION}")
    video_options = config["writer"]["video"]
    timings = {"png_drain": 0.0, "video": 0.0}

    class ConfiguredDataset(LeRobotDataset):
        def _wait_image_writer(self):
            start = time.monotonic()
            super()._wait_image_writer()
            timings["png_drain"] += time.monotonic() - start

        def encode_episode_videos(self, episode_index):
            start = time.monotonic()
            paths = {}
            for key in self.meta.video_keys:
                path = self.root / self.meta.get_video_file_path(episode_index, key)
                image_dir = self._get_image_file_path(episode_index, key, 0).parent
                for frame_index in range(len(indices)):
                    png = self._get_image_file_path(episode_index, key, frame_index)
                    if not png.is_file() or png.stat().st_size == 0:
                        raise ValueError(f"Incomplete PNG output: {png}")
                encode_video_frames(image_dir, path, self.fps, overwrite=True, **video_options)
                paths[key] = str(path)
            timings["video"] += time.monotonic() - start
            return paths

    features = copy.deepcopy(config["features"])
    # Offline demonstrations are positive action examples; no model or p70 needed.
    if root.name == "action":
        features["advantage"] = {
            "dtype": "float32",
            "shape": [1],
            "names": ["advantage"],
        }
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    dataset = ConfiguredDataset.create(
        repo_id=f"collect/{root.parent.name.removesuffix('_temp')}_{root.name}",
        root=root,
        robot_type=config["robot_type"],
        fps=config["conversion"]["fps"],
        features=features,
        image_writer_processes=config["writer"]["image_writer_processes"],
        image_writer_threads=config["writer"]["image_writer_threads"],
        video_backend="pyav",
    )
    try:
        start = time.monotonic()
        for idx in indices:
            frame = dict(frames[idx])
            if root.name == "action":
                frame["advantage"] = np.ones(1, dtype=np.float32)
            for key, feature in features.items():
                if feature["dtype"] == "video":
                    frame[key] = legacy_jpeg_rgb(frame[key])
            dataset.add_frame(frame)
        timings["add_frame"] = time.monotonic() - start
        start = time.monotonic()
        dataset.save_episode()
        timings["save_episode"] = time.monotonic() - start
    finally:
        dataset.stop_image_writer()
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    tables = list((root / "data").rglob("*.parquet"))
    videos = list((root / "videos").rglob("*.mp4"))
    if info["total_episodes"] != 1 or info["total_frames"] != len(indices):
        raise ValueError("LeRobot metadata count mismatch")
    if len(tables) != 1 or pq.read_metadata(tables[0]).num_rows != len(indices):
        raise ValueError("LeRobot table count mismatch")
    if len(videos) != 6 or any(p.stat().st_size == 0 for p in videos):
        raise ValueError("LeRobot video count mismatch")
    import av

    for key in dataset.meta.video_keys:
        path = root / dataset.meta.get_video_file_path(0, key)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            height, width, _ = features[key]["shape"]
            if (
                stream.frames != len(indices)
                or (stream.height, stream.width) != (height, width)
                or stream.average_rate != config["conversion"]["fps"]
            ):
                raise ValueError(f"LeRobot video frame count/shape/FPS mismatch: {key}")
    return timings


def finalize_episode(directory, *, defer_publication=False):
    """Retry only sealed recordings. Publish both outputs together, never overwrite."""
    path = Path(directory)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not path.name.endswith("_temp") or not manifest.get("normal_end"):
        raise ValueError("Only normally ended temporary episodes can be finalized")
    final = path.with_name(path.name.removesuffix("_temp"))
    if final.exists():
        raise FileExistsError(final)
    config = validate_config(manifest["config"])
    if config.get("publication"):
        # A prior attempt may have committed ready before cleanup or local
        # materialization was interrupted. Recover without re-encoding sources.
        for committed in (Path(config["output_dir"]) / "ready").glob("episode_*/manifest.json"):
            if json.loads(committed.read_text())["source_episode_id"] == manifest["source_episode_id"]:
                if defer_publication:
                    return str(path)
                from utils.lerobot_collect_publish import publish_episode

                result = publish_episode(path, config, manifest)
                shutil.rmtree(path)
                return result
    start = time.monotonic()
    try:
        manifest.update(status="finalizing", error=None)
        atomic_json(manifest_path, manifest)
        data, timestamps = load_cached_streams(path, manifest)
        from utils.lerobot_collect_publish import check_offline_streams

        check_offline_streams(data, timestamps, config)
        builder = _builder_module()(
            window_size=config["conversion"]["window_size"],
            max_time_diff_ms=config["conversion"]["max_time_diff_ms"],
            features=config["features"],
            hardware_latency_s=config["conversion"]["latency_s"],
        )
        frames = builder.build_offline_frames_by_action_index(data, timestamps, config["inputs"], config["task_name"])
        from preprocess_data.utils.sparse_reward_utils import (
            downsampled_action_indices_with_terminal,
        )

        manifest["output_frames"] = {}
        manifest["writer_timings_s"] = {}
        manifest["timings_s"] = {"build": time.monotonic() - start}
        for name, output in config["outputs"].items():
            root = path / name
            if root.exists():
                shutil.rmtree(root)  # Only uncommitted output; the source cache is retained.
            indices = downsampled_action_indices_with_terminal(
                sorted(frames),
                end_idx=len(data[config["inputs"]["leader_joint_pos"]]) - 1,
                frame_stride=output["frame_stride"],
            )
            write_start = time.monotonic()
            manifest["writer_timings_s"][name] = write_lerobot_output(root, frames, indices, config)
            manifest["timings_s"][name] = time.monotonic() - write_start
            manifest["output_frames"][name] = len(indices)
        manifest.update(status="complete", completed_at_ns=time.time_ns())
        manifest["timings_s"]["total"] = time.monotonic() - start
        atomic_json(manifest_path, manifest)
        if config.get("publication"):
            if defer_publication:
                return str(path)
            from utils.lerobot_collect_publish import publish_episode

            ready_path = publish_episode(path, config, manifest)
            logger.info(f"Ready for incremental upload: {ready_path}")
            final = Path(ready_path)
        else:
            if final.exists():
                raise FileExistsError(final)
            path.rename(final)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(manifest_path, manifest)
        raise
    try:
        # Publication retains payload and recording diagnostics in ready. The
        # temporary independent repos are no longer needed after both commits.
        shutil.rmtree(path if config.get("publication") else final / "cache")
    except OSError:
        logger.exception(f"Output complete but cache cleanup failed: {final}")
    logger.info(
        f"✅ Episode saved and validated: {final} | "
        f"action={manifest['output_frames']['action']} frames, "
        f"value={manifest['output_frames']['value']} frames | "
        f"finalization={manifest['timings_s']['total']:.2f}s"
    )
    return str(final)


def _convert_for_publication(directory):
    return finalize_episode(directory, defer_publication=True)


def _publish_converted_episode(conversion):
    """Wait in the serial publisher thread, never in the capture loop."""
    path = Path(conversion.result())
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["config"].get("publication"):
        # Recovery honors the sealed recording config, including legacy output.
        return str(path)
    from utils.lerobot_collect_publish import publish_episode

    try:
        ready = publish_episode(path, manifest["config"], manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(manifest_path, manifest)
        raise
    try:
        shutil.rmtree(path)
    except OSError:
        logger.exception(f"Output complete but cache cleanup failed: {ready}")
    logger.info(f"Ready for incremental upload: {ready}")
    return ready


class RecordingSession:
    """Own a directory, one active cache, and bounded background finalization."""

    def __init__(self, config, *, executor=None):
        self.config = validate_config(config)
        self.root = Path(self.config["output_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".recorder.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"Another recorder/finalizer owns {self.root}") from None
        self.executor = executor
        self.publisher = None
        self.futures = []
        self.errors = []
        self.active = None
        self.next_index = self.config["start_index"]
        self._numbering_initialized = False

    def reap(self):
        pending = []
        for future in self.futures:
            if future.done():
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(str(exc))
                    logger.exception(f"Episode finalization failed; cache retained: {exc}")
            else:
                pending.append(future)
        self.futures = pending

    def can_start(self):
        self.reap()
        return self.active is None and len(self.futures) < self.config["capture"]["max_pending_episodes"]

    def start_episode(self):
        if not self.can_start():
            raise RuntimeError("Finalization backlog is full or an episode is already active")
        if not self._numbering_initialized:
            for ready_manifest in (self.root / "ready").glob("episode_*/manifest.json"):
                published = json.loads(ready_manifest.read_text())
                if published.get("start_index", 0) != self.config["start_index"]:
                    raise ValueError("start_index changed; keep it fixed when resuming a batch")
                source_name = published.get("source_episode_name", ready_manifest.parent.name)
                self.next_index = max(self.next_index, int(source_name.removeprefix("episode_")) + 1)
                self.next_index = max(
                    self.next_index,
                    int(ready_manifest.parent.name.removeprefix("episode_")) + 1,
                )
            self._numbering_initialized = True
        while True:
            name = f"episode_{self.next_index:06d}"
            self.next_index += 1
            if not (self.root / name).exists() and not (self.root / f"{name}_temp").exists():
                break
        self.active = EpisodeCache(self.root / f"{name}_temp", self.config)
        return self.active

    def submit(self, path):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(
                max_workers=self.config["capture"]["conversion_workers"],
                mp_context=multiprocessing.get_context("spawn"),
            )
        if self.config.get("publication") and self.config["capture"]["conversion_workers"] > 1:
            if self.publisher is None:
                self.publisher = ThreadPoolExecutor(max_workers=1)
            conversion = self.executor.submit(_convert_for_publication, str(path))
            # FIFO publication preserves recording order even when encoding finishes
            # out of order. Pending includes converted episodes awaiting publication.
            self.futures.append(self.publisher.submit(_publish_converted_episode, conversion))
        else:
            self.futures.append(self.executor.submit(finalize_episode, str(path)))

    def end_episode(self, *, discard=False, interrupted=False):
        cache = self.active
        if cache is None:
            return
        self.active = None
        cache.close(normal_end=not (discard or interrupted), discard=discard)
        if not discard and cache.manifest["normal_end"]:
            self.submit(cache.path)
            logger.info(f"⏳ Recording finished: {cache.path.name}. Saving in the background...")
        elif cache.error:
            logger.error(f"Episode cache incomplete, not submitted: {cache.path}: {cache.error}")
        elif discard:
            logger.info(f"Episode discarded: {cache.path.name}")
        else:
            logger.warning(f"Recording interrupted: {cache.path.name}; not submitted as a successful episode.")

    def finalize_pending(self):
        for path in sorted(self.root.glob("episode_*_temp")):
            manifest_path = path / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()).get("normal_end"):
                while len(self.futures) >= self.config["capture"]["max_pending_episodes"]:
                    time.sleep(0.05)
                    self.reap()
                self.submit(path)

    def close(self):
        try:
            self.end_episode(interrupted=True)
            if self.executor is not None:
                if self.futures:
                    logger.info("⏳ Waiting for background episode saves before shutdown...")
                self.executor.shutdown(wait=True)
            if self.publisher is not None:
                self.publisher.shutdown(wait=True)
            self.reap()
        finally:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
