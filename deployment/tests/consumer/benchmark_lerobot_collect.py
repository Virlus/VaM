"""Synthetic latest-frame SHM benchmark; never connects devices or sends signals.

Run with the recording environment, e.g.:
python tests/consumer/benchmark_lerobot_collect.py --seconds 30 --output-dir /tmp/collect-benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from benchmark_lerobot_conversion import ProcessTreeMonitor
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from consumer.lerobot_data_saver import LeRobotCollectConsumer
from utils.lerobot_recording import RecordingSession, atomic_json, check_runtime


def benchmark(seconds, output_dir, codec=None):
    check_runtime()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = OmegaConf.to_container(
            compose(config_name="collect").consumer,
            resolve=True,
        )
    config["output_dir"] = str(output_dir)
    if codec is not None:
        config["writer"]["video"]["vcodec"] = codec
    session = RecordingSession(config)
    # Bypass BaseConsumer initialization: no SHM connection or signal handlers.
    collector = LeRobotCollectConsumer.__new__(LeRobotCollectConsumer)
    collector.summary_shm = None
    collector.config, collector.session = config, session
    collector.poll_stats = {"cycles": 0, "max_cycle_s": 0.0, "overruns": 0}
    process = psutil.Process()
    report = {"seconds_per_episode": seconds, "episodes": [], "peak_rss_bytes": 0}
    monitor = ProcessTreeMonitor()
    try:
        for episode in range(2):
            cache = session.start_episode()
            collector.poll_stats = {"cycles": 0, "max_cycle_s": 0.0, "overruns": 0}
            camera_images = {}
            for name, role in zip(
                ("base_image", "left_wrist_image", "right_wrist_image"),
                ("img_ego", "img_left", "img_right"),
            ):
                shape = config["features"][name]["shape"]
                y, x = np.indices(shape[:2])
                camera_images[role] = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
            latest = {}
            start = time.monotonic()
            cycles = []
            peak_queue = 0
            origin = time.time_ns()
            roles = {device: role for role, device in config["inputs"].items()}

            def read_latest(
                device,
                *,
                roles=roles,
                start=start,
                latest=latest,
                origin=origin,
                camera_images=camera_images,
            ):
                role = roles[device]
                fps = 30 if role.startswith("img") else 100
                index = int((time.monotonic() - start) * fps)
                if device not in latest or latest[device][0] != index:
                    timestamp = origin + round(index / fps * 1e9)
                    if role.startswith("img"):
                        timestamp += 33_000_000
                        value = np.roll(camera_images[role], index % 100, axis=1)
                    else:
                        value = np.full(14, np.sin(index / 100), dtype=np.float64)
                    latest[device] = (index, timestamp, value)
                _, timestamp, value = latest[device]
                # BaseConsumer returns a copy under its SHM lock, even for duplicates.
                return timestamp, value.copy()

            collector._read_device_data = read_latest
            while time.monotonic() - start < seconds:
                begin = time.monotonic()
                collector.collect_once()
                cycles.append(time.monotonic() - begin)
                peak_queue = max(peak_queue, cache.queue.qsize())
                if len(cycles) % 20 == 0:
                    rss = process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    report["peak_rss_bytes"] = max(report["peak_rss_bytes"], rss)
                time.sleep(
                    max(
                        0,
                        config["capture"]["poll_interval"] - (time.monotonic() - begin),
                    )
                )
            sealing = time.monotonic()
            cache.manifest["polling"] = collector.poll_stats
            session.end_episode()
            entry = {
                "episode": episode,
                "source_episode_id": cache.manifest["source_episode_id"],
                "cycles": len(cycles),
                "poll_p95_ms": float(np.percentile(cycles, 95) * 1000),
                "poll_max_ms": max(cycles) * 1000,
                "peak_queue": peak_queue,
                "accepted": cache.accepted,
                "dropped": cache.dropped,
                "not_observed_by_latest_poll": {
                    device: latest[device][0] + 1 - cache.accepted[device] - cache.dropped[device] for device in latest
                },
                "seal_s": time.monotonic() - sealing,
            }
            report["episodes"].append(entry)
    finally:
        try:
            session.close()
        finally:
            report["process_tree_resources"] = monitor.close()
    report["errors"] = session.errors
    report["outputs"] = []
    for path in sorted((session.root / "ready").glob("episode_*/manifest.json")):
        if path.parent.name.endswith("_temp"):
            continue
        manifest = json.loads(path.read_text())["recording"]
        report["outputs"].append(
            {
                "path": str(path.parent),
                "frames": manifest["output_frames"],
                "timings_s": manifest["timings_s"],
                "writer_timings_s": manifest["writer_timings_s"],
            }
        )
    atomic_json(session.root / "benchmark.json", report)
    print(json.dumps(report, indent=2))
    if session.errors:
        raise RuntimeError("Benchmark finalization failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=["h264", "hevc", "libsvtav1"])
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    benchmark(args.seconds, args.output_dir, args.codec)
