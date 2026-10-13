"""Compare one/two conversion workers on identical synthetic full-resolution input.

No devices, SHM, network or upload. Each job has an independent output root;
this measures conversion capacity, NOT concurrent cumulative publication safety.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.lerobot_recording import EpisodeCache, atomic_json, finalize_episode


class ProcessTreeMonitor:
    """Sample CPU deltas and aggregate RSS/PSS; retain CPU from exited children."""

    def __init__(self):
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.cpu = {}
        self.cpu_seconds = 0.0
        self.peak_cpu_cores = 0.0
        self.peak_rss = self.peak_pss = self.peak_threads = self.peak_processes = 0
        self.samples = 0
        self.started = self.last = time.monotonic()
        self.sample(baseline=True)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def sample(self, baseline=False):
        now = time.monotonic()
        elapsed = now - self.last
        delta = 0.0
        rss = pss = threads = processes = 0
        for p in [self.process, *self.process.children(recursive=True)]:
            try:
                identity = (p.pid, p.create_time())
                times = p.cpu_times()
                current = times.user + times.system
                delta += max(0, current - self.cpu.get(identity, current if baseline else 0))
                self.cpu[identity] = current
                memory = p.memory_full_info()
                rss += memory.rss
                pss += getattr(memory, "pss", memory.rss)
                threads += p.num_threads()
                processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not baseline:
            self.cpu_seconds += delta
            self.peak_cpu_cores = max(self.peak_cpu_cores, delta / max(elapsed, 0.001))
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_pss = max(self.peak_pss, pss)
        self.peak_threads = max(self.peak_threads, threads)
        self.peak_processes = max(self.peak_processes, processes)
        self.last = now
        self.samples += 1

    def run(self):
        while not self.stop_event.wait(0.2):
            self.sample()

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        elapsed = time.monotonic() - self.started
        return {
            "wall_s": elapsed,
            "sampled_cpu_seconds": self.cpu_seconds,
            "mean_cpu_cores": self.cpu_seconds / elapsed,
            "peak_sampled_cpu_cores": self.peak_cpu_cores,
            "peak_rss_gib": self.peak_rss / 2**30,
            "peak_pss_gib": self.peak_pss / 2**30,
            "peak_processes": self.peak_processes,
            "peak_threads": self.peak_threads,
            "samples": self.samples,
        }


def build_source(path, config, seconds):
    cache = EpisodeCache(path, config)
    for role, device in config["inputs"].items():
        fps = 30 if role.startswith("img") else 100
        shape_key = {
            "img_ego": "base_image",
            "img_left": "left_wrist_image",
            "img_right": "right_wrist_image",
        }
        if role.startswith("img"):
            height, width, _ = config["features"][shape_key[role]]["shape"]
            y, x = np.indices((height, width))
            base = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
        for index in range(round(seconds * fps)):
            ts = 10 + index / fps
            if role.startswith("img"):
                ts += 0.033
                value = np.roll(base, index % 100, axis=1)
            else:
                value = np.full(14, np.sin(index / 100), dtype=np.float64)
            cache.record(device, round(ts * 1e9), value)
    cache.close(normal_end=True)
    if cache.error or any(cache.dropped.values()):
        raise RuntimeError("Synthetic cache generation lost samples")


def benchmark(seconds, output):
    output.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = OmegaConf.to_container(
            compose(config_name="collect").consumer,
            resolve=True,
        )
    config.pop("publication", None)
    config["output_dir"] = str(output)
    source = output / "source_temp"
    build_source(source, config, seconds)
    reports = []
    for workers in (1, 2):
        paths = []
        for index in range(2):
            job = output / f"workers_{workers}" / f"job_{index}"
            job.mkdir(parents=True)
            cache = job / "episode_000000_temp"
            shutil.copytree(source, cache, copy_function=shutil.copy2)
            paths.append(cache)
        monitor = ProcessTreeMonitor()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(finalize_episode, map(str, paths)))
        finally:
            stats = monitor.close()
        stats.update(
            workers=workers,
            outputs=[json.loads((Path(p) / "manifest.json").read_text())["timings_s"] for p in results],
        )
        reports.append(stats)
        print(json.dumps(stats), flush=True)
    report = {
        "seconds_per_episode": seconds,
        "jobs_per_trial": 2,
        "config": copy.deepcopy(config),
        "results": reports,
    }
    atomic_json(output / "benchmark.json", report)
    # Only generated benchmark sources, never production data.
    shutil.rmtree(source)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error("Use >=3 seconds to satisfy camera min_frames")
    benchmark(args.seconds, args.output_dir)
