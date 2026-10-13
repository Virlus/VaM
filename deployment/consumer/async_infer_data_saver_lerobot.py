"""Direct LeRobot inference saver using the existing start/stop/result signal paths."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lerobot_direct import DirectSession, validate_config
from utils.lerobot_online.protocol import atomic_json, read_json
from utils.lerobot_shm import RingStream
from utils.logger_config import logger


class DirectLeRobotSaver:
    def __init__(self, config, *, session=None, readers=None):
        self.config = validate_config(config)
        self.signals = {
            key: Path(value)
            for key, value in config["signal_files"].items()
            if key in {"start_recording", "stop_recording", "episode_result", "rollout_ready"}
        }
        self.session = session or DirectSession(config)
        self.readers = readers or {}
        self.running = True
        self.pending_stop = None
        self.completed_requests = {}

    def connect(self):
        deadline = time.monotonic() + self.config["capture"].get("connect_timeout_s", 30)
        while self.running:
            for logical, stream in self.config["inputs"].items():
                if stream not in self.readers:
                    try:
                        self.readers[stream] = RingStream(stream, index_is_latest=logical == "control_flag")
                    except FileNotFoundError:
                        pass
            if len(self.readers) == len(self.config["inputs"]):
                self.signals["rollout_ready"].write_text("ready")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Required SHM missing: {set(self.config['inputs'].values()) - self.readers.keys()}")
            time.sleep(0.1)

    def _ack(self, name, request, state, error=None):
        ack = {"request_id": request["request_id"], "state": state, "error": error}
        if state == "recording" and self.session.active is not None:
            ack["rollout_index"] = self.session.active.manifest["rollout_index"]
            ack["episode_name"] = self.session.active.manifest["episode_name"]
        atomic_json(str(self.signals[name]) + ".ack", ack)
        self.completed_requests[request["request_id"]] = ack
        self.signals[name].unlink(missing_ok=True)

    def collect(self):
        cache = self.session.active
        if cache is None:
            return
        end = self.pending_stop["stop_ns"] if self.pending_stop else None
        latencies = {
            stream: int(self.config.get("latency_ms", {}).get(name, 0) * 1e6)
            for name, stream in self.config["inputs"].items()
        }
        for stream, reader in self.readers.items():
            for timestamp, array in reader.read_new():
                if (end is None or timestamp - latencies[stream] <= end) and not cache.record(stream, timestamp, array):
                    # Reader removes duplicates; false means a real sample was lost.
                    raise RuntimeError(f"Raw cache rejected {stream} at {timestamp}")

    def tick(self):
        self.session.reap()
        for name in ("start_recording", "stop_recording"):
            path = self.signals[name]
            if not path.exists():
                continue
            request = read_json(path)
            previous = self.completed_requests.get(request["request_id"])
            if previous is not None:
                atomic_json(str(path) + ".ack", previous)
                path.unlink(missing_ok=True)
                continue
            if name == "start_recording":
                if self.session.start(request["policy"], request["start_ns"], request.get("rollout_index")):
                    lower = request["start_ns"] - int(self.config["capture"].get("preroll_s", 0) * 1e9)
                    for reader in self.readers.values():
                        reader.reset(lower)
                    self.collect()
                    self._ack(name, request, "recording")
            elif self.pending_stop is None:
                self.pending_stop = request
                self.stop_deadline = time.monotonic() + self.config["capture"].get("value_drain_s", 0)
        self.collect()
        if self.pending_stop is not None and time.monotonic() >= self.stop_deadline:
            request, self.pending_stop = self.pending_stop, None
            self.session.stop(request["result"], request["stop_ns"])
            self._ack(
                "stop_recording",
                request,
                ("sealed" if request["result"] in {"success", "failure"} else "discarded"),
            )

    def run(self):
        try:
            self.connect()
            while self.running:
                try:
                    self.tick()
                except Exception as exc:
                    logger.exception("Direct LeRobot recorder failed; keeping unfinished source")
                    for name in ("start_recording", "stop_recording"):
                        if self.signals[name].exists():
                            self._ack(name, read_json(self.signals[name]), "error", str(exc))
                    raise
                time.sleep(self.config["capture"].get("poll_interval", 0.002))
        finally:
            self.signals["rollout_ready"].unlink(missing_ok=True)
            try:
                # A stop request written before Ctrl+C may not have been consumed
                # yet. Honor its explicit result; never invent success for an
                # unmarked active episode.
                pending = self.pending_stop
                stop_path = self.signals["stop_recording"]
                if pending is None and stop_path.exists():
                    pending = read_json(stop_path)
                if pending is not None and self.session.active is not None:
                    self.pending_stop = pending
                    self.collect()
                    self.session.stop(pending["result"], pending["stop_ns"])
                    self._ack(
                        "stop_recording",
                        pending,
                        ("sealed" if pending["result"] in {"success", "failure"} else "discarded"),
                    )
            finally:
                try:
                    for reader in self.readers.values():
                        reader.close()
                finally:
                    # Sealed caches are independent of SHM from this point on.
                    logger.info("LeRobot acquisition stopped; draining all sealed episodes")
                    self.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved JSON from PolicyConnector")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    if args.check_config:
        return
    saver = DirectLeRobotSaver(config)

    def stop(signum, frame):
        saver.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    saver.run()
    if saver.session.errors:
        raise RuntimeError(f"LeRobot conversion failed; source caches retained: {saver.session.errors}")


if __name__ == "__main__":
    main()
