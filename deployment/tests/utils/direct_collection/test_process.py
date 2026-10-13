"""Full saver subprocess -> eight real SHMs -> background writer -> READY."""

import json
import os
import subprocess
import sys
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path

from utils.lerobot_direct import validate_direct_episode
from utils.lerobot_online.protocol import atomic_json, read_json
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    pack_buffer_header,
    pack_device_header,
    pack_frame_header,
)


def test_process_records_failure_from_real_shm(config, policy, samples, tmp_path):
    segments = []
    process = None
    log_path = tmp_path / "saver.log"
    try:
        for logical, data in samples[0].items():
            capacity = 128
            frame_size = FRAME_HEADER_SIZE + data[0].nbytes
            shm = shared_memory.SharedMemory(
                create=True,
                name=f"lr_test_{uuid.uuid4().hex}",
                size=DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + capacity * frame_size,
            )
            segments.append(shm)
            config["inputs"][logical] = shm.name
            shm.buf[:DEVICE_HEADER_SIZE] = pack_device_header(
                "test", 0, 100, str(data.dtype), data.shape[1:], frame_size
            )
            for i, (ts, array) in enumerate(zip(samples[1][logical], data, strict=True)):
                offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * frame_size
                shm.buf[offset : offset + FRAME_HEADER_SIZE] = pack_frame_header(int(ts))
                shm.buf[offset + FRAME_HEADER_SIZE : offset + frame_size] = array.tobytes()
            shm.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = pack_buffer_header(
                capacity, len(data), len(data) - (logical == "control_flag")
            )
        config_file = tmp_path / "saver.json"
        atomic_json(config_file, config)
        root = Path(__file__).resolve().parents[3]
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "consumer/async_infer_data_saver_lerobot.py"),
                    "--config",
                    str(config_file),
                ],
                cwd=root,
                stdout=log,
                stderr=log,
                env={**os.environ, "PYTHONPATH": str(root)},
            )

            def wait_for(predicate):
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    if predicate():
                        return
                    time.sleep(0.02)
                raise AssertionError(log_path.read_text())

            signals = config["signal_files"]
            wait_for(lambda: Path(signals["rollout_ready"]).exists())
            atomic_json(
                signals["start_recording"],
                {"request_id": "start", "policy": policy, "start_ns": samples[2]},
            )
            wait_for(lambda: Path(signals["start_recording"] + ".ack").exists())
            assert read_json(signals["start_recording"] + ".ack")["state"] == "recording"
            atomic_json(
                signals["stop_recording"],
                {"request_id": "stop", "result": "failure", "stop_ns": samples[3]},
            )
            wait_for(lambda: Path(signals["stop_recording"] + ".ack").exists())
            assert read_json(signals["stop_recording"] + ".ack")["state"] == "sealed"
            output = Path(config["output_dir"])
            wait_for(lambda: bool(list(output.glob("ready/*/manifest.json"))))
            bundle = next(output.glob("ready/*/manifest.json")).parent
            manifest = validate_direct_episode(bundle, decode=True)
            assert manifest["result"] == "failure"
            import pyarrow.parquet as pq

            action = pq.read_table(next((bundle / "action/data").rglob("*.parquet")))
            assert all(value == 0 for value in action["success"].to_pylist())
            assert all(sum(value) == 0 for value in action["reward"].to_pylist())
            process.terminate()
            assert process.wait(timeout=15) == 0, log_path.read_text()
            assert not Path(signals["rollout_ready"]).exists()
            assert not list(output.rglob("*.mcap"))
            assert json.loads((bundle / "manifest.json").read_text())["advantage_status"] == "cloud_p70_pending"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for shm in segments:
            shm.close()
            shm.unlink()
