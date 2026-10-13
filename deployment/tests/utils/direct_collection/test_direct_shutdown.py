"""Real process-group interrupts while multiple sealed recordings await conversion."""

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from textwrap import dedent

import pytest

from utils.lerobot_direct import DirectSession, validate_direct_episode
from utils.lerobot_online.protocol import atomic_json, read_json


class DeferredExecutor:
    def __init__(self):
        self.futures = []

    def submit(self, *args):
        future = Future()
        self.futures.append(future)
        return future

    def shutdown(self, wait=True):
        pass


@pytest.mark.parametrize("fail_episode", [False, True])
def test_interrupt_drains_recovered_queue_and_preserves_unfinished(config, policy, samples, tmp_path, fail_episode):
    config["progress_dir"] = str(tmp_path / "progress")
    config["collection_session_id"] = "test-session"
    config["capture"]["max_pending_episodes"] = 4
    executor = DeferredExecutor()
    session = DirectSession(config, executor=executor)
    data, times, start, end = samples
    for index in range(40, 43):
        assert session.start(policy, start, index)
        for name in data:
            for timestamp, array in zip(times[name], data[name], strict=True):
                assert session.active.record(config["inputs"][name], timestamp, array)
        session.stop("success", end)
    assert session.start(policy, start, 43)
    unfinished = session.active.path
    # Leave completed caches on disk to exercise recovery beyond the pool limit.
    for future in executor.futures:
        future.set_result(None)
    session.close()
    config["capture"]["max_pending_episodes"] = 2
    config_file = tmp_path / "config.json"
    atomic_json(config_file, config)
    gate = tmp_path / "release-writer"
    harness = tmp_path / "saver_harness.py"
    harness.write_text(
        dedent("""
        import os
        import time
        from pathlib import Path
        from utils import lerobot_direct as direct
        from consumer import async_infer_data_saver_lerobot as saver

        original = direct.finalize_direct_episode
        def gated_finalize(path, **kwargs):
            while not Path(os.environ["TEST_WRITER_GATE"]).exists():
                time.sleep(0.02)
            if os.environ["TEST_FAIL_EPISODE"] == "1" and direct.read_json(Path(path) / "manifest.json")["rollout_index"] == 41:
                raise ValueError("deliberate conversion failure")
            return original(path, **kwargs)
        direct.finalize_direct_episode = gated_finalize
        # Recovery only: sealed caches no longer depend on any live SHM streams.
        def connect_without_hardware(self):
            self.signals["rollout_ready"].write_text("ready")
        saver.DirectLeRobotSaver.connect = connect_without_hardware
        if __name__ == "__main__":
            saver.main()
    """)
    )
    root = Path(__file__).resolve().parents[3]
    process = None
    log_path = tmp_path / "shutdown.log"
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(harness), "--config", str(config_file)],
                cwd=root,
                stdout=log,
                stderr=log,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(root),
                    "TEST_WRITER_GATE": str(gate),
                    "TEST_FAIL_EPISODE": str(int(fail_episode)),
                },
            )

            def wait_for_status(name, predicate):
                path = Path(config["progress_dir"]) / name
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    if path.exists() and predicate(read_json(path)):
                        return
                    time.sleep(0.02)
                raise AssertionError(log_path.read_text())

            wait_for_status("worker.json", lambda s: s.get("state") == "processing")
            # Same delivery as terminal Ctrl+C: recorder AND writer process group.
            os.killpg(process.pid, signal.SIGINT)
            wait_for_status("session.json", lambda s: s.get("phase") == "draining")
            os.killpg(process.pid, signal.SIGINT)
            time.sleep(0.1)
            assert process.poll() is None
            assert not list(Path(config["output_dir"]).glob("ready/*"))
            gate.touch()
            assert process.wait(timeout=45) == int(fail_episode), log_path.read_text()
        ready = list(Path(config["output_dir"]).glob("ready/*"))
        assert len(ready) == 3 - int(fail_episode)
        manifests = [validate_direct_episode(path, decode=True) for path in ready]
        assert sorted(m["rollout_index"] for m in manifests) == ([40, 42] if fail_episode else [40, 41, 42])
        assert all(m["collection_session_id"] == "test-session" for m in manifests)
        assert read_json(unfinished / "manifest.json")["normal_end"] is False
        status = read_json(Path(config["progress_dir"]) / "session.json")
        assert status["phase"] == ("failed" if fail_episode else "finished")
        assert status["completed"] == len(ready) and status["pending"] == 0
        assert len(status["errors"]) == int(fail_episode)
        if fail_episode:
            caches = list(Path(config["output_dir"]).glob("capture/*/manifest.json"))
            assert any(read_json(p).get("rollout_index") == 41 for p in caches)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
