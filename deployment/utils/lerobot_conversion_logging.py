"""Keep third-party writer output in per-episode logs in conversion workers."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from functools import wraps
from pathlib import Path

_output_lock = threading.RLock()


def _flush_output():
    sys.stdout.flush()
    sys.stderr.flush()
    # Native encoders can bypass Python streams, including their buffers.
    ctypes.CDLL(None).fflush(None)


@contextmanager
def conversion_output(log_path):
    """Capture Python and native stdout/stderr; restore both even on failure.

    File descriptor redirection is process-wide. Use only in the dedicated
    conversion worker, never around inference or SHM acquisition in the parent.
    Logs live outside dataset directories so they cannot change bundle hashes.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock, log_path.open("a", buffering=1) as output:
        _flush_output()
        saved = {}
        try:
            for fd in (1, 2):
                saved[fd] = os.dup(fd)
                os.dup2(output.fileno(), fd)
            with redirect_stdout(output), redirect_stderr(output):
                print(f"\n--- Conversion writer started {datetime.now().isoformat()} ---")
                try:
                    yield
                except BaseException:
                    traceback.print_exc()
                    raise
                finally:
                    _flush_output()
        finally:
            for fd, original in saved.items():
                os.dup2(original, fd)
                os.close(original)


def quiet_lerobot_writer(function):
    """Keep writer diagnostics while leaving success/error lifecycle logs visible."""

    @wraps(function)
    def wrapped(root, *args, **kwargs):
        root = Path(root)
        log_path = (Path("logs/lerobot_conversion") / f"{root.parent.name}_{root.name}.log").resolve()
        try:
            with conversion_output(log_path):
                return function(root, *args, **kwargs)
        except Exception as exc:
            # Preserve the original exception type and traceback for the saver.
            exc.add_note(f"LeRobot conversion details: {log_path}")
            from utils.logger_config import logger

            logger.error(f"LeRobot writer failed for {root}; details: {log_path}")
            raise

    return wrapped
