"""Compute normalization statistics for the async / value-learning workflow.

Thin wrapper over ``scripts.compute_norm_stats`` that pins ``action_horizon=1`` (single-frame
low-dim stats) and lets next-observation keys fall out of ``data_config.use_next_obs``. This is
the preferred entry point for value-learning configs: single-frame stats avoid the slow
image-processing path of full action-chunk stats.

Equivalent to::

    uv run scripts/compute_norm_stats.py <config_name> --action-horizon 1
"""

from pathlib import Path
import sys

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tyro

from scripts.compute_norm_stats import main as _main


def main(config_name: str, max_frames: int | None = None):
    _main(config_name, max_frames=max_frames, action_horizon=1)


if __name__ == "__main__":
    tyro.cli(main)
