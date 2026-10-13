#!/usr/bin/env python3
"""Online DAgger/Learn from Experience: launch LeRobot ingestion or a split trainer from the job YAML."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from preprocess_data.utils.release_config import expand_template

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def _python(path):
    path = Path(path).expanduser()
    return os.path.abspath(path if path.is_absolute() else REPO_ROOT / path)


_TRAINER_OPTIONS = {
    "seed": (int, 0),
    "batch_size": (int, 1),
    "fsdp_devices": (int, 1),
    "num_workers": (int, 0),
    "prefetch_size": (int, 0),
    "prefetch_factor": (int, 1),
    "log_interval": (int, 1),
    "device_prefetch": (bool, None),
    "wandb_enabled": (bool, None),
}


def trainer_arguments(job, role):
    """Translate role runtime settings into the existing training CLI, without shell parsing."""
    overrides = job.get("trainer_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {"common", "action", "value"}:
        raise ValueError("job.trainer_overrides must contain common/action/value mappings")
    for group in overrides.values():
        if not isinstance(group, dict) or set(group) - _TRAINER_OPTIONS.keys():
            raise ValueError(f"Supported trainer_overrides fields: {sorted(_TRAINER_OPTIONS)}")
        for name, value in group.items():
            kind, minimum = _TRAINER_OPTIONS[name]
            if type(value) is not kind or (minimum is not None and value < minimum):
                raise ValueError(f"Invalid trainer_overrides.{name}: {value!r}")
    options = {**overrides.get("common", {}), **overrides.get(role, {})}
    arguments = []
    for name, value in options.items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(f"--{'' if value else 'no-'}{flag}")
        else:
            arguments.extend([f"--{flag}", str(value)])
    return arguments


def build_plan(document, component="ingest"):
    """Validate a plan without importing models, reading datasets or spawning children."""
    if component not in {"ingest", "action", "value"}:
        raise ValueError(f"Unknown component: {component}")
    job = document["job"]
    runtime = document["runtime"]["python"]
    for key in ("round_steps", "num_train_steps"):
        if type(job[key]) is not int or job[key] < 1:
            raise ValueError(f"job.{key} must be positive")
    if type(job["resume"]) is not bool:
        raise ValueError("job.resume must be a boolean")
    overrides = {role: trainer_arguments(job, role) for role in ("action", "value")}
    fields = {
        "RUN_NAME": "run_name",
        "SPLIT_ROOT": "split_root",
        "CLOUD_ROOT": "cloud_root",
        "DATA_HF_LEROBOT_HOME": "dataset_home",
        "OFFLINE_HF_LEROBOT_HOME": "offline_dataset_home",
        "HF_HOME": "hf_home",
        "WORK_DIR": "work_dir",
        "LOG_DIR": "log_dir",
        "LIVE_REPO_PREFIX": "repo_prefix",
        "ACTION_OFFLINE_REPO": "action_offline_repo",
        "VALUE_OFFLINE_REPO": "value_offline_repo",
        "CHECKPOINT_POLL_INTERVAL_S": "checkpoint_poll_interval_s",
        "ALOHA_ASYNC_BASE_PARAMS": "base_params",
        "ALOHA_ASYNC_NORM_ASSETS_DIR": "norm_assets_dir",
        "SPLIT_ROUND_STEPS": "round_steps",
        "NUM_TRAIN_STEPS": "num_train_steps",
        "XLA_CLIENT_MEM_FRACTION": "xla_memory_fraction",
    }
    env = {name: str(job[field]) for name, field in fields.items()}
    env.update(
        HF_LEROBOT_HOME=str(job["dataset_home"]),
        RESUME="1" if job["resume"] else "0",
        TRAIN_PYTHON=_python(runtime["training"]),
        XLA_PYTHON_CLIENT_PREALLOCATE=str(job["xla_preallocate"]).lower(),
    )
    norm = job.get("norm_stats_path")
    if norm is not None:
        if not isinstance(norm, str) or not norm.strip():
            raise ValueError("job.norm_stats_path must be a nonempty path or null")
        env["ALOHA_ASYNC_NORM_STATS_PATH"] = norm
    if component == "ingest":
        command = [
            _python(runtime["ingest"]),
            str(ROOT / "preprocess_data/register_lerobot_uploads.py"),
            "--inbox",
            job["inbox"],
            "--dataset-home",
            job["dataset_home"],
            "--cloud-root",
            job["cloud_root"],
            "--repo-prefix",
            job["repo_prefix"],
            "--action-offline-repo",
            job["action_offline_repo"],
            "--value-offline-repo",
            job["value_offline_repo"],
            "--poll-interval",
            str(job["poll_interval_s"]),
        ]
        if job["run_once"]:
            command.append("--once")
    else:
        env.update(ROLE=component, CUDA_VISIBLE_DEVICES=str(job[f"{component}_devices"]))
        command = ["bash", str(ROOT / "scripts/run_async_recap_split_train.sh"), *overrides[component]]
    return {"command": command, "environment": env}


def link_offline_repos(job):
    """Expose read-only offline inputs without replacing existing run data."""
    destination = Path(job["dataset_home"])
    source = Path(job["offline_dataset_home"])
    for role in ("action", "value"):
        repo_id = Path(job[f"{role}_offline_repo"])
        if repo_id.is_absolute() or ".." in repo_id.parts:
            raise ValueError(f"Invalid offline repo ID: {repo_id}")
        src, dst = source / repo_id, destination / repo_id
        if not src.is_dir():
            raise FileNotFoundError(src)
        if dst.is_symlink() and dst.resolve() != src.resolve():
            raise ValueError(f"Offline repo link has another source: {dst}")
        if dst.exists():
            if not dst.is_dir():
                raise ValueError(f"Offline repo target is not a directory: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve(), target_is_directory=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--component", choices=("ingest", "action", "value"), default="ingest")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    document = expand_template(args.config)
    plan = build_plan(document, args.component)
    if args.check:
        print(json.dumps(plan, indent=2))
        return
    if args.component == "ingest":
        link_offline_repos(document["job"])
    env = dict(os.environ, **plan["environment"])
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), str(REPO_ROOT / "packages/vam-client/src"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(plan["command"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
