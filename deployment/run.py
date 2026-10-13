#!/usr/bin/env python3
"""Launch data collection, offline DAgger warmup data collection, or Online DAgger/Learn from Experience."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

from utils.release_config import (
    launch_plan,
    resolve_release_config,
    write_resolved_session,
)

ROOT = Path(__file__).resolve().parent


def child_environment():
    env = dict(os.environ)
    sources = [
        ROOT,
        ROOT.parent / "training",
        ROOT.parent / "packages/i2rt",
        ROOT.parent / "training/src",
        ROOT.parent / "packages/vam-client/src",
    ]
    env["PYTHONPATH"] = os.pathsep.join([*(str(p) for p in sources), env.get("PYTHONPATH", "")])
    return env


def preflight(config, python):
    paths = {python, config["storage"]["python"]}
    if config["mode"] == "rollout":
        paths.add(config["runtime"]["python"]["policy"])
    for executable in paths:
        if not Path(executable).expanduser().is_file():
            raise FileNotFoundError(f"Configure runtime.python before starting: {executable}")
    modules = ["lerobot", "pyarrow"]
    check = (
        "import importlib.util,sys; "
        "missing=[n for n in sys.argv[1:] if importlib.util.find_spec(n) is None]; "
        "assert not missing, 'Missing storage dependencies: '+', '.join(missing)"
    )
    storage_python = os.path.abspath(Path(config["storage"]["python"]).expanduser())
    subprocess.run(
        [storage_python, "-c", check, *modules],
        check=True,
        cwd=ROOT,
        env=child_environment(),
    )
    if config["mode"] == "rollout":
        model_python = os.path.abspath(Path(config["runtime"]["python"]["policy"]).expanduser())
        subprocess.run(
            [model_python, "-c", check, "vam", "jax"],
            check=True,
            cwd=ROOT,
            env=child_environment(),
        )
    if config["component"] == "devices" and any(
        str(config["camera"][key]).startswith("SET_") for key in ("ego_serial", "left_serial", "right_serial")
    ):
        raise ValueError("Set camera serials in the main config before starting devices")
    if config["component"] == "upload" and not config["augmental_data_upload"]["enabled"]:
        raise ValueError("Set upload.enabled=true to run the upload manager")


@hydra.main(config_path="configs", config_name="collect", version_base="1.3")
def main(cfg: DictConfig):
    config = resolve_release_config(cfg)
    python, script = launch_plan(config)
    if config["check_only"]:
        print(
            json.dumps(
                {
                    "mode": config["mode"],
                    "format": config["storage"]["format"],
                    "python": python,
                    "script": script,
                }
            )
        )
        return
    preflight(config, python)
    with tempfile.TemporaryDirectory(prefix="vam-session-") as directory:
        path = write_resolved_session(config, directory)
        args = (
            ["--config_file", str(path)]
            if config["component"] == "upload"
            else ["--config-path", directory, "--config-name", "session"]
        )
        result = subprocess.run(
            [python, str(ROOT / script), *args],
            cwd=ROOT,
            check=False,
            env=child_environment(),
        )
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
