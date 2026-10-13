"""Resolved release configuration and launch planning; no device/model imports."""

import copy
import os
from pathlib import Path

from omegaconf import OmegaConf

SCRIPTS = {
    "devices": "device_starter.py",
    "collector": "consumer/collect.py",
    "policy": "consumer/vam_policy_infer.py",
    "upload": "consumer/data_upload_manager.py",
}


def resolve_release_config(config):
    result = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else copy.deepcopy(config)
    mode = result["mode"]
    storage = result["storage"]
    if mode not in {"collect", "rollout"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if storage["format"] != "lerobot":
        raise ValueError(f"Unsupported storage format: {storage['format']}")
    expected_payload = "lerobot_collect_v1" if mode == "collect" else "lerobot_direct_v1"
    upload = result["augmental_data_upload"]
    if upload["payload_format"] != expected_payload:
        raise ValueError("Upload payload must follow storage.format")
    suffix = "/ready"
    if upload["data_dir"] != result["recording_output_dir"] + suffix:
        raise ValueError("Upload directory must follow the selected recorder")
    if mode == "rollout":
        rollout = result["policy"]["infer_rollout"]
        if rollout["format"] != storage["format"]:
            raise ValueError("Rollout format must follow storage.format")
        if rollout["python_path"] != storage["python"]:
            raise ValueError("Rollout Python must follow the selected storage environment")
        if rollout["output_dir"] != result["recording_output_dir"]:
            raise ValueError("Rollout output must match the upload source")
        if result["online_training"]["enabled"] and not upload["enabled"]:
            raise ValueError("Online training requires the upload/checkpoint manager")
    elif result["online_training"]["enabled"]:
        raise ValueError("Online training belongs to the rollout config")
    return result


def launch_plan(config):
    component = config["component"]
    if component not in SCRIPTS:
        raise ValueError(f"Unknown component: {component}")
    if component == "collector" and config["mode"] != "collect":
        raise ValueError("Use the collect config for the collector")
    if component == "policy" and config["mode"] != "rollout":
        raise ValueError("Use the rollout config for policy inference")
    runtime_key = "hardware" if component == "devices" else component
    python = config["storage"]["python"] if component == "collector" else config["runtime"]["python"][runtime_key]
    # Keep venv symlinks: the interpreter uses its invoked path to locate packages.
    return os.path.abspath(Path(python).expanduser()), SCRIPTS[component]


def write_resolved_session(config, directory):
    """Write transient subprocess inputs; the operator maintains only the main config."""
    config = copy.deepcopy(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    anomaly_path = directory / "anomaly.yaml"
    OmegaConf.save(OmegaConf.create(config["anomaly_config"]), anomaly_path)
    config["_anomaly_config_path"] = str(anomaly_path)
    if config["mode"] == "rollout":
        config["policy"]["infer_rollout"]["anomaly_config_path"] = str(anomaly_path)
    path = directory / "session.yaml"
    config["_resolved_config_path"] = str(path)
    OmegaConf.save(OmegaConf.create(config), path)
    return path
