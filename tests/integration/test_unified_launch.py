"""Cross-directory configuration and launch contracts without hardware or training."""

import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
import pytest

import run_online
from preprocess_data.utils.release_config import expand_template
from utils.release_config import launch_plan, resolve_release_config

ROOT = Path(__file__).resolve().parents[2]


def local(name):
    with initialize_config_dir(config_dir=str(ROOT / "deployment/configs"), version_base="1.3"):
        return resolve_release_config(
            compose(
                config_name=name,
                overrides=["device_manager.device_manager.params.config_path=/tmp/vam-test-session.yaml"],
            )
        )


@pytest.mark.parametrize("name", ["collect", "rollout", "example"])
def test_templates_only_launch_existing_lerobot_components(name):
    cfg = local(name)
    roles = ("devices", "upload", "collector" if name == "collect" else "policy")
    for component in roles:
        cfg["component"] = component
        _, script = launch_plan(cfg)
        assert (ROOT / "deployment" / script).is_file()
    assert cfg["storage"]["format"] == "lerobot"
    cfg["storage"]["format"] = "unsupported"
    with pytest.raises(ValueError, match="storage format"):
        resolve_release_config(cfg)


def test_paired_example_preserves_training_and_upload_identity():
    deployment = local("example")
    document = expand_template(ROOT / "training/configs/example.yaml")
    job = document["job"]
    assert deployment["run_name"] == job["run_name"] == "example_run"
    assert deployment["remote"]["root"] == job["split_root"]
    assert job["inbox"] == f"{job['split_root']}/lerobot_rollout_data/{job['run_name']}"
    for role, batch in [("action", "64"), ("value", "32")]:
        plan = run_online.build_plan(document, role)
        arguments = plan["command"]
        assert arguments[arguments.index("--seed") + 1] == "142"
        assert arguments[arguments.index("--batch-size") + 1] == batch
        assert plan["environment"]["SPLIT_ROUND_STEPS"] == "1000"
        assert plan["environment"]["NUM_TRAIN_STEPS"] == "8001"
        assert plan["environment"]["RESUME"] == "0"
        assert plan["environment"]["TRAIN_PYTHON"] == str(ROOT / ".venv/bin/python")


def test_private_overlay_keeps_types_and_check_has_no_side_effects(tmp_path):
    config = tmp_path / "local.yaml"
    destination = tmp_path / "unused dataset home"
    config.write_text(
        f"include: {ROOT}/training/configs/example.yaml\njob:\n  resume: true\n  dataset_home: {destination}\n  trainer_overrides:\n    common:\n      wandb_enabled: false\n"
    )
    document = expand_template(config)
    plan = run_online.build_plan(document, "action")
    assert "--no-wandb-enabled" in plan["command"]
    assert plan["environment"]["RESUME"] == "1"
    for role in ("ingest", "action", "value"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "training/run_online.py"), str(config), "--component", role, "--check"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert json.loads(result.stdout)["environment"]["RESUME"] == "1"
    assert not destination.exists()


@pytest.mark.parametrize("value", [0, -1, True, "100"])
def test_invalid_training_budget_is_rejected(value):
    document = expand_template(ROOT / "training/configs/example.yaml")
    document["job"]["round_steps"] = value
    with pytest.raises(ValueError, match="round_steps"):
        run_online.build_plan(document)


def test_include_and_variable_cycles_are_explicit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("include: config.yaml\n")
    with pytest.raises(ValueError, match="Cyclic template"):
        expand_template(path)
    path.write_text("template_version: 1\nvariables:\n  x: ${y}\n  y: ${x}\njob:\n  name: ${x}\n")
    with pytest.raises(ValueError, match="Cyclic variable"):
        expand_template(path)


def test_offline_link_restart_and_source_conflict(tmp_path):
    job = {
        "dataset_home": str(tmp_path / "run"),
        "offline_dataset_home": str(tmp_path / "offline"),
        "action_offline_repo": "example/action",
        "value_offline_repo": "example/value",
    }
    for role in ("action", "value"):
        source = Path(job["offline_dataset_home"]) / job[f"{role}_offline_repo"]
        source.mkdir(parents=True)
        (source / "identity").write_text(role)
    run_online.link_offline_repos(job)
    run_online.link_offline_repos(job)
    assert (Path(job["dataset_home"]) / "example/action/identity").read_text() == "action"
    target = Path(job["dataset_home"]) / "example/action"
    target.unlink()
    target.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="another source"):
        run_online.link_offline_repos(job)
    job["action_offline_repo"] = "../escape"
    with pytest.raises(ValueError, match="repo ID"):
        run_online.link_offline_repos(job)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("role", ["action", "value"])
def test_real_split_shell_preserves_mode_and_python_arguments(tmp_path, role, resume):
    """Execute the actual shell launcher, replacing only the training interpreter."""
    document = expand_template(ROOT / "training/configs/example.yaml")
    job = document["job"]
    for field in ("cloud_root", "dataset_home", "norm_assets_dir", "base_params", "log_dir"):
        job[field] = str(tmp_path / field)
        Path(job[field]).mkdir()
    Path(job["cloud_root"], "dataset_views").mkdir()
    for name in ("action", "value"):
        Path(job["cloud_root"], "dataset_views", f"{name}_latest.json").write_text("{}")
    Path(job["dataset_home"], "example/offline_action").mkdir(parents=True)
    job["norm_stats_path"] = str(tmp_path / "norm.json")
    Path(job["norm_stats_path"]).write_text("{}")
    job["resume"] = resume
    fake_python = tmp_path / "fake python"
    output = tmp_path / "arguments.json"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    document["runtime"]["python"]["training"] = str(fake_python)
    plan = run_online.build_plan(document, role)
    env = {
        **os.environ,
        "RESUME": "1" if not resume else "0",
        **plan["environment"],
        "ALOHA_ASYNC_OFFLINE_REPO_ID": "example/offline_action",
    }
    subprocess.run(plan["command"], env=env, check=True, capture_output=True, text=True, timeout=20)
    arguments = json.loads(output.read_text())
    assert arguments[:2] == ["scripts/train_async.py", f"pi05_aloha_async_recap_split_{role}"]
    assert ("--resume" in arguments) == resume
    assert ("--overwrite" in arguments) != resume
    assert arguments[arguments.index("--seed") + 1] == "142"
