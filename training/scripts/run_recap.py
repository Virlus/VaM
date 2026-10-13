import argparse
from collections.abc import Mapping
import dataclasses
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flax.traverse_util as traverse_util
import wandb

from preprocess_data.utils import advantage_annotation
from preprocess_data.utils import value_annotation
from scripts import train
import vam.models.vam_config as vam_config
from vam.shared.logging_utils import init_cli_logging
import vam.training.config as _config
import vam.training.optimizer as _optimizer
import vam.training.weight_loaders as weight_loaders
import vam.transforms as _transforms

VALUE_OBSERVATION_STRIDE = 1
ACTION_OBSERVATION_STRIDE = 30
VALUE_NUM_TRAIN_STEPS = 20_000
ACTION_NUM_TRAIN_STEPS = 10_000
VALUE_TRAIN_TYPE = {"td_learning"}
ACTION_TRAIN_TYPE = {"bc"}


@dataclasses.dataclass(frozen=True)
class AdvantageAwareDataFactory:
    base_factory: Any
    include_advantage: bool

    def create(self, assets_dirs: Path, model_config: Any) -> _config.DataConfig:
        data_config = self.base_factory.create(assets_dirs, model_config)
        repack_transforms = _set_advantage_mapping(
            data_config.repack_transforms,
            include_advantage=self.include_advantage,
        )
        return dataclasses.replace(data_config, repack_transforms=repack_transforms)


def _set_advantage_mapping(group: _transforms.Group, *, include_advantage: bool) -> _transforms.Group:
    patched_inputs = []
    for transform in group.inputs:
        if isinstance(transform, _transforms.RepackTransform):
            patched_inputs.append(
                dataclasses.replace(
                    transform,
                    structure=_set_advantage_in_structure(transform.structure, include_advantage=include_advantage),
                )
            )
        else:
            patched_inputs.append(transform)
    return dataclasses.replace(group, inputs=tuple(patched_inputs))


def _set_advantage_in_structure(structure: Any, *, include_advantage: bool) -> Any:
    if not isinstance(structure, Mapping):
        raise TypeError(f"Expected RepackTransform.structure to be a mapping, got {type(structure)!r}")

    patched = dict(structure)
    if include_advantage:
        patched["advantage"] = "advantage"
    else:
        patched.pop("advantage", None)
    return patched


def _has_advantage_mapping(group: _transforms.Group) -> bool:
    for transform in group.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue
        flat = traverse_util.flatten_dict(transform.structure)
        if flat.get(("advantage",)) == "advantage":
            return True
    return False


def _validate_advantage_mapping(config: _config.TrainConfig, *, expect_present: bool, stage_name: str) -> None:
    data_config = config.data.create(config.assets_dirs, config.model)
    has_advantage = _has_advantage_mapping(data_config.repack_transforms)
    if has_advantage != expect_present:
        expected = "present" if expect_present else "absent"
        raise ValueError(f"Stage '{stage_name}' requires advantage mapping to be {expected} in repack transforms.")


def _is_td_weight_loader(loader: weight_loaders.WeightLoader) -> bool:
    return isinstance(
        loader,
        weight_loaders.TDLearningWeightLoader | weight_loaders.Pi05ValueStateHistoryWeightLoader,
    )


def _build_td_weight_loader(loader: weight_loaders.WeightLoader) -> weight_loaders.WeightLoader:
    if _is_td_weight_loader(loader):
        return loader
    if isinstance(loader, weight_loaders.CheckpointWeightLoader):
        return weight_loaders.TDLearningWeightLoader(base_params_path=loader.params_path)
    raise ValueError(
        "RECAP cannot derive TD initialization from "
        f"{type(loader).__name__}; use TDLearningWeightLoader, Pi05ValueStateHistoryWeightLoader, "
        "or CheckpointWeightLoader in the base config."
    )


def _replace_action_expert(
    model: vam_config.VAMConfig,
    action_expert: vam_config.ActionExpertConfig,
) -> vam_config.VAMConfig:
    return dataclasses.replace(
        model,
        action_expert=action_expert,
    )


def _replace_action_obs_encoder(
    model: vam_config.VAMConfig,
    action_obs_encoder: vam_config.ObsEncoderConfig,
) -> vam_config.VAMConfig:
    return dataclasses.replace(
        model,
        action_obs_encoder=action_obs_encoder,
    )


def _build_pi06_action_model(model: vam_config.VAMConfig) -> vam_config.VAMConfig:
    if model.action_expert is None:
        raise ValueError("RECAP action training requires an action expert branch, but action_expert is None.")

    action_expert = dataclasses.replace(
        model.action_expert,
        pi05=True,
        pi06=True,
        advantage_conditioning=True,
    )
    return _replace_action_expert(dataclasses.replace(model, td_learning=False), action_expert)


def _build_annotation_value_model(model: vam_config.VAMConfig) -> vam_config.VAMConfig:
    annotation_model = dataclasses.replace(model, td_learning=False)
    if annotation_model.action_obs_encoder is None:
        return annotation_model

    # Advantage annotation only calls predict_value(), so it should follow the value branch input horizon.
    # This keeps action-branch history tokens from forcing value_prediction/state history fields to exist
    # while annotating a rollout repo for stage-4 action training.
    action_obs_encoder = dataclasses.replace(
        annotation_model.action_obs_encoder,
        observation_horizon=annotation_model.value_observation_horizon,
        history_tokens=vam_config.HistoryTokenConfig(),
    )
    return _replace_action_obs_encoder(annotation_model, action_obs_encoder)


def _unwrap_data_factory(data_factory: Any) -> Any:
    current = data_factory
    while hasattr(current, "base_factory"):
        current = current.base_factory
    return current


def _override_data_repo_id(data_factory: Any, repo_id: str | None) -> Any:
    if repo_id is None:
        return data_factory
    if not dataclasses.is_dataclass(data_factory):
        raise TypeError(f"Cannot override repo_id for non-dataclass data factory {type(data_factory)!r}.")
    if not hasattr(data_factory, "repo_id") or not hasattr(data_factory, "repo_ids"):
        raise TypeError(f"Data factory {type(data_factory)!r} does not expose repo_id/repo_ids fields.")
    return dataclasses.replace(data_factory, repo_id=repo_id, repo_ids=None)


def _override_observation_stride(data_factory: Any, observation_stride: int | None) -> Any:
    if observation_stride is None:
        return data_factory
    if not dataclasses.is_dataclass(data_factory):
        raise TypeError(f"Cannot override observation_stride for non-dataclass data factory {type(data_factory)!r}.")

    field_names = {field.name for field in dataclasses.fields(data_factory)}
    replacements = {}
    if "observation_stride" in field_names:
        replacements["observation_stride"] = observation_stride
    if "base_factory" in field_names:
        replacements["base_factory"] = _override_observation_stride(data_factory.base_factory, observation_stride)
    if not replacements:
        return data_factory
    return dataclasses.replace(data_factory, **replacements)


def _prepare_stage_data_factory(
    data_factory: Any,
    *,
    td_enabled: bool,
    repo_id: str | None = None,
    observation_stride: int | None = None,
) -> Any:
    data_factory = _override_data_repo_id(data_factory, repo_id)
    data_factory = _override_observation_stride(data_factory, observation_stride)
    if dataclasses.is_dataclass(data_factory) and hasattr(data_factory, "use_next_obs"):
        data_factory = dataclasses.replace(data_factory, use_next_obs=td_enabled)
    return data_factory


def _drop_observation_sequence_keys(data_factory: Any, keys_to_drop: set[str]) -> Any:
    if not dataclasses.is_dataclass(data_factory) or not hasattr(data_factory, "observation_sequence_keys"):
        return data_factory

    observation_sequence_keys = data_factory.observation_sequence_keys
    if observation_sequence_keys is None:
        return data_factory

    return dataclasses.replace(
        data_factory,
        observation_sequence_keys=tuple(key for key in observation_sequence_keys if key not in keys_to_drop),
    )


def _effective_repo_ids(data_factory: Any) -> tuple[str, ...]:
    data_factory = _unwrap_data_factory(data_factory)
    repo_ids = getattr(data_factory, "repo_ids", None)
    repo_id = getattr(data_factory, "repo_id", None)
    if repo_ids is not None:
        return tuple(repo_ids)
    if repo_id is None:
        return ()
    return (repo_id,)


def _effective_single_repo_id(data_factory: Any, *, stage_name: str) -> str:
    repo_ids = _effective_repo_ids(data_factory)
    if len(repo_ids) != 1:
        raise ValueError(f"Stage '{stage_name}' requires exactly one repo_id, got {repo_ids}.")
    return repo_ids[0]


def _assets_root(config: _config.TrainConfig, data_factory: Any) -> Path:
    data_factory = _unwrap_data_factory(data_factory)
    assets = getattr(data_factory, "assets", None)
    assets_dir = getattr(assets, "assets_dir", None)
    if assets_dir is None:
        return config.assets_dirs
    if str(assets_dir).startswith("gs://"):
        raise ValueError("RECAP norm-stat fallback only supports local assets directories, got gs:// assets_dir.")
    return Path(assets_dir).expanduser().resolve()


def _repo_norm_stats_path(config: _config.TrainConfig, data_factory: Any, repo_id: str) -> Path:
    return _assets_root(config, data_factory) / repo_id / "norm_stats.json"


def _ensure_repo_norm_stats(
    config: _config.TrainConfig,
    data_factory: Any,
    *,
    repo_id: str,
    offline_repo_id: str | None,
    stage_name: str,
) -> None:
    if repo_id == "fake" or not config.use_norm_stats:
        return

    target_path = _repo_norm_stats_path(config, data_factory, repo_id)
    if target_path.exists():
        return

    if offline_repo_id is None:
        raise FileNotFoundError(
            f"Stage '{stage_name}' is missing norm stats for repo_id '{repo_id}' at {target_path}. "
            "Run norm-stat computation for this repo or pass --offline-repo-id to copy compatible stats."
        )

    source_dir = _assets_root(config, data_factory) / offline_repo_id
    source_path = source_dir / "norm_stats.json"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Stage '{stage_name}' is missing norm stats for repo_id '{repo_id}', "
            f"and offline fallback stats were not found at {source_path}."
        )

    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    logging.info(
        "Copied norm stats fallback for stage '%s': %s -> %s",
        stage_name,
        source_dir,
        target_dir,
    )


def _ensure_stage_norm_stats(
    config: _config.TrainConfig,
    data_factory: Any,
    *,
    offline_repo_id: str | None,
    stage_name: str,
) -> None:
    for repo_id in _effective_repo_ids(data_factory):
        _ensure_repo_norm_stats(
            config,
            data_factory,
            repo_id=repo_id,
            offline_repo_id=offline_repo_id,
            stage_name=stage_name,
        )


def _validate_td_learning_consistency(config: _config.TrainConfig, *, stage_name: str) -> None:
    has_td_learning = "td_learning" in config.train_type
    data_factory = _unwrap_data_factory(getattr(config, "data", None))
    data_use_next_obs = getattr(data_factory, "use_next_obs", None)

    if has_td_learning:
        if not config.model.td_learning:
            raise ValueError(f"Stage '{stage_name}' includes td_learning but model.td_learning is False.")
        if data_use_next_obs is not True:
            raise ValueError(f"Stage '{stage_name}' includes td_learning but data.use_next_obs is not True.")
        if not _is_td_weight_loader(config.weight_loader):
            raise ValueError(
                f"Stage '{stage_name}' includes td_learning but weight_loader is not TDLearningWeightLoader."
            )
    else:
        if config.model.td_learning:
            raise ValueError(f"Stage '{stage_name}' excludes td_learning but model.td_learning is True.")
        if data_use_next_obs is not False:
            raise ValueError(f"Stage '{stage_name}' excludes td_learning but data.use_next_obs is not False.")
        if _is_td_weight_loader(config.weight_loader):
            raise ValueError(f"Stage '{stage_name}' excludes td_learning but uses a TD-compatible weight loader.")


def validate_recap_base_config(config: _config.TrainConfig) -> None:
    if not isinstance(config.model, vam_config.VAMConfig):
        raise TypeError(f"RECAP requires VAMConfig, got {type(config.model)!r} for config '{config.name}'.")
    if not config.model.has_value_expert_branch:
        raise ValueError(f"RECAP stage 1 requires a value expert branch for config '{config.name}'.")
    if not config.model.has_action_expert_branch:
        raise ValueError(f"RECAP stage 4 requires an action expert branch for config '{config.name}'.")
    if config.model.pi06:
        raise ValueError(f"RECAP stage 1 must start from pi06=False, got pi06=True for config '{config.name}'.")
    if not hasattr(config.data, "create"):
        raise TypeError(f"Config '{config.name}' data factory does not expose create(...).")
    _build_td_weight_loader(config.weight_loader)


def build_value_stage_config(
    base_config: _config.TrainConfig,
    exp_suffix: str,
    *,
    overwrite: bool,
    repo_id: str | None = None,
    observation_stride: int = VALUE_OBSERVATION_STRIDE,
    num_train_steps: int = VALUE_NUM_TRAIN_STEPS,
) -> _config.TrainConfig:
    checkpoint_base_dir = str((Path.cwd() / "checkpoints").resolve())
    value_model = dataclasses.replace(base_config.model, td_learning=True)
    value_data_factory = _prepare_stage_data_factory(
        base_config.data,
        td_enabled=True,
        repo_id=repo_id,
        observation_stride=observation_stride,
    )
    return dataclasses.replace(
        base_config,
        exp_name=f"value_{exp_suffix}",
        checkpoint_base_dir=checkpoint_base_dir,
        overwrite=overwrite,
        resume=False,
        model=value_model,
        num_train_steps=num_train_steps,
        train_type=set(VALUE_TRAIN_TYPE),
        weight_loader=_build_td_weight_loader(base_config.weight_loader),
        freeze_filter=value_model.get_freeze_filter(freeze_value=True),
        data=AdvantageAwareDataFactory(value_data_factory, include_advantage=False),
    )


def build_annotation_stage_config(
    value_config: _config.TrainConfig,
    *,
    data_factory: Any | None = None,
    observation_stride: int = ACTION_OBSERVATION_STRIDE,
) -> _config.TrainConfig:
    annotation_model = _build_annotation_value_model(value_config.model)
    annotation_data_factory = _unwrap_data_factory(value_config.data) if data_factory is None else data_factory
    annotation_data_factory = _prepare_stage_data_factory(
        annotation_data_factory,
        td_enabled=False,
        observation_stride=observation_stride,
    )
    annotation_data_factory = _drop_observation_sequence_keys(annotation_data_factory, {"value_prediction"})
    return dataclasses.replace(
        value_config,
        model=annotation_model,
        train_type=set(),
        batch_size=64,
        weight_loader=weight_loaders.NoOpWeightLoader(),
        data=AdvantageAwareDataFactory(annotation_data_factory, include_advantage=False),
    )


def build_action_stage_config(
    base_config: _config.TrainConfig,
    exp_suffix: str,
    ckpt_params_path: Path,
    *,
    overwrite: bool,
    repo_id: str | None = None,
    observation_stride: int = ACTION_OBSERVATION_STRIDE,
    num_train_steps: int = ACTION_NUM_TRAIN_STEPS,
) -> _config.TrainConfig:
    checkpoint_base_dir = str((Path.cwd() / "checkpoints").resolve())
    action_model = _build_pi06_action_model(base_config.model)

    action_data_factory = _prepare_stage_data_factory(
        base_config.data,
        td_enabled=False,
        repo_id=repo_id,
        observation_stride=observation_stride,
    )

    return dataclasses.replace(
        base_config,
        exp_name=f"action_{exp_suffix}",
        checkpoint_base_dir=checkpoint_base_dir,
        overwrite=overwrite,
        resume=False,
        model=action_model,
        batch_size=64,
        num_train_steps=num_train_steps,
        save_interval=10_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=1_000_000,
            decay_lr=1.0e-5,
        ),
        train_type=set(ACTION_TRAIN_TYPE),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ckpt_params_path)),
        freeze_filter=action_model.get_freeze_filter(freeze_value=True),
        validation=dataclasses.replace(base_config.validation, enabled=False),
        data=AdvantageAwareDataFactory(action_data_factory, include_advantage=True),
    )


def resolve_latest_checkpoint_step(checkpoint_dir: Path) -> int | None:
    if not checkpoint_dir.exists():
        return None

    steps = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "params").exists():
            steps.append(int(child.name))
    return max(steps) if steps else None


def resolve_checkpoint_params_path(value_config: _config.TrainConfig) -> Path:
    checkpoint_dir = value_config.checkpoint_dir
    latest_step = resolve_latest_checkpoint_step(checkpoint_dir)
    if latest_step is not None:
        return checkpoint_dir / str(latest_step) / "params"

    expected_step = value_config.num_train_steps - 1
    expected_path = checkpoint_dir / str(expected_step) / "params"
    if expected_path.exists():
        return expected_path

    raise FileNotFoundError(
        f"No checkpoint params found under {checkpoint_dir}. Expected latest step or fallback {expected_step}."
    )


def _finish_wandb_run() -> None:
    if wandb.run is not None:
        wandb.finish()


def run_recap(
    config_name: str,
    *,
    exp_name: str,
    percentile: float,
    overwrite: bool,
    advantage_obs_horizon: int | None = 1,
    advantage_source: str = "value_prediction",
    advantage_method: str = "gae",
    gae_lambda: float = 0.9,
    value_observation_stride: int = VALUE_OBSERVATION_STRIDE,
    action_observation_stride: int = ACTION_OBSERVATION_STRIDE,
    value_num_train_steps: int = VALUE_NUM_TRAIN_STEPS,
    action_num_train_steps: int = ACTION_NUM_TRAIN_STEPS,
    value_repo_id: str | None = None,
    action_repo_id: str | None = None,
    offline_repo_id: str | None = None,
) -> None:
    if advantage_obs_horizon is not None and advantage_obs_horizon < 1:
        raise ValueError(f"advantage_obs_horizon must be >= 1 when set, got {advantage_obs_horizon}")
    if advantage_source not in {"model", "value_prediction"}:
        raise ValueError(f"advantage_source must be one of {{'model', 'value_prediction'}}, got {advantage_source!r}")
    if advantage_method not in {"value_avg", "gae"}:
        raise ValueError(f"advantage_method must be one of {{'value_avg', 'gae'}}, got {advantage_method!r}")
    if advantage_method == "gae" and advantage_obs_horizon is None:
        raise ValueError("--advantage-method=gae requires --advantage-obs-horizon")
    if advantage_source == "value_prediction" and advantage_obs_horizon is None:
        raise ValueError("--advantage-source=value_prediction requires --advantage-obs-horizon")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
    if value_observation_stride < 1:
        raise ValueError(f"value_observation_stride must be >= 1, got {value_observation_stride}")
    if action_observation_stride < 1:
        raise ValueError(f"action_observation_stride must be >= 1, got {action_observation_stride}")
    if value_num_train_steps < 1:
        raise ValueError(f"value_num_train_steps must be >= 1, got {value_num_train_steps}")
    if action_num_train_steps < 1:
        raise ValueError(f"action_num_train_steps must be >= 1, got {action_num_train_steps}")

    base_config = _config.get_config(config_name)
    validate_recap_base_config(base_config)

    value_config = build_value_stage_config(
        base_config,
        exp_name,
        overwrite=overwrite,
        repo_id=value_repo_id,
        observation_stride=value_observation_stride,
        num_train_steps=value_num_train_steps,
    )
    _ensure_stage_norm_stats(value_config, value_config.data, offline_repo_id=offline_repo_id, stage_name="value_train")
    _validate_advantage_mapping(value_config, expect_present=False, stage_name="value_train")
    _validate_td_learning_consistency(value_config, stage_name="value_train")
    logging.info("Running stage 1 value training: %s", value_config.exp_name)
    train.main(value_config)
    _finish_wandb_run()

    action_annotation_data_factory = _prepare_stage_data_factory(
        base_config.data,
        td_enabled=False,
        repo_id=action_repo_id,
        observation_stride=action_observation_stride,
    )
    annotation_config = build_annotation_stage_config(
        value_config,
        data_factory=action_annotation_data_factory,
        observation_stride=action_observation_stride,
    )
    _ensure_stage_norm_stats(
        annotation_config,
        action_annotation_data_factory,
        offline_repo_id=offline_repo_id,
        stage_name="advantage_annotate",
    )
    _validate_advantage_mapping(annotation_config, expect_present=False, stage_name="advantage_annotate")
    _validate_td_learning_consistency(annotation_config, stage_name="advantage_annotate")
    checkpoint_params_path = resolve_checkpoint_params_path(value_config)

    action_value_repo_id = _effective_single_repo_id(
        action_annotation_data_factory,
        stage_name="value_prediction_annotate",
    )
    _ensure_repo_norm_stats(
        annotation_config,
        action_annotation_data_factory,
        repo_id=action_value_repo_id,
        offline_repo_id=offline_repo_id,
        stage_name="value_prediction_annotate",
    )
    logging.info(
        "Running stage 2 value_prediction annotation for action repo '%s' from %s",
        action_value_repo_id,
        checkpoint_params_path,
    )
    value_annotation.annotate_value_predictions_with_config(
        config=annotation_config,
        ckpt_path=str(checkpoint_params_path),
        overwrite=overwrite,
    )

    logging.info("Running stage 3 advantage annotation from %s", checkpoint_params_path)
    advantage_annotation.run_annotate_advantage_with_config(
        config=annotation_config,
        ckpt_path=str(checkpoint_params_path),
        percentile=percentile,
        overwrite=overwrite,
        advantage_obs_horizon=advantage_obs_horizon,
        advantage_source=advantage_source,
        advantage_method=advantage_method,
        gae_lambda=gae_lambda,
        force_control_flag_one_positive=True,
    )

    action_config = build_action_stage_config(
        base_config,
        exp_name,
        checkpoint_params_path,
        overwrite=overwrite,
        repo_id=action_repo_id,
        observation_stride=action_observation_stride,
        num_train_steps=action_num_train_steps,
    )
    _ensure_stage_norm_stats(
        action_config, action_config.data, offline_repo_id=offline_repo_id, stage_name="action_train"
    )
    _validate_advantage_mapping(action_config, expect_present=True, stage_name="action_train")
    _validate_td_learning_consistency(action_config, stage_name="action_train")
    logging.info("Running stage 4 action training: %s", action_config.exp_name)
    train.main(action_config)
    _finish_wandb_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline RL warmup: TD value learning, annotation and advantage-conditioned flow matching."
    )
    parser.add_argument("config_name", help="Base TrainConfig name for offline RL warmup.")
    parser.add_argument("--exp-name", required=True, help="Shared experiment suffix used for value_*/action_*.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=70.0,
        help="Percentile threshold used when annotating advantage labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage checkpoint directories and annotation columns.",
    )
    parser.add_argument(
        "--value-repo-id",
        default=None,
        help="Optional LeRobot repo id override for stage-1 value training.",
    )
    parser.add_argument(
        "--action-repo-id",
        default=None,
        help="Optional LeRobot repo id override for stage-2/3 annotation and stage-4 action training.",
    )
    parser.add_argument(
        "--offline-repo-id",
        default=None,
        help="Repo id whose norm stats are copied when value/action repo norm stats are missing.",
    )
    parser.add_argument(
        "--advantage-obs-horizon",
        type=int,
        default=1,
        help=(
            "Horizon for stage-3 advantage annotation. Defaults to 1, matching scripts/train.sh. "
            "Pass a different value for longer-horizon GAE/value averaging."
        ),
    )
    parser.add_argument(
        "--advantage-source",
        choices=("model", "value_prediction"),
        default="value_prediction",
        help=(
            "Use stage-1 model inference or refreshed value_prediction parquet columns for stage-3 "
            "advantage annotation."
        ),
    )
    parser.add_argument(
        "--advantage-method",
        choices=("value_avg", "gae"),
        default="gae",
        help=(
            "Stage-3 advantage method when --advantage-obs-horizon is set. value_avg keeps the future-value "
            "average behavior; gae computes chunk-level generalized advantage estimation."
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.9,
        help="GAE lambda used with --advantage-method=gae.",
    )
    parser.add_argument(
        "--value-observation-stride",
        type=int,
        default=VALUE_OBSERVATION_STRIDE,
        help=f"Observation stride used for stage-1 value training. Defaults to {VALUE_OBSERVATION_STRIDE}.",
    )
    parser.add_argument(
        "--action-observation-stride",
        type=int,
        default=ACTION_OBSERVATION_STRIDE,
        help=(
            "Observation stride used for stage-2/3 annotation and stage-4 action training. "
            f"Defaults to {ACTION_OBSERVATION_STRIDE}."
        ),
    )
    parser.add_argument(
        "--value-num-train-steps",
        type=int,
        default=VALUE_NUM_TRAIN_STEPS,
        help=f"Number of stage-1 value training steps. Defaults to {VALUE_NUM_TRAIN_STEPS}.",
    )
    parser.add_argument(
        "--action-num-train-steps",
        type=int,
        default=ACTION_NUM_TRAIN_STEPS,
        help=f"Number of stage-4 action training steps. Defaults to {ACTION_NUM_TRAIN_STEPS}.",
    )
    return parser.parse_args()


def main() -> None:
    init_cli_logging()
    args = parse_args()
    run_recap(
        args.config_name,
        exp_name=args.exp_name,
        percentile=args.percentile,
        overwrite=args.overwrite,
        advantage_obs_horizon=args.advantage_obs_horizon,
        advantage_source=args.advantage_source,
        advantage_method=args.advantage_method,
        gae_lambda=args.gae_lambda,
        value_observation_stride=args.value_observation_stride,
        action_observation_stride=args.action_observation_stride,
        value_num_train_steps=args.value_num_train_steps,
        action_num_train_steps=args.action_num_train_steps,
        value_repo_id=args.value_repo_id,
        action_repo_id=args.action_repo_id,
        offline_repo_id=args.offline_repo_id,
    )


if __name__ == "__main__":
    main()
