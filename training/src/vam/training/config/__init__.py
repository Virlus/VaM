"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias, cast

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import vam.models.model as _model
import vam.models.tokenizer as _tokenizer
import vam.models.vam_config as vam_config
import vam.policies.async_aloha_policy as async_aloha_policy
import vam.shared.download as _download
import vam.shared.normalize as _normalize

# Async streaming control-plane dataclasses live in a submodule; re-export them so that
# `vam.training.config.<Name>` keeps working for all existing importers.
from vam.training.config.async_specs import AsyncTrainingConfig
from vam.training.config.async_specs import AsyncTrainStreamConfig
from vam.training.config.async_specs import CheckpointMergeConfig
from vam.training.config.async_specs import CheckpointPublisherConfig
from vam.training.config.async_specs import CheckpointSwitchConfig as CheckpointSwitchConfig  # noqa: PLC0414
from vam.training.config.async_specs import DatasetViewConfig
from vam.training.config.async_specs import EpisodeReplayConfig
from vam.training.config.async_specs import GpuUtilizationMonitorConfig
from vam.training.config.async_specs import RepoSpec
from vam.training.config.async_specs import SamplerConfig
from vam.training.config.async_specs import SamplerGroupConfig
import vam.training.optimizer as _optimizer
import vam.training.weight_loaders as weight_loaders
import vam.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


def _required_environment_value(name: str) -> str:
    value = os.environ.get(name)
    return value if value else tyro.MISSING


def _required_environment_int(name: str) -> int:
    value = os.environ.get(name)
    return int(value) if value else tyro.MISSING


_ALOHA_ASYNC_OFFLINE_REPO_ID = _required_environment_value("ALOHA_ASYNC_OFFLINE_REPO_ID")
_ALOHA_ASYNC_NORM_ASSETS_DIR = _required_environment_value("ALOHA_ASYNC_NORM_ASSETS_DIR")
_ALOHA_ASYNC_FIXED_NORM_STATS_PATH = os.environ.get("ALOHA_ASYNC_NORM_STATS_PATH")
if (
    not _ALOHA_ASYNC_FIXED_NORM_STATS_PATH
    and isinstance(_ALOHA_ASYNC_NORM_ASSETS_DIR, str)
    and isinstance(_ALOHA_ASYNC_OFFLINE_REPO_ID, str)
):
    _ALOHA_ASYNC_FIXED_NORM_STATS_PATH = str(
        pathlib.Path(_ALOHA_ASYNC_NORM_ASSETS_DIR) / _ALOHA_ASYNC_OFFLINE_REPO_ID / "norm_stats.json"
    )
if not _ALOHA_ASYNC_FIXED_NORM_STATS_PATH:
    _ALOHA_ASYNC_FIXED_NORM_STATS_PATH = tyro.MISSING
_ALOHA_ASYNC_BASE_PARAMS = _required_environment_value("ALOHA_ASYNC_BASE_PARAMS")


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None

    # Optional absolute or relative path to a fixed norm_stats.json file, or to a directory containing one. When set,
    # this takes precedence over assets_dir/asset_id and keeps streaming dataset-view changes from changing stats.
    norm_stats_path: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional multiple LeRobot repo ids for mixed-dataset training.
    repo_ids: tuple[str, ...] | None = None
    # Optional per-repo episode selection. Used by streaming dataset views.
    repo_specs: tuple[RepoSpec, ...] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the next observation in the sequence as the observation for the next step.
    use_next_obs: bool = False
    # Names of raw LeRobot keys that will be used by the data loader to generate observation history. The length
    # of the sequence is defined by `observation_horizon` in the model config.
    observation_sequence_keys: tuple[str, ...] | None = None
    # Frame stride used when loading observation history. The default of 30 corresponds to one-second spacing for
    # 30 FPS LeRobot datasets.
    observation_stride: int = 30

    disable_proprio_for_value: bool = False

    td_learning: bool = False
    # If set, use balanced sampling across these exact data_source values.
    # For example, (-1.0, 0.0), (-1.0, 1.0), or (-1.0, 0.0, 1.0).
    balanced_data_source_values: tuple[float, ...] | None = None
    # If true and balanced_data_source_values includes 1.0, split the online data_source=1.0 quota 1:1 between
    # success=1 and success=0 samples.
    balance_online_success_failure: bool = False
    # If true and balanced_data_source_values includes 1.0, split the online data_source=1.0 quota 1:1 between
    # terminate=1 and terminate=0 samples. If combined with balance_online_success_failure, split online samples
    # across success/terminate combinations in a 1:1:1:1 ratio.
    balance_online_termination: bool = False
    sampler: SamplerConfig | None = None

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    def effective_repo_ids(self) -> tuple[str, ...]:
        if self.repo_specs is not None:
            if self.repo_id is not None or self.repo_ids is not None:
                raise ValueError("DataConfig must not set repo_id/repo_ids when repo_specs is set.")
            return tuple(spec.repo_id for spec in self.repo_specs)
        if self.repo_ids is not None:
            if self.repo_id is not None:
                raise ValueError("DataConfig must not set both repo_id and repo_ids.")
            return tuple(self.repo_ids)
        if self.repo_id is None:
            return ()
        return (self.repo_id,)

    def is_multi_repo(self) -> bool:
        return len(self.effective_repo_ids()) > 1


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    use_next_obs: bool = False
    disable_proprio_for_value: bool = False
    advantage_conditioning: bool | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not isinstance(model_config, vam_config.VAMConfig):
            raise TypeError("VAM supports VAMConfig models")
        model_transforms = [
            _transforms.InjectDefaultPrompt(self.default_prompt),
            _transforms.ResizeImages(224, 224, self.use_next_obs),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
                use_next_obs=self.use_next_obs,
                disable_proprio_for_value=self.disable_proprio_for_value,
                omit_state_from_prompt=model_config.use_history_state_tokens,
            ),
        ]
        if model_config.pi06:
            model_transforms.append(
                _transforms.InjectAdvantage(
                    _tokenizer.PaligemmaTokenizer(),
                    advantage_conditioning=(
                        model_config.advantage_conditioning
                        if self.advantage_conditioning is None
                        else self.advantage_conditioning
                    ),
                )
            )
        model_transforms.append(_transforms.PadStatesAndActions(model_config.action_dim, self.use_next_obs))
        return _transforms.Group(inputs=model_transforms)


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | None = None
    # Optional multiple LeRobot repo ids for mixed-dataset training.
    repo_ids: tuple[str, ...] | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id
        repo_ids = tuple(self.repo_ids) if self.repo_ids is not None else None
        base_repo_specs = self.base_config.repo_specs if self.base_config is not None else None
        if repo_ids is not None and len(repo_ids) == 0:
            raise ValueError("repo_ids must not be empty.")
        if repo_id is not None and repo_ids is not None:
            raise ValueError("Data config must not set both repo_id and repo_ids.")
        if base_repo_specs is not None and (repo_id is not None or repo_ids is not None):
            raise ValueError("Data config must not set repo_id/repo_ids when base_config.repo_specs is set.")
        asset_id = self.assets.asset_id or repo_id or (base_repo_specs[0].repo_id if base_repo_specs else None)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            repo_ids=repo_ids,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if self.assets.norm_stats_path:
            norm_stats_path = epath.Path(self.assets.norm_stats_path).expanduser()
            norm_stats_dir = norm_stats_path.parent if norm_stats_path.name == "norm_stats.json" else norm_stats_path
            norm_stats = _normalize.load(_download.maybe_download(str(norm_stats_dir)))
            logging.info(f"Loaded fixed norm stats from {norm_stats_dir}")
            return norm_stats
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotMyDataAlohaValueConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = True
    extra_value_transform: bool = False
    bimanual_robot: bool = False
    n_bins: int = 256
    use_next_obs: bool = False
    disable_proprio_for_value: bool = False
    observation_sequence_keys: tuple[str, ...] | None = (
        "base_image",
        "left_wrist_image",
        "right_wrist_image",
        "state",
    )
    observation_stride: int = 30
    # If set, will use balanced sampling from the listed exact data_source values.
    balanced_data_source_values: tuple[float, ...] | None = None
    # If true, split the online data_source=1.0 quota 1:1 between success=1 and success=0 samples.
    balance_online_success_failure: bool = False
    # If true, split the online data_source=1.0 quota 1:1 between terminate=1 and terminate=0 samples. If combined
    # with balance_online_success_failure, split online samples across success/terminate combinations 1:1:1:1.
    balance_online_termination: bool = False
    # RECAP/PI06 action training consumes the parquet advantage column written by preprocess_data/annotate.py advantage.
    # None (default) inherits the model's advantage conditioning (``model_config.advantage_conditioning``, which
    # itself defaults to ``pi06``): a pi06 action expert loads and injects advantage without a separate flag. Set an
    # explicit bool to override, e.g. a TD value stream on a pi06 model whose repo has no ``advantage`` column.
    include_advantage: bool | None = None
    action_sequence_keys: Sequence[str] = ("actions", "next_actions")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the ALOHA policy adapter).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        # ``include_advantage`` defaults to the model's advantage conditioning (pi06); an explicit bool overrides.
        include_advantage = (
            bool(getattr(model_config, "advantage_conditioning", False))
            if self.include_advantage is None
            else self.include_advantage
        )
        if self.bimanual_robot:
            repack_structure = {
                "observation/base_image": "base_image",
                "observation/left_wrist_image": "left_wrist_image",
                "observation/right_wrist_image": "right_wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "value": "value",
                "success": "success",
                "reward": "reward",
                "data_source": "data_source",
                "terminate": "terminate",
                "prompt": "task",
            }
            if self.use_next_obs:
                repack_structure.update(
                    {
                        "observation/next_base_image": "next_base_image",
                        "observation/next_left_wrist_image": "next_left_wrist_image",
                        "observation/next_right_wrist_image": "next_right_wrist_image",
                        "observation/next_state": "next_state",
                        "next_actions": "next_actions",
                    }
                )
            if include_advantage:
                repack_structure["advantage"] = "advantage"
            if getattr(model_config, "observation_horizon", 1) > 1:
                repack_structure.update(
                    {
                        "observation/base_image_is_pad": "base_image_is_pad",
                        "observation/left_wrist_image_is_pad": "left_wrist_image_is_pad",
                        "observation/right_wrist_image_is_pad": "right_wrist_image_is_pad",
                        "observation/state_is_pad": "state_is_pad",
                    }
                )
                if self.observation_sequence_keys is not None and "value_prediction" in self.observation_sequence_keys:
                    repack_structure.update(
                        {
                            "value_prediction": "value_prediction",
                            "value_prediction_is_pad": "value_prediction_is_pad",
                        }
                    )
            optional_keys = {"control_flag", "data_source", "terminate"}
            if getattr(model_config, "observation_horizon", 1) > 1 and self.use_next_obs:
                repack_structure.update(
                    {
                        "observation/next_base_image_is_pad": "next_base_image_is_pad",
                        "observation/next_left_wrist_image_is_pad": "next_left_wrist_image_is_pad",
                        "observation/next_right_wrist_image_is_pad": "next_right_wrist_image_is_pad",
                        "observation/next_state_is_pad": "next_state_is_pad",
                    }
                )
            repack_transform = _transforms.Group(
                inputs=[_transforms.RepackTransform(repack_structure, optional_keys=optional_keys)]
            )
        else:
            raise NotImplementedError("Only bimanual robot is supported for now")

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        if self.bimanual_robot:
            aloha_inputs = (
                async_aloha_policy.AsyncSequenceAlohaInputs(
                    model_type=model_config.model_type, action_horizon=model_config.action_horizon
                )
                if self.use_next_obs
                else async_aloha_policy.AsyncAlohaInputs(
                    model_type=model_config.model_type, action_horizon=model_config.action_horizon
                )
            )
            data_transforms = _transforms.Group(
                inputs=[
                    _transforms.InjectDefaultPrompt(None),
                    aloha_inputs,
                    _transforms.EnsureValueDim(),
                    _transforms.EnsureSuccessDim(),
                    _transforms.EnsureControlFlagDim(),
                    _transforms.EnsureValuePredictionDim(),
                    # Rewards already have action_horizon dimensions; do not add a singleton axis.
                    _transforms.EnsureDataSourceDim(),
                    _transforms.EnsureTerminateDim(),
                ],
                outputs=[async_aloha_policy.AsyncSequenceAlohaOutputs()],
            )
        else:
            raise NotImplementedError("Only bimanual robot is supported for now")

        # Joint deltas are applied below to both arms; gripper targets remain absolute.
        if self.extra_value_transform:
            if not isinstance(model_config, vam_config.VAMConfig):
                raise TypeError("extra_value_transform requires a VAMConfig so value support is config-driven.")
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.DiscretizeValues(
                        n_bins=model_config.n_bins,
                        vmin=model_config.vmin,
                        vmax=model_config.vmax,
                        gaussian_sigma=model_config.gaussian_sigma,
                    )
                ],
                outputs=[
                    _transforms.SerializeValues(
                        n_bins=model_config.n_bins,
                        vmin=model_config.vmin,
                        vmax=model_config.vmax,
                    )
                ],
            )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory(
            use_next_obs=self.use_next_obs,
            disable_proprio_for_value=self.disable_proprio_for_value,
            advantage_conditioning=include_advantage,
        )(model_config)

        # Use sequential observations to reweight training samples with value estimation
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_next_obs=self.use_next_obs,
            disable_proprio_for_value=self.disable_proprio_for_value,
            balanced_data_source_values=self.balanced_data_source_values,
            balance_online_success_failure=self.balance_online_success_failure,
            balance_online_termination=self.balance_online_termination,
            action_sequence_keys=self.action_sequence_keys,
            observation_sequence_keys=self.observation_sequence_keys,
            observation_stride=self.observation_stride,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "vam"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., VAMConfig) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=vam_config.VAMConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    # If true, apply normalization stats in the training dataloader.
    use_norm_stats: bool = True

    # Optional synchronous validation settings. When enabled, saved checkpoints pause training
    # and evaluate the current in-memory model on a fixed episode subset from a separate dataset.
    validation: "ValidationConfig" = dataclasses.field(default_factory=lambda: ValidationConfig())

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 20000

    # Dataloader and prefetch controls.
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of host batches to prefetch on a background thread.
    prefetch_size: int = 2
    # If true, move prefetched batches onto device ahead of use.
    device_prefetch: bool = True
    # Torch DataLoader tuning knobs.
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    # Training types
    train_type: set[str] = dataclasses.field(default_factory=set)
    # If true, masks action loss with the success signal when success labels are available.
    filter_action_loss_by_success: bool = True

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # If true, log first-batch camera previews and augmentation previews before entering the training loop.
    startup_previews: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # Async streaming training control plane. train.py ignores this; scripts/train_async.py consumes it.
    async_training: AsyncTrainingConfig = dataclasses.field(default_factory=AsyncTrainingConfig)

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

        if getattr(self.model, "td_loss_type", None) == "distributional" and not getattr(
            self.model, "discrete_value", False
        ):
            raise ValueError("td_loss_type='distributional' requires a discrete value/Q head.")

        if self.validation.enabled:
            if not self.validation.effective_repo_ids():
                raise ValueError("validation.repo_id or validation.repo_ids must be set when validation is enabled.")
            if self.validation.num_episodes <= 0:
                raise ValueError("validation.num_episodes must be > 0 when validation is enabled.")
            if self.validation.start_index < 0:
                raise ValueError("validation.start_index must be >= 0 when validation is enabled.")
            if self.validation.inference_batch_size is not None and self.validation.inference_batch_size <= 0:
                raise ValueError("validation.inference_batch_size must be > 0 when set.")
            if self.validation.num_workers is not None and self.validation.num_workers < 0:
                raise ValueError("validation.num_workers must be >= 0 when set.")
            if self.validation.prefetch_size is not None and self.validation.prefetch_size < 0:
                raise ValueError("validation.prefetch_size must be >= 0 when set.")
            if self.validation.prefetch_factor is not None and self.validation.prefetch_factor <= 0:
                raise ValueError("validation.prefetch_factor must be > 0 when set.")


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = False
    repo_id: str | None = None
    repo_ids: tuple[str, ...] | None = None
    use_norm_stats: bool | None = None
    num_episodes: int = 0
    start_index: int = 0
    inference_batch_size: int | None = None
    num_workers: int | None = None
    prefetch_size: int | None = None
    prefetch_factor: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None

    def effective_repo_ids(self) -> tuple[str, ...]:
        if self.repo_ids is not None:
            if self.repo_id is not None:
                raise ValueError("ValidationConfig must not set both repo_id and repo_ids.")
            return tuple(self.repo_ids)
        if self.repo_id is None:
            return ()
        return (self.repo_id,)


# Use `get_config` if you need to get a config by name in your
ALOHA_ASYNC_RECAP_NORM_ASSETS = AssetsConfig(
    assets_dir=_ALOHA_ASYNC_NORM_ASSETS_DIR,
    asset_id=_ALOHA_ASYNC_OFFLINE_REPO_ID,
    norm_stats_path=_ALOHA_ASYNC_FIXED_NORM_STATS_PATH,
)


def _aloha_async_recap_data(
    *,
    repo_id: str,
    sampler: SamplerConfig | None,
    include_advantage: bool | None = None,
    use_next_obs: bool,
    observation_stride: int,
) -> LeRobotMyDataAlohaValueConfig:
    return LeRobotMyDataAlohaValueConfig(
        repo_id=repo_id,
        assets=ALOHA_ASYNC_RECAP_NORM_ASSETS,
        base_config=DataConfig(prompt_from_task=False, sampler=sampler),
        extra_delta_transform=True,
        extra_value_transform=True,
        use_next_obs=use_next_obs,
        bimanual_robot=True,
        n_bins=256,
        disable_proprio_for_value=True,
        include_advantage=include_advantage,
        observation_stride=observation_stride,
        observation_sequence_keys=(
            "base_image",
            "left_wrist_image",
            "right_wrist_image",
            "state",
            "value_prediction",
        ),
    )


ALOHA_ASYNC_ACTION_SAMPLER = SamplerConfig(
    type="weighted_column_groups",
    groups=(
        SamplerGroupConfig("offline_demo", 1.0, {"data_source": 0.0}),
        SamplerGroupConfig(
            "off_policy_human_intervention",
            1.0,
            {"control_flag": 1.0, "policy_id_lt_current": True},
        ),
        SamplerGroupConfig(
            "on_policy_rollout_or_dagger",
            1.0,
            {"policy_id_eq_current": True},
            episode_replay=EpisodeReplayConfig(),
        ),
    ),
    drop_empty_groups=True,
)

ALOHA_ASYNC_VALUE_SAMPLER = SamplerConfig(
    type="weighted_column_groups",
    groups=(
        SamplerGroupConfig(
            "offline_or_off_policy_trajectory",
            1.0,
            {"policy_id_lt_current": True},
        ),
        SamplerGroupConfig(
            "on_policy_trajectory",
            1.0,
            {"policy_id_eq_current": True},
            episode_replay=EpisodeReplayConfig(),
        ),
    ),
    drop_empty_groups=True,
)


_CONFIGS = [
    TrainConfig(
        name="pi05_aloha_value_expert",
        exp_name="offline",
        model=(
            model := vam_config.VAMConfig(
                action_obs_encoder=vam_config.ObsEncoderConfig(
                    variant="gemma_2b",
                    observation_horizon=1,
                    history_tokens=vam_config.HistoryTokenConfig(use_state_tokens=False, use_value_tokens=False),
                ),
                action_expert=vam_config.ActionExpertConfig(
                    pi05=True,
                    pi06=True,
                    discrete_state_input=True,
                    async_action_horizon=1,
                ),
                value_obs_encoder=vam_config.ObsEncoderConfig(
                    variant="gemma_300m",
                    observation_horizon=1,
                    image_encoder=vam_config.ImageEncoderConfig(encoder_type="siglip"),
                ),
                value_expert=vam_config.ValueExpertConfig(n_bins=256),
                action_horizon=100,
                td_learning=False,
                gamma=0.999,
            )
        ),
        data=LeRobotMyDataAlohaValueConfig(
            repo_id="example/offline",
            assets=AssetsConfig(asset_id="example/offline"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            extra_value_transform=True,
            bimanual_robot=True,
            disable_proprio_for_value=True,
            include_advantage=False,
            use_next_obs=False,
            observation_stride=30,
            observation_sequence_keys=("base_image", "left_wrist_image", "right_wrist_image", "state"),
        ),
        weight_loader=weight_loaders.Pi05ValueStateHistoryWeightLoader(
            params_path="/path/to/base/checkpoint/params",
        ),
        train_type={"bc", "value_mc_learning"},
        freeze_filter=model.get_freeze_filter(freeze_value=False),
        validation=ValidationConfig(enabled=False),
        filter_action_loss_by_success=False,
        batch_size=64,
        num_workers=16,
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=5_000,
        ema_decay=0.999,
    ),
    TrainConfig(
        name="pi05_aloha_history_value_expert",
        exp_name="offline",
        model=(
            model := vam_config.VAMConfig(
                action_obs_encoder=vam_config.ObsEncoderConfig(
                    variant="gemma_2b",
                    observation_horizon=4,
                    image_encoder=vam_config.ImageEncoderConfig(encoder_type="siglip_video"),
                    history_tokens=vam_config.HistoryTokenConfig(use_state_tokens=True, use_value_tokens=True),
                ),
                action_expert=vam_config.ActionExpertConfig(
                    pi05=True,
                    pi06=True,
                    discrete_state_input=False,
                    async_action_horizon=1,
                ),
                value_obs_encoder=vam_config.ObsEncoderConfig(
                    variant="gemma_300m",
                    observation_horizon=2,
                    image_encoder=vam_config.ImageEncoderConfig(encoder_type="siglip_video"),
                ),
                value_expert=vam_config.ValueExpertConfig(n_bins=256),
                action_horizon=100,
                td_learning=False,
                gamma=0.9997,
            )
        ),
        data=LeRobotMyDataAlohaValueConfig(
            repo_id="example/offline",
            assets=AssetsConfig(asset_id="example/offline"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            extra_value_transform=True,
            bimanual_robot=True,
            disable_proprio_for_value=True,
            include_advantage=False,
            use_next_obs=False,
            observation_stride=30,
            observation_sequence_keys=(
                "base_image",
                "left_wrist_image",
                "right_wrist_image",
                "state",
                "value_prediction",
            ),
        ),
        weight_loader=weight_loaders.Pi05ValueStateHistoryWeightLoader(
            params_path="/path/to/base/checkpoint/params",
        ),
        train_type={"bc", "value_mc_learning"},
        freeze_filter=model.get_freeze_filter(freeze_value=False),
        validation=ValidationConfig(enabled=False),
        filter_action_loss_by_success=False,
        batch_size=64,
        num_workers=16,
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=5_000,
        ema_decay=0.999,
    ),
    TrainConfig(
        name="pi05_aloha_async_recap_dummy",
        model=(
            model := vam_config.VAMConfig(
                action_obs_encoder=vam_config.ObsEncoderConfig(
                    kind="paligemma",
                    variant="gemma_2b",
                    observation_horizon=4,
                    image_encoder=vam_config.ImageEncoderConfig(encoder_type="siglip_video"),
                    history_tokens=vam_config.HistoryTokenConfig(use_state_tokens=True, use_value_tokens=True),
                ),
                action_expert=vam_config.ActionExpertConfig(
                    pi05=True,
                    pi06=True,
                    discrete_state_input=False,
                    async_action_horizon=1,
                ),
                value_obs_encoder=vam_config.ObsEncoderConfig(
                    kind="paligemma",
                    variant="gemma_300m",
                    observation_horizon=2,
                    image_encoder=vam_config.ImageEncoderConfig(encoder_type="siglip_video"),
                    history_tokens=vam_config.HistoryTokenConfig(use_state_tokens=False, use_value_tokens=False),
                ),
                value_expert=vam_config.ValueExpertConfig(
                    n_bins=256,
                    vmin=0.0,
                    vmax=1.0,
                ),
                action_horizon=100,
                td_learning=True,
                gamma=0.9997,
            )
        ),
        data=_aloha_async_recap_data(
            repo_id="example/online_action",
            sampler=ALOHA_ASYNC_ACTION_SAMPLER,
            use_next_obs=True,
            observation_stride=1,
        ),
        batch_size=64,
        num_workers=24,
        prefetch_size=16,
        device_prefetch=True,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        startup_previews=False,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=1.0e-5,
            decay_steps=1_000_000,
            decay_lr=1.0e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.TDLearningWeightLoader(
            base_params_path=_ALOHA_ASYNC_BASE_PARAMS,
        ),
        num_train_steps=20_000,
        train_type={"td_learning", "bc"},
        filter_action_loss_by_success=False,
        freeze_filter=model.get_freeze_filter(freeze_value=False),
        validation=ValidationConfig(enabled=False),
        async_training=AsyncTrainingConfig(
            dataset_view=DatasetViewConfig(
                latest_path="./cloud_root/dataset_views/latest.json",
                poll_interval_steps=100,
            ),
            checkpoint_publisher=CheckpointPublisherConfig(
                output_dir="./cloud_root",
                policy_id=0,
                increment_policy_id=True,
                copy_checkpoint=True,
                reload_after_publish=True,
            ),
            gpu_monitor=GpuUtilizationMonitorConfig(
                enabled=True,
                sample_interval_s=5.0,
                target_percent=90.0,
                warmup_steps=1,
            ),
            streams=(
                AsyncTrainStreamConfig(
                    name="bc_action",
                    train_type=("bc",),
                    data=_aloha_async_recap_data(
                        repo_id="example/online_action",
                        sampler=ALOHA_ASYNC_ACTION_SAMPLER,
                        use_next_obs=False,
                        # Threefold action-data downsampling: ten retained rows span about one second.
                        observation_stride=10,
                    ),
                    dataset_view=DatasetViewConfig(
                        latest_path="./cloud_root/dataset_views/action_latest.json",
                        poll_interval_steps=100,
                    ),
                    steps_per_cycle=1,
                ),
                AsyncTrainStreamConfig(
                    name="td_value",
                    train_type=("td_learning",),
                    data=_aloha_async_recap_data(
                        repo_id="example/online_value",
                        sampler=ALOHA_ASYNC_VALUE_SAMPLER,
                        # Model is pi06, but this TD value stream trains no BC objective and its value repo has
                        # no `advantage` column, so override the pi06-inherited default and keep it disabled.
                        include_advantage=False,
                        use_next_obs=True,
                        observation_stride=1,
                    ),
                    dataset_view=DatasetViewConfig(
                        latest_path="./cloud_root/dataset_views/value_latest.json",
                        poll_interval_steps=100,
                    ),
                    steps_per_cycle=1,
                ),
            ),
        ),
        fsdp_devices=2,
    ),
]

_ALOHA_ASYNC_RECAP_DUMMY_CONFIG = next(config for config in _CONFIGS if config.name == "pi05_aloha_async_recap_dummy")
_ALOHA_ASYNC_RECAP_RUN = _required_environment_value("RUN_NAME")
_ALOHA_ASYNC_RECAP_CLOUD_ROOT = _required_environment_value("CLOUD_ROOT")
_ALOHA_ASYNC_RECAP_CHECKPOINT_BASE_DIR = _required_environment_value("CHECKPOINT_BASE_DIR")
_ALOHA_ASYNC_RECAP_SPLIT_RUN = _required_environment_value("RUN_NAME")
_ALOHA_ASYNC_RECAP_SPLIT_BASE = _required_environment_value("SPLIT_ROOT")
# Split closed-loop budget. Blocking on rollout/convert does not consume training steps. The launcher must provide both
# values explicitly; for example, 801 total steps includes boundaries at steps 200, 400, 600, and 800.
_ALOHA_ASYNC_RECAP_SPLIT_ROUND_STEPS = _required_environment_int("SPLIT_ROUND_STEPS")
_ALOHA_ASYNC_RECAP_SPLIT_NUM_TRAIN_STEPS = _required_environment_int("NUM_TRAIN_STEPS")
_ALOHA_ASYNC_RECAP_SPLIT_CLOUD_ROOT = _required_environment_value("CLOUD_ROOT")
_ALOHA_ASYNC_RECAP_SPLIT_MERGED_DIR = os.environ.get("MERGED_CKPT_ROOT") or (
    f"{_ALOHA_ASYNC_RECAP_SPLIT_BASE}/checkpoints/pi05_aloha_async_recap_split_merged/{_ALOHA_ASYNC_RECAP_SPLIT_RUN}"
    if isinstance(_ALOHA_ASYNC_RECAP_SPLIT_BASE, str) and isinstance(_ALOHA_ASYNC_RECAP_SPLIT_RUN, str)
    else tyro.MISSING
)


_ALOHA_ASYNC_RECAP_CONFIG = dataclasses.replace(
    _ALOHA_ASYNC_RECAP_DUMMY_CONFIG,
    name="pi05_aloha_async_recap",
    exp_name=_ALOHA_ASYNC_RECAP_RUN,
    checkpoint_base_dir=_ALOHA_ASYNC_RECAP_CHECKPOINT_BASE_DIR,
    num_train_steps=1_000_000,
    batch_size=32,
    num_workers=0,
    prefetch_size=1,
    prefetch_factor=None,
    persistent_workers=False,
    save_interval=500,
    keep_period=500,
    log_interval=50,
    fsdp_devices=2,
    async_training=dataclasses.replace(
        _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training,
        publish_on_save_interval=True,
        compute_burst_steps=2,
        checkpoint_publisher=dataclasses.replace(
            _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training.checkpoint_publisher,
            output_dir=_ALOHA_ASYNC_RECAP_CLOUD_ROOT,
            policy_id=0,
            increment_policy_id=True,
            copy_checkpoint=False,
            reload_after_publish=True,
            async_publish=False,
        ),
        checkpoint_switch=dataclasses.replace(
            _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training.checkpoint_switch,
            reset_optimizer=True,
            reset_ema=True,
        ),
        streams=(
            dataclasses.replace(
                _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training.streams[0],
                dataset_view=DatasetViewConfig(
                    latest_path=(
                        f"{_ALOHA_ASYNC_RECAP_CLOUD_ROOT}/dataset_views/action_latest.json"
                        if isinstance(_ALOHA_ASYNC_RECAP_CLOUD_ROOT, str)
                        else tyro.MISSING
                    ),
                    poll_interval_steps=100,
                ),
            ),
            dataclasses.replace(
                _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training.streams[1],
                dataset_view=DatasetViewConfig(
                    latest_path=(
                        f"{_ALOHA_ASYNC_RECAP_CLOUD_ROOT}/dataset_views/value_latest.json"
                        if isinstance(_ALOHA_ASYNC_RECAP_CLOUD_ROOT, str)
                        else tyro.MISSING
                    ),
                    poll_interval_steps=100,
                ),
            ),
        ),
    ),
)
_CONFIGS.append(_ALOHA_ASYNC_RECAP_CONFIG)


def _aloha_async_recap_split_config(*, role: Literal["action", "value"], stream_index: int) -> TrainConfig:
    base_config = _ALOHA_ASYNC_RECAP_DUMMY_CONFIG
    stream = base_config.async_training.streams[stream_index]
    stream_data = dataclasses.replace(
        stream.data,
        repo_id=None,
        repo_ids=None,
        assets=ALOHA_ASYNC_RECAP_NORM_ASSETS,
        base_config=stream.data.base_config or DataConfig(),
    )
    latest_name = "action_latest.json" if role == "action" else "value_latest.json"
    # Poll every training step so fresh live dataset views are picked up with minimal delay.
    dataset_view_poll_interval_steps = 1
    model = dataclasses.replace(base_config.model, td_learning=False) if role == "action" else base_config.model
    weight_loader: weight_loaders.WeightLoader = (
        weight_loaders.CheckpointWeightLoader(params_path=_ALOHA_ASYNC_BASE_PARAMS)
        if role == "action"
        else weight_loaders.TDLearningWeightLoader(base_params_path=_ALOHA_ASYNC_BASE_PARAMS)
    )
    runtime = (
        {
            "batch_size": 64,
            "num_workers": 16,
            "prefetch_size": 8,
            "prefetch_factor": 8,
        }
        if role == "action"
        else {
            "batch_size": 32,
            "num_workers": 32,
            "prefetch_size": 24,
            "prefetch_factor": 24,
        }
    )
    return dataclasses.replace(
        base_config,
        name=f"pi05_aloha_async_recap_split_{role}",
        project_name="online-split",
        exp_name=_ALOHA_ASYNC_RECAP_SPLIT_RUN,
        checkpoint_base_dir=cast(
            str,
            f"{_ALOHA_ASYNC_RECAP_SPLIT_BASE}/checkpoints"
            if isinstance(_ALOHA_ASYNC_RECAP_SPLIT_BASE, str)
            else tyro.MISSING,
        ),
        model=model,
        weight_loader=weight_loader,
        freeze_filter=model.get_freeze_filter(freeze_value=False),
        num_train_steps=_ALOHA_ASYNC_RECAP_SPLIT_NUM_TRAIN_STEPS,
        batch_size=runtime["batch_size"],
        num_workers=runtime["num_workers"],
        prefetch_size=runtime["prefetch_size"],
        device_prefetch=True,
        pin_memory=True,
        prefetch_factor=runtime["prefetch_factor"],
        persistent_workers=False,
        save_interval=_ALOHA_ASYNC_RECAP_SPLIT_ROUND_STEPS,
        keep_period=_ALOHA_ASYNC_RECAP_SPLIT_ROUND_STEPS,
        log_interval=5,
        fsdp_devices=2,
        train_type=set(stream.train_type),
        async_training=dataclasses.replace(
            base_config.async_training,
            publish_on_save_interval=False,
            split_round_steps=_ALOHA_ASYNC_RECAP_SPLIT_ROUND_STEPS,
            save_steps_after_view_update=None,
            max_publish_steps_after_view_update=None,
            save_final_checkpoint=False,
            dual_source_loader=True,
            compute_burst_steps=1,
            disable_persistent_workers_on_reload=True,
            checkpoint_publisher=dataclasses.replace(
                base_config.async_training.checkpoint_publisher,
                output_dir=_ALOHA_ASYNC_RECAP_SPLIT_CLOUD_ROOT,
                policy_id=0,
                increment_policy_id=False,
                copy_checkpoint=False,
                reload_after_publish=True,
                async_publish=False,
                merge=CheckpointMergeConfig(
                    role=role,
                    output_dir=_ALOHA_ASYNC_RECAP_SPLIT_MERGED_DIR,
                ),
            ),
            checkpoint_switch=dataclasses.replace(
                base_config.async_training.checkpoint_switch,
                reset_optimizer=True,
                reset_ema=False,
            ),
            streams=(
                dataclasses.replace(
                    stream,
                    data=stream_data,
                    dataset_view=DatasetViewConfig(
                        latest_path=(
                            f"{_ALOHA_ASYNC_RECAP_SPLIT_CLOUD_ROOT}/dataset_views/{latest_name}"
                            if isinstance(_ALOHA_ASYNC_RECAP_SPLIT_CLOUD_ROOT, str)
                            else tyro.MISSING
                        ),
                        poll_interval_steps=dataset_view_poll_interval_steps,
                    ),
                ),
            ),
        ),
    )


_CONFIGS.extend(
    [
        _aloha_async_recap_split_config(role="action", stream_index=0),
        _aloha_async_recap_split_config(role="value", stream_index=1),
    ]
)
_CONFIGS.append(
    dataclasses.replace(
        _ALOHA_ASYNC_RECAP_DUMMY_CONFIG,
        name="pi05_aloha_async_recap_streaming_validation",
        exp_name="async_recap_streaming_validation",
        num_train_steps=101,
        batch_size=2,
        num_workers=2,
        prefetch_size=2,
        prefetch_factor=2,
        persistent_workers=False,
        save_interval=1_000_000,
        keep_period=1_000_000,
        log_interval=10,
        wandb_enabled=False,
        startup_previews=False,
        async_training=dataclasses.replace(
            _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training,
            checkpoint_publisher=dataclasses.replace(
                _ALOHA_ASYNC_RECAP_DUMMY_CONFIG.async_training.checkpoint_publisher,
                output_dir="./cloud_root/streaming_validation",
                policy_id=0,
                increment_policy_id=True,
                reload_after_publish=False,
                async_publish=False,
            ),
            gpu_monitor=GpuUtilizationMonitorConfig(enabled=False),
            save_steps_after_view_update=100,
        ),
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

_ALOHA_ASYNC_CONFIG_NAMES = {
    "pi05_aloha_async_recap_dummy",
    "pi05_aloha_async_recap",
    "pi05_aloha_async_recap_split_action",
    "pi05_aloha_async_recap_split_value",
    "pi05_aloha_async_recap_streaming_validation",
}
_ALOHA_ASYNC_COMMON_REQUIRED_ENV = (
    "ALOHA_ASYNC_BASE_PARAMS",
    "ALOHA_ASYNC_OFFLINE_REPO_ID",
    "ALOHA_ASYNC_NORM_ASSETS_DIR",
)
_ALOHA_ASYNC_PRODUCTION_REQUIRED_ENV = ("RUN_NAME", "CLOUD_ROOT", "CHECKPOINT_BASE_DIR")
_ALOHA_ASYNC_SPLIT_REQUIRED_ENV = (
    "RUN_NAME",
    "SPLIT_ROOT",
    "CLOUD_ROOT",
    "SPLIT_ROUND_STEPS",
    "NUM_TRAIN_STEPS",
)


def _validate_aloha_async_environment(config: TrainConfig) -> TrainConfig:
    if config.name not in _ALOHA_ASYNC_CONFIG_NAMES:
        return config

    required = list(_ALOHA_ASYNC_COMMON_REQUIRED_ENV)
    if config.name == "pi05_aloha_async_recap":
        required.extend(_ALOHA_ASYNC_PRODUCTION_REQUIRED_ENV)
    elif config.name.startswith("pi05_aloha_async_recap_split_"):
        required.extend(_ALOHA_ASYNC_SPLIT_REQUIRED_ENV)
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Config {config.name!r} requires explicit environment variables; missing: {', '.join(missing)}. "
            "Launch through the corresponding async training script or export every value explicitly."
        )
    return config


def cli() -> TrainConfig:
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
    return _validate_aloha_async_environment(config)


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _validate_aloha_async_environment(_CONFIGS_DICT[config_name])
