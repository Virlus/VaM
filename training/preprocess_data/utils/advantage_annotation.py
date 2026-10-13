"""
Annotate a LeRobot dataset with per-frame advantage labels using a pretrained value expert.

Optimized v2 with:
  1. PyTorch DataLoader for multi-process data loading (replaces custom threaded reader).
  2. FSDP sharding to distribute model parameters across all GPUs (matches training).
  3. Double-buffered D2H transfer for maximum GPU utilization.

Pipeline:
  1. PyTorch DataLoader with num_workers prefetches transformed samples in batches.
  2. Prefetch thread moves batches to device ahead of GPU inference.
  3. Main thread runs FSDP-sharded batched model.predict_value to compute raw advantages.
  4. Writer thread handles D2H transfer and advantage computation.
  5. After all inference, apply a percentile threshold and write labels to parquet.
"""

import argparse
from collections.abc import Iterator
import dataclasses
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
import queue
import sys
import threading
import time
import typing

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm_loggable.auto as tqdm

from preprocess_data.utils.advantage_utils import calculate_episode_gae_advantages
from preprocess_data.utils.advantage_utils import calculate_episode_horizon_advantages
from preprocess_data.utils.advantage_utils import require_finite_next_obs_value
from preprocess_data.utils.lerobot_ops_common import ensure_annotation_column_writable
import vam.models.model as _model
import vam.shared.array_typing as at
import vam.shared.nnx_utils as nnx_utils
import vam.training.config as _config
import vam.training.data_loader as _data_loader
import vam.training.sharding as sharding
import vam.training.weight_loaders as _weight_loaders

logger = logging.getLogger("vam")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# FSDP-aware model loading
# ---------------------------------------------------------------------------


def load_model_fsdp(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    ckpt_path: str,
    mesh: jax.sharding.Mesh,
) -> nnx.Module:
    """Load model with FSDP sharding, distributing parameters across all devices.

    Approach:
      1. Create model on host and load checkpoint weights.
      2. Convert frozen params to bfloat16 (matches training behavior).
      3. Compute FSDP sharding specs from the model state's pure dict.
      4. Place parameters on devices with jax.device_put (committed shardings).

    Returns:
        model: The model with FSDP-sharded parameters across all devices.
    """
    _, model_rng = jax.random.split(init_rng)

    # Step 1: Create model and load checkpoint weights on host.
    model = config.model.create(model_rng)
    params = nnx.state(model)

    loaded_params = _weight_loaders.CheckpointWeightLoader(ckpt_path).load(params.to_pure_dict())
    try:
        at.check_pytree_equality(
            expected=params.to_pure_dict(), got=loaded_params, check_shapes=True, check_dtypes=True
        )
    except ValueError as e:
        logging.warning(f"Weights validation warning: {e}")

    def has_shape_dtype_struct(x) -> bool:
        return any(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree_util.tree_leaves(x))

    loaded_params = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not has_shape_dtype_struct(v)}
    )

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)

    # Step 2: Convert frozen params to bfloat16 (matches training).
    params = nnx.state(model)
    params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)

    # Step 3: Compute FSDP sharding from the pure dict (nested dict of arrays).
    graphdef, state = nnx.split(model)
    pure_dict = state.to_pure_dict()
    shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), pure_dict)
    dict_sharding = sharding.fsdp_sharding(shapes, mesh, log=True)

    # Step 4: Place parameters on devices with FSDP sharding.
    sharded_dict = jax.device_put(pure_dict, dict_sharding)
    state.replace_by_pure_dict(sharded_dict)
    model = nnx.merge(graphdef, state)

    return model


# ---------------------------------------------------------------------------
# JIT-compiled predict_value with FSDP sharding
# ---------------------------------------------------------------------------


def make_sharded_predict_value(
    model: nnx.Module,
    mesh: jax.sharding.Mesh,
):
    """Create a JIT-compiled predict_value with proper FSDP sharding.

    Unlike module_jit (which has no sharding), this leverages committed arrays:
    the model state has FSDP shardings from jax.device_put, and the observation
    data has DATA_AXIS sharding from jax.make_array_from_process_local_data.
    JAX infers the shardings automatically from these committed arrays.

    The mesh must be set (via sharding.set_mesh) during the first call so that
    activation_sharding_constraint inside the model can apply properly.

    Args:
        model: The model with FSDP-sharded parameters (from load_model_fsdp).
        mesh: The FSDP mesh (needed for activation_sharding_constraint).
    """
    graphdef, state = nnx.split(model)

    def _predict_value(state: nnx.State, observation: _model.Observation):
        module = nnx.merge(graphdef, state)
        return module.predict_value(observation)

    # JAX infers in/out shardings from committed arrays (state + observation).
    jitted_fn = jax.jit(_predict_value)

    def wrapper(observation: _model.Observation):
        # set_mesh is needed during the first call (JIT tracing) so that
        # activation_sharding_constraint inside the model can resolve the mesh.
        with sharding.set_mesh(mesh):
            return jitted_fn(state, observation)

    return wrapper


# ---------------------------------------------------------------------------
# PyTorch DataLoader setup
# ---------------------------------------------------------------------------


def _collate_fn(items):
    """Collate batch elements into batched numpy arrays (from data_loader.py)."""
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate GPU memory."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _set_data_factory_use_next_obs(data_factory: object, use_next_obs: bool) -> object:
    if not dataclasses.is_dataclass(data_factory):
        return data_factory

    field_names = {field.name for field in dataclasses.fields(data_factory)}
    replacements = {}
    if "use_next_obs" in field_names:
        replacements["use_next_obs"] = use_next_obs
    if "base_factory" in field_names:
        replacements["base_factory"] = _set_data_factory_use_next_obs(data_factory.base_factory, use_next_obs)
    if not replacements:
        return data_factory
    return dataclasses.replace(data_factory, **replacements)


def create_inference_dataloader(
    config: _config.TrainConfig,
    inference_batch_size: int,
    num_workers: int = 16,
    pin_memory: bool = True,
    prefetch_factor: int = 8,
    persistent_workers: bool = True,
    use_next_obs: bool = True,
) -> tuple[torch.utils.data.DataLoader, lerobot_dataset.LeRobotDataset]:
    """Create a PyTorch DataLoader for sequential inference.

    Returns:
        (dataloader, raw_dataset): The DataLoader and the underlying raw LeRobot dataset
                                   (needed for episode_data_index).
    """
    # Enable use_next_obs before DataConfig creation so repack/model transforms include
    # next-observation masks, tokenization, and state/action padding.
    loader_config = dataclasses.replace(
        config,
        data=_set_data_factory_use_next_obs(config.data, use_next_obs),
    )
    data_config = loader_config.data.create(loader_config.assets_dirs, loader_config.model)

    action_horizon = (
        loader_config.model.async_action_horizon
        if getattr(loader_config.model, "async_action_horizon", -1) > 0
        else loader_config.model.action_horizon
    )

    # Create raw + transformed dataset (same as training pipeline).
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, loader_config.model)
    transformed_dataset = _data_loader.transform_dataset(dataset, data_config)

    # Get the underlying raw LeRobot dataset for episode_data_index.
    raw_dataset = dataset
    while hasattr(raw_dataset, "_dataset"):
        raw_dataset = raw_dataset._dataset

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    dataloader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, transformed_dataset),
        batch_size=inference_batch_size,
        shuffle=False,  # Sequential order to preserve episode mapping.
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,  # Process every frame.
        generator=generator,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return dataloader, raw_dataset


# ---------------------------------------------------------------------------
# Prefetch iterator (moves batches to device on background thread)
# ---------------------------------------------------------------------------


def _pad_batch(batch: dict, target_size: int) -> tuple[dict, int]:
    """Pad a batch to target_size by repeating the last element.

    Returns (padded_batch, actual_size) so the caller knows how many
    real samples are in the batch.
    """
    # Pick any leaf to determine current batch size.
    first_leaf = jax.tree.leaves(batch)[0]
    actual_size = first_leaf.shape[0]
    if actual_size >= target_size:
        return batch, actual_size

    pad_count = target_size - actual_size

    def _pad(x):
        if isinstance(x, np.ndarray) and x.ndim >= 1:
            return np.concatenate([x, np.repeat(x[-1:], pad_count, axis=0)], axis=0)
        return x

    return jax.tree.map(_pad, batch), actual_size


def _prefetch(
    iterator: Iterator,
    prefetch_size: int,
    data_sharding: jax.sharding.Sharding,
    batch_size: int,
) -> Iterator:
    """Prefetch batches on a background thread, transferring them to device.

    Incomplete batches (e.g. the last batch) are padded to ``batch_size``
    with repeated copies of their last element **before** being placed on
    devices so that ``jax.make_array_from_process_local_data`` always sees a
    batch dimension divisible by the mesh size.

    Yields ``(sharded_batch, actual_size)`` tuples.
    """

    def _process(batch):
        padded, actual = _pad_batch(batch, batch_size)
        sharded = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), padded)
        return sharded, actual

    if prefetch_size <= 0:
        for batch in iterator:
            yield _process(batch)
        return

    sentinel = object()
    batch_queue: queue.Queue = queue.Queue(maxsize=prefetch_size)

    def _worker():
        for batch in iterator:
            batch_queue.put(_process(batch))
        batch_queue.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = batch_queue.get()
        if item is sentinel:
            break
        yield item


# ---------------------------------------------------------------------------
# Stage 2: Batched inference with FSDP
# ---------------------------------------------------------------------------


def _distribution_to_value(distribution: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.sum(support * distribution, axis=-1).reshape(-1)


def _require_finite_next_obs_value(next_obs_value: float, frame_idx: int, *, method: str) -> None:
    require_finite_next_obs_value(next_obs_value, frame_idx, method=method)


def _calculate_episode_horizon_advantages(
    *,
    all_pred_values: np.ndarray,
    all_next_obs_values: np.ndarray,
    all_data_sources: np.ndarray,
    all_terminates: np.ndarray,
    ep_starts: list[int],
    ep_ends: list[int],
    advantage_obs_horizon: int,
) -> np.ndarray:
    """Compute advantage from current next_obs plus later episode observations."""
    return calculate_episode_horizon_advantages(
        all_pred_values=all_pred_values,
        all_next_obs_values=all_next_obs_values,
        all_data_sources=all_data_sources,
        all_terminates=all_terminates,
        ep_starts=ep_starts,
        ep_ends=ep_ends,
        advantage_obs_horizon=advantage_obs_horizon,
    )


def _calculate_episode_gae_advantages(
    *,
    all_pred_values: np.ndarray,
    all_next_obs_values: np.ndarray,
    all_data_sources: np.ndarray,
    all_reward_sums: np.ndarray,
    all_terminates: np.ndarray,
    ep_starts: list[int],
    ep_ends: list[int],
    advantage_obs_horizon: int,
    gamma: float,
    gae_lambda: float,
    action_horizon: int,
    action_step_frames: int,
) -> np.ndarray:
    """Compute chunk-level GAE by stepping from each obs to its next_obs frame.

    Each TD residual uses the current frame and the model-predicted value for that frame's
    next_obs. The next residual starts from the dataset frame aligned with that next_obs,
    i.e. ``idx + action_step_frames`` for 100 Hz action chunks in a 30 Hz LeRobot dataset.
    """
    return calculate_episode_gae_advantages(
        all_pred_values=all_pred_values,
        all_next_obs_values=all_next_obs_values,
        all_data_sources=all_data_sources,
        all_reward_sums=all_reward_sums,
        all_terminates=all_terminates,
        ep_starts=ep_starts,
        ep_ends=ep_ends,
        advantage_obs_horizon=advantage_obs_horizon,
        gamma=gamma,
        gae_lambda=gae_lambda,
        action_horizon=action_horizon,
        action_step_frames=action_step_frames,
    )


def _column_to_1d_float(table: pa.Table, key: str, *, episode_idx: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} not found in episode {episode_idx}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in table.column(key).to_pylist()]
    if not values:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([value[0] if len(value) else np.nan for value in values], dtype=np.float64)


def _column_to_2d_float(table: pa.Table, key: str, *, episode_idx: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} not found in episode {episode_idx}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in table.column(key).to_pylist()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = len(values[0])
    if any(len(value) != width for value in values):
        raise ValueError(f"Column {key!r} in episode {episode_idx} has variable-length rows")
    return np.stack(values, axis=0)


def _optional_terminate_column(table: pa.Table, *, episode_idx: int, length: int) -> np.ndarray:
    if "terminate" not in table.column_names:
        return np.zeros(length, dtype=np.float64)
    terminates = _column_to_1d_float(table, "terminate", episode_idx=episode_idx)
    if len(terminates) != length:
        raise ValueError(
            f"Episode {episode_idx} has inconsistent row counts: terminate={len(terminates)}, expected={length}"
        )
    return terminates


def _episode_idx_from_parquet_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _load_dataset_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    return float(info.get("fps", 30))


def calculate_advantage_from_value_predictions(
    dataset_root: Path,
    *,
    percentile: float,
    gamma: float,
    action_horizon: int,
    advantage_obs_horizon: int | None,
    advantage_method: str,
    gae_lambda: float,
    advantage_action_fps: float,
    overwrite: bool,
) -> tuple[dict[int, list[float]], dict[int, list[int]]]:
    """Compute advantages directly from existing per-frame value_prediction parquet columns."""
    if advantage_obs_horizon is None:
        raise ValueError("--advantage-source=value_prediction requires --advantage-obs-horizon")

    parquet_files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}/data/")

    ensure_annotation_column_writable(dataset_root, "advantage", overwrite=overwrite)

    dataset_fps = _load_dataset_fps(dataset_root)
    action_step_frames = max(1, round(action_horizon * dataset_fps / advantage_action_fps))
    discount_weights = np.array([gamma**t for t in range(action_horizon)], dtype=np.float64)

    all_pred_values: list[np.ndarray] = []
    all_next_obs_values: list[np.ndarray] = []
    all_data_sources: list[np.ndarray] = []
    all_reward_sums: list[np.ndarray] = []
    all_terminates: list[np.ndarray] = []
    ep_starts: list[int] = []
    ep_ends: list[int] = []
    episode_indices: list[int] = []

    cursor = 0
    for parquet_file in tqdm.tqdm(parquet_files, desc="Reading value_prediction", dynamic_ncols=True):
        ep_idx = _episode_idx_from_parquet_path(parquet_file)
        table = pq.ParquetFile(parquet_file).read()
        pred_values = _column_to_1d_float(table, "value_prediction", episode_idx=ep_idx)
        data_sources = _column_to_1d_float(table, "data_source", episode_idx=ep_idx)
        rewards = _column_to_2d_float(table, "reward", episode_idx=ep_idx)
        terminates = _optional_terminate_column(table, episode_idx=ep_idx, length=len(pred_values))

        if rewards.shape[1] != action_horizon:
            raise ValueError(
                f"Episode {ep_idx} reward horizon is {rewards.shape[1]}, but config action_horizon is {action_horizon}."
            )
        if not (len(pred_values) == len(data_sources) == len(rewards)):
            raise ValueError(
                f"Episode {ep_idx} has inconsistent row counts: "
                f"value_prediction={len(pred_values)}, data_source={len(data_sources)}, reward={len(rewards)}"
            )

        next_indices = np.minimum(np.arange(len(pred_values)) + action_step_frames, max(len(pred_values) - 1, 0))
        next_obs_values = pred_values[next_indices] if len(pred_values) else pred_values

        episode_indices.append(ep_idx)
        ep_starts.append(cursor)
        cursor += len(pred_values)
        ep_ends.append(cursor)
        all_pred_values.append(pred_values)
        all_next_obs_values.append(next_obs_values)
        all_data_sources.append(data_sources.astype(np.int32))
        all_reward_sums.append(rewards @ discount_weights)
        all_terminates.append(terminates)

    flat_pred_values = np.concatenate(all_pred_values, axis=0)
    flat_next_obs_values = np.concatenate(all_next_obs_values, axis=0)
    flat_data_sources = np.concatenate(all_data_sources, axis=0)
    flat_reward_sums = np.concatenate(all_reward_sums, axis=0)
    flat_terminates = np.concatenate(all_terminates, axis=0)

    logger.info(
        "Computing %s advantage from value_prediction: advantage_obs_horizon=%s, dataset_fps=%g, "
        "advantage_action_fps=%g, action_step_frames=%d",
        advantage_method,
        advantage_obs_horizon,
        dataset_fps,
        advantage_action_fps,
        action_step_frames,
    )
    if advantage_method == "value_avg":
        flat_advantages = _calculate_episode_horizon_advantages(
            all_pred_values=flat_pred_values,
            all_next_obs_values=flat_next_obs_values,
            all_data_sources=flat_data_sources,
            all_terminates=flat_terminates,
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            advantage_obs_horizon=advantage_obs_horizon,
        )
    elif advantage_method == "gae":
        flat_advantages = _calculate_episode_gae_advantages(
            all_pred_values=flat_pred_values,
            all_next_obs_values=flat_next_obs_values,
            all_data_sources=flat_data_sources,
            all_reward_sums=flat_reward_sums,
            all_terminates=flat_terminates,
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            advantage_obs_horizon=advantage_obs_horizon,
            gamma=gamma,
            gae_lambda=gae_lambda,
            action_horizon=action_horizon,
            action_step_frames=action_step_frames,
        )
    else:
        raise ValueError(f"Unsupported advantage_method: {advantage_method}")

    episode_advantages: dict[int, list[float]] = {}
    episode_data_sources: dict[int, list[int]] = {}
    for ep_idx, start, end in zip(episode_indices, ep_starts, ep_ends, strict=True):
        episode_advantages[ep_idx] = flat_advantages[start:end].tolist()
        episode_data_sources[ep_idx] = flat_data_sources[start:end].tolist()

    logger.info(
        "Computed value_prediction advantages for %d episodes at p%s threshold source.",
        len(episode_advantages),
        percentile,
    )
    return episode_advantages, episode_data_sources


def calculate_advantage(
    predict_value_fn,
    dataloader: torch.utils.data.DataLoader,
    raw_dataset: lerobot_dataset.LeRobotDataset,
    n_bins: int,
    vmin: float,
    vmax: float,
    gamma: float,
    action_horizon: int,
    data_sharding: jax.sharding.Sharding,
    prefetch_size: int = 16,
    advantage_obs_horizon: int | None = None,
    advantage_method: str = "value_avg",
    gae_lambda: float = 0.95,
    advantage_action_fps: float = 100.0,
) -> tuple[dict[int, list[float]], dict[int, list[int]]]:
    """Run batched value inference with PyTorch DataLoader + FSDP sharding.

    Pipeline:
      - PyTorch DataLoader: loads data with multi-process workers.
      - Prefetch thread: moves batches to device (H2D).
      - Main thread: dispatches predict_value (async GPU).
      - Writer thread: handles D2H sync + advantage computation.

    Advantage calculation:
      - Offline data (data_source == 0): advantage = 1.0 (fixed positive)
      - Online data (data_source == 1):
        - if advantage_obs_horizon is None:
          advantage = sum_{t=0}^{H-1}(gamma^t * r_t) + (1 - terminate) * gamma^H * V(next_obs) - V(obs)
        - otherwise, with advantage_method == "value_avg":
          advantage = mean(V(next_obs_t), V(obs_{t+1}), ..., V(obs_{t+advantage_obs_horizon})) - V(obs_t)
        - otherwise, with advantage_method == "gae":
          advantage = sum_{l=0}^{advantage_obs_horizon-1}(gamma^H * lambda)^l * delta_{t+l}
          where delta_{t+l} jumps by the dataset-frame offset aligned with next_obs.

    Returns:
        episode_advantages: dict mapping episode_idx -> list of per-frame advantage floats.
        episode_data_sources: dict mapping episode_idx -> list of per-frame data_source ints (0=offline, 1=online).
    """
    episode_data_index = raw_dataset.episode_data_index
    num_episodes = raw_dataset.num_episodes
    total_frames = len(raw_dataset)
    support = np.linspace(vmin, vmax, n_bins)

    # Pre-compute episode boundaries for post-inference splitting.
    ep_starts = [int(episode_data_index["from"][i].item()) for i in range(num_episodes)]
    ep_ends = [int(episode_data_index["to"][i].item()) for i in range(num_episodes)]

    # Precompute per-timestep discount weights: [gamma^0, gamma^1, ..., gamma^(H-1)]
    discount_weights = np.array([gamma**t for t in range(action_horizon)], dtype=np.float64)

    # Flat advantage array: filled during inference, split by episode afterward.
    all_advantages = np.zeros(total_frames, dtype=np.float64)

    # Flat data_source array: tracks offline/online for each frame.
    all_data_sources = np.zeros(total_frames, dtype=np.int32)
    all_pred_values = np.full(total_frames, np.nan, dtype=np.float64)
    all_next_obs_values = np.full(total_frames, np.nan, dtype=np.float64)
    all_reward_sums = np.zeros(total_frames, dtype=np.float64)
    all_terminates = np.zeros(total_frames, dtype=np.float64)

    # Writer thread: handles blocking D2H + CPU math.
    write_queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
    writer_error: list[BaseException] = []

    def _writer_consumer():
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                (
                    global_offset,
                    actual_size,
                    pred_future,
                    data_source_future,
                    reward_future,
                    terminate_future,
                    next_obs_pred_future,
                ) = item

                # Block on D2H transfer.
                pred_vals = np.asarray(pred_future)
                data_sources = np.asarray(data_source_future)
                rewards = np.asarray(reward_future)
                terminates = np.asarray(terminate_future) if terminate_future is not None else None
                next_obs_preds = np.asarray(next_obs_pred_future) if next_obs_pred_future is not None else None

                # Flatten data_source from (bsz, 1) to (bsz,)
                data_sources = data_sources.flatten()
                if terminates is None:
                    terminates = np.zeros_like(data_sources, dtype=np.float64)
                else:
                    terminates = terminates.flatten()

                # Collect data_source info for later thresholding.
                all_data_sources[global_offset : global_offset + actual_size] = data_sources[:actual_size]
                all_terminates[global_offset : global_offset + actual_size] = terminates[:actual_size]

                # Compute value from quantile distributions: (bsz, n_bins) -> (bsz,)
                pred_value = _distribution_to_value(pred_vals, support)
                all_pred_values[global_offset : global_offset + actual_size] = pred_value[:actual_size]
                next_obs_value = None
                if next_obs_preds is not None:
                    next_obs_value = _distribution_to_value(next_obs_preds, support)
                    all_next_obs_values[global_offset : global_offset + actual_size] = next_obs_value[:actual_size]

                # Discounted reward sum: sum_t gamma^t * r_t, (bsz, action_horizon) -> (bsz,)
                rewards_sum = rewards @ discount_weights
                all_reward_sums[global_offset : global_offset + actual_size] = rewards_sum[:actual_size]

                if advantage_obs_horizon is not None:
                    continue

                # Initialize advantages array.
                advantages = np.zeros_like(data_sources, dtype=np.float64)

                # Create masks for offline and online data.
                offline_mask = data_sources < 0.5
                online_mask = ~offline_mask

                # Offline data (data_source < 0.5): advantage = 1 (fixed positive)
                advantages[offline_mask] = 1.0

                # Online data (data_source >= 0.5):
                #   advantage = sum_t(gamma^t * r_t) + (1 - terminate) * gamma^H * V(next_obs) - V(obs)
                if online_mask.any() and next_obs_value is not None:
                    discount = gamma**action_horizon
                    bootstrap_mask = 1.0 - (terminates >= 0.5).astype(np.float64)
                    advantages[online_mask] = (
                        rewards_sum[online_mask]
                        + discount * bootstrap_mask[online_mask] * next_obs_value[online_mask]
                        - pred_value[online_mask]
                    )

                # Write into flat array.
                all_advantages[global_offset : global_offset + actual_size] = advantages[:actual_size]
        except Exception as e:
            writer_error.append(e)

    writer_thread = threading.Thread(target=_writer_consumer, daemon=True)
    writer_thread.start()

    # Main inference loop.
    processed = 0
    t0 = time.perf_counter()
    global_offset = 0
    inference_batch_size = dataloader.batch_size

    data_iter = _prefetch(iter(dataloader), prefetch_size, data_sharding, inference_batch_size)

    for batch, actual_size in tqdm.tqdm(
        data_iter,
        total=(total_frames + inference_batch_size - 1) // inference_batch_size,
        desc="Inference",
        dynamic_ncols=True,
    ):
        batch_obs = _model.Observation.from_dict(batch)
        batch_data_source = batch["data_source"]
        batch_reward = batch["reward"]
        batch_terminate = batch.get("terminate")

        # Build next_obs dict from batch keys starting with "next_"
        next_obs_dict = {k[5:]: v for k, v in batch.items() if k.startswith("next_")}
        has_next_obs = len(next_obs_dict) > 0

        # Dispatch inference for current obs (async on GPU).
        pred = predict_value_fn(batch_obs)

        # For online data, also dispatch inference for next_obs.
        next_obs_pred = None
        if has_next_obs:
            next_obs = _model.Observation.from_dict(next_obs_dict)
            next_obs_pred = predict_value_fn(next_obs)

        # Send to writer thread for async D2H.
        write_queue.put(
            (
                global_offset,
                actual_size,
                pred,
                batch_data_source,
                batch_reward,
                batch_terminate,
                next_obs_pred,
            )
        )

        global_offset += actual_size
        processed += actual_size

    # Signal writer to finish.
    write_queue.put(None)
    writer_thread.join()

    if writer_error:
        raise RuntimeError("Writer thread failed") from writer_error[0]

    elapsed = time.perf_counter() - t0
    logger.info(f"Inference done: {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.1f} fps)")

    if advantage_obs_horizon is not None:
        if advantage_method == "value_avg":
            logger.info(
                "Computing value_avg advantage with episode observation horizon: "
                f"advantage_obs_horizon={advantage_obs_horizon}"
            )
            all_advantages = _calculate_episode_horizon_advantages(
                all_pred_values=all_pred_values,
                all_next_obs_values=all_next_obs_values,
                all_data_sources=all_data_sources,
                all_terminates=all_terminates,
                ep_starts=ep_starts,
                ep_ends=ep_ends,
                advantage_obs_horizon=advantage_obs_horizon,
            )
        elif advantage_method == "gae":
            dataset_fps = float(raw_dataset.fps)
            action_step_frames = max(1, round(action_horizon * dataset_fps / advantage_action_fps))
            logger.info(
                "Computing GAE advantage: "
                f"advantage_obs_horizon={advantage_obs_horizon}, gae_lambda={gae_lambda}, "
                f"transition_discount=gamma^action_horizon={gamma**action_horizon:.6f}, "
                f"dataset_fps={dataset_fps:g}, advantage_action_fps={advantage_action_fps:g}, "
                f"action_step_frames={action_step_frames}"
            )
            all_advantages = _calculate_episode_gae_advantages(
                all_pred_values=all_pred_values,
                all_next_obs_values=all_next_obs_values,
                all_data_sources=all_data_sources,
                all_reward_sums=all_reward_sums,
                all_terminates=all_terminates,
                ep_starts=ep_starts,
                ep_ends=ep_ends,
                advantage_obs_horizon=advantage_obs_horizon,
                gamma=gamma,
                gae_lambda=gae_lambda,
                action_horizon=action_horizon,
                action_step_frames=action_step_frames,
            )
        else:
            raise ValueError(f"Unsupported advantage_method: {advantage_method}")

    # Split flat advantages and data_sources into per-episode lists.
    episode_advantages: dict[int, list[float]] = {}
    episode_data_sources: dict[int, list[int]] = {}
    for ep_idx in range(num_episodes):
        start, end = ep_starts[ep_idx], ep_ends[ep_idx]
        episode_advantages[ep_idx] = all_advantages[start:end].tolist()
        episode_data_sources[ep_idx] = all_data_sources[start:end].tolist()

    return episode_advantages, episode_data_sources


# ---------------------------------------------------------------------------
# Stage 3: Percentile labeling + PyArrow write (reused from v1)
# ---------------------------------------------------------------------------


def write_advantages_to_parquet(
    dataset_root: Path,
    episode_advantages: dict[int, list[float]],
    episode_data_sources: dict[int, list[int]],
    percentile: float,
    *,
    overwrite: bool = False,
    force_control_flag_one_positive: bool = False,
):
    """Apply percentile threshold to online advantages and write float32 labels to parquet.

    Each frame gets:
      - Offline (data_source == 0): 1.0 (fixed positive)
      - Online (data_source == 1): 1.0 if advantage >= threshold, else 0.0
      - Optionally, control_flag == 1: 1.0, overriding the label above

    Threshold is computed only from online data.
    """
    ensure_annotation_column_writable(dataset_root, "advantage", overwrite=overwrite)

    # Separate offline and online advantages, compute threshold from online only.
    online_advs = []
    offline_advs = []
    offline_count = 0
    online_count = 0
    non_finite_examples = []

    for ep_idx, advs in episode_advantages.items():
        data_sources = episode_data_sources[ep_idx]
        for frame_idx, (a, ds) in enumerate(zip(advs, data_sources, strict=True)):
            if not np.isfinite(a) and len(non_finite_examples) < 10:
                non_finite_examples.append((ep_idx, frame_idx, float(a), int(ds)))
            if ds < 0.5:  # offline
                offline_count += 1
                offline_advs.append(a)
            else:  # online
                online_count += 1
                online_advs.append(a)

    if non_finite_examples:
        raise ValueError(
            "Computed non-finite advantage values; refusing to write labels. "
            f"First examples (episode, frame, advantage, data_source): {non_finite_examples}"
        )

    if not online_advs:
        logger.warning("No online data found, setting threshold to -inf")
        threshold = float("-inf")
    else:
        threshold = float(np.percentile(online_advs, percentile))

    if offline_advs:
        logger.info(
            f"Advantage stats (offline: n={offline_count}): "
            f"min={min(offline_advs):.4f}, max={max(offline_advs):.4f}, mean={np.mean(offline_advs):.4f}"
        )
    else:
        logger.info("Advantage stats (offline: n=0): no offline frames")
    if online_advs:
        logger.info(
            f"Advantage stats (online: n={online_count}): "
            f"min={min(online_advs):.4f}, max={max(online_advs):.4f}, "
            f"mean={np.mean(online_advs):.4f}, p{percentile}={threshold:.4f}"
        )
    else:
        logger.info(f"Advantage stats (online: n=0): no online frames, p{percentile}={threshold:.4f}")

    parquet_files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}/data/")

    ep_to_file: dict[int, Path] = {}
    for pf in parquet_files:
        ep_idx = int(pf.stem.split("_")[1])
        ep_to_file[ep_idx] = pf

    modified = 0
    forced_positive_count = 0
    for ep_idx, advs in episode_advantages.items():
        pf = ep_to_file.get(ep_idx)
        if pf is None:
            logger.warning(f"No parquet file found for episode {ep_idx}, skipping")
            continue

        # NOTE: Parquet files can (unfortunately) contain duplicate field names.
        # pyarrow.parquet.read_table may fail in that case ("Can't unify schema with duplicate field names.").
        # ParquetFile.read() can still read such a file, so use it here.
        table = pq.ParquetFile(pf).read()
        if len(table) != len(advs):
            logger.warning(f"Episode {ep_idx}: parquet has {len(table)} rows but got {len(advs)} advantages, skipping")
            continue

        if force_control_flag_one_positive:
            control_flags = _column_to_1d_float(table, "control_flag", episode_idx=ep_idx)
            if len(control_flags) != len(advs):
                raise ValueError(
                    f"Episode {ep_idx} has inconsistent row counts: "
                    f"control_flag={len(control_flags)}, advantage={len(advs)}"
                )
            force_positive_mask = control_flags == 1.0
            forced_positive_count += int(np.count_nonzero(force_positive_mask))
        else:
            force_positive_mask = np.zeros(len(advs), dtype=bool)

        # If the file already has 'advantage' (possibly duplicated from a previous run),
        # drop all existing 'advantage' columns and replace with the newly computed one.
        if "advantage" in table.column_names:
            keep_idxs = [i for i, n in enumerate(table.column_names) if n != "advantage"]
            table = table.select(keep_idxs)

        # Generate labels: offline -> 1 (fixed), online -> 0/1 based on threshold
        data_sources = episode_data_sources[ep_idx]
        labels = []
        for a, ds, force_positive in zip(advs, data_sources, force_positive_mask, strict=True):
            if force_positive or ds < 0.5:  # intervention override or offline
                labels.append([np.float32(1.0)])
            else:  # online
                labels.append([np.float32(1.0)] if a >= threshold else [np.float32(0.0)])
        col = pa.array(labels, type=pa.list_(pa.float32()))
        table = table.append_column("advantage", col)
        pq.write_table(table, pf)
        modified += 1

    logger.info(f"Wrote advantage labels to {modified}/{len(episode_advantages)} episode parquet files")
    if force_control_flag_one_positive:
        logger.info("Forced advantage=1.0 for %d frames with control_flag=1", forced_positive_count)

    # Update meta/info.json
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if "features" in info and "advantage" not in info["features"]:
            info["features"]["advantage"] = {
                "dtype": "float32",
                "shape": [1],
                "names": ["advantage"],
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            logger.info("Added 'advantage' feature to meta/info.json")
    else:
        logger.warning(f"meta/info.json not found at {info_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_annotate_advantage_with_config(
    *,
    config: _config.TrainConfig,
    ckpt_path: str | None,
    percentile: float = 30.0,
    inference_batch_size: int = 64,
    num_workers: int = 16,
    prefetch_size: int = 16,
    prefetch_factor: int = 8,
    fsdp_devices: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    overwrite: bool = False,
    advantage_obs_horizon: int | None = None,
    advantage_method: str = "value_avg",
    gae_lambda: float = 0.95,
    advantage_action_fps: float = 100.0,
    advantage_source: str = "model",
    dataset_path: str | None = None,
    force_control_flag_one_positive: bool = False,
) -> None:
    if advantage_obs_horizon is not None and advantage_obs_horizon < 1:
        raise ValueError(f"advantage_obs_horizon must be >= 1 when set, got {advantage_obs_horizon}")
    if advantage_method not in {"value_avg", "gae"}:
        raise ValueError(f"advantage_method must be one of {{'value_avg', 'gae'}}, got {advantage_method!r}")
    if advantage_method == "gae" and advantage_obs_horizon is None:
        raise ValueError("--advantage-method=gae requires --advantage-obs-horizon")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
    if advantage_action_fps <= 0.0:
        raise ValueError(f"advantage_action_fps must be > 0, got {advantage_action_fps}")
    if advantage_source not in {"model", "value_prediction"}:
        raise ValueError(f"advantage_source must be one of {{'model', 'value_prediction'}}, got {advantage_source!r}")

    # Enable JAX compilation cache.
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Advantage annotation requires a single LeRobot repo_id.")
    if dataset_path is not None and advantage_source != "value_prediction":
        raise ValueError("--dataset-path is only supported with --advantage-source=value_prediction")
    dataset_root = (
        Path(dataset_path).expanduser().resolve()
        if dataset_path is not None
        else (lerobot_dataset.HF_LEROBOT_HOME / data_config.repo_id)
    )
    ensure_annotation_column_writable(dataset_root, "advantage", overwrite=overwrite)

    action_horizon = config.model.action_horizon

    if advantage_source == "value_prediction":
        logger.info("Using existing value_prediction column for advantage computation; skipping model inference.")
        episode_advantages, episode_data_sources = calculate_advantage_from_value_predictions(
            dataset_root,
            percentile=percentile,
            gamma=config.model.gamma,
            action_horizon=action_horizon,
            advantage_obs_horizon=advantage_obs_horizon,
            advantage_method=advantage_method,
            gae_lambda=gae_lambda,
            advantage_action_fps=advantage_action_fps,
            overwrite=overwrite,
        )
        logger.info("Writing advantage labels to parquet...")
        write_advantages_to_parquet(
            dataset_root,
            episode_advantages,
            episode_data_sources,
            percentile,
            overwrite=overwrite,
            force_control_flag_one_positive=force_control_flag_one_positive,
        )
        logger.info("Done.")
        return

    if ckpt_path is None:
        raise ValueError("--ckpt-path is required when --advantage-source=model")

    fsdp_devices = fsdp_devices if fsdp_devices is not None else config.fsdp_devices

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    logger.info(f"FSDP mesh: shape={mesh.shape}, devices={jax.device_count()}, fsdp_devices={fsdp_devices}")

    logger.info("Loading model with FSDP sharding...")
    model = load_model_fsdp(config, init_rng, ckpt_path, mesh)
    jax.block_until_ready(nnx.state(model))
    logger.info("Model loaded and distributed across devices.")

    logger.info("Compiling sharded predict_value...")
    predict_value_fn = make_sharded_predict_value(model, mesh)

    logger.info("Creating PyTorch DataLoader...")
    dataloader, raw_dataset = create_inference_dataloader(
        config,
        inference_batch_size=inference_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0 if persistent_workers is None else persistent_workers,
    )
    logger.info(
        f"DataLoader ready: {len(raw_dataset)} frames, batch_size={inference_batch_size}, "
        f"num_workers={num_workers}, prefetch_factor={prefetch_factor}"
    )

    if advantage_obs_horizon is None:
        logger.info(
            f"Running advantage inference (gamma={config.model.gamma}, action_horizon={action_horizon}, "
            f"discount=gamma^action_horizon={config.model.gamma**action_horizon:.6f})..."
        )
    else:
        logger.info(
            "Running advantage inference with episode observation horizon "
            f"(advantage_method={advantage_method}, advantage_obs_horizon={advantage_obs_horizon})..."
        )
    episode_advantages, episode_data_sources = calculate_advantage(
        predict_value_fn,
        dataloader,
        raw_dataset,
        n_bins=config.model.n_bins,
        vmin=config.model.vmin,
        vmax=config.model.vmax,
        gamma=config.model.gamma,
        action_horizon=action_horizon,
        data_sharding=data_sharding,
        prefetch_size=prefetch_size,
        advantage_obs_horizon=advantage_obs_horizon,
        advantage_method=advantage_method,
        gae_lambda=gae_lambda,
        advantage_action_fps=advantage_action_fps,
    )

    logger.info("Writing advantage labels to parquet...")
    write_advantages_to_parquet(
        dataset_root,
        episode_advantages,
        episode_data_sources,
        percentile,
        overwrite=overwrite,
        force_control_flag_one_positive=force_control_flag_one_positive,
    )

    logger.info("Done.")


def run_annotate_advantage(
    *,
    config_name: str,
    ckpt_path: str | None,
    percentile: float = 30.0,
    inference_batch_size: int = 64,
    num_workers: int = 16,
    prefetch_size: int = 16,
    prefetch_factor: int = 8,
    fsdp_devices: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    overwrite: bool = False,
    advantage_obs_horizon: int | None = None,
    advantage_method: str = "value_avg",
    gae_lambda: float = 0.95,
    advantage_action_fps: float = 100.0,
    advantage_source: str = "model",
    dataset_path: str | None = None,
) -> None:
    config = _config.get_config(config_name)
    run_annotate_advantage_with_config(
        config=config,
        ckpt_path=ckpt_path,
        percentile=percentile,
        inference_batch_size=inference_batch_size,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        prefetch_factor=prefetch_factor,
        fsdp_devices=fsdp_devices,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        overwrite=overwrite,
        advantage_obs_horizon=advantage_obs_horizon,
        advantage_method=advantage_method,
        gae_lambda=gae_lambda,
        advantage_action_fps=advantage_action_fps,
        advantage_source=advantage_source,
        dataset_path=dataset_path,
    )


def main(argv=None):
    init_logging()
    logger.info(f"Running on: {platform.node()}")

    parser = argparse.ArgumentParser(
        description="Annotate advantage labels on an existing LeRobot dataset (v2: FSDP + PyTorch DataLoader)"
    )
    parser.add_argument("config_name", type=str, help="Config name, e.g. pi05_aloha_history_value_expert")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to pretrained value-expert checkpoint params. Required when --advantage-source=model.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=70.0,
        help="Percentile threshold: frames with advantage >= this percentile get 'Positive'",
    )
    parser.add_argument("--inference-batch-size", type=int, default=64, help="Batch size for value inference")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of DataLoader worker processes")
    parser.add_argument(
        "--prefetch-size", type=int, default=16, help="Number of batches to prefetch on background thread"
    )
    parser.add_argument("--prefetch-factor", type=int, default=8, help="PyTorch DataLoader prefetch factor per worker")
    parser.add_argument(
        "--fsdp-devices", type=int, default=None, help="Number of FSDP devices (default: use config value)"
    )
    parser.add_argument(
        "--advantage-obs-horizon",
        type=int,
        default=None,
        help=(
            "If set, compute online advantage as the mean of V(next_obs) and the next N "
            "episode-observation values, minus V(current_obs). Default keeps the legacy formula."
        ),
    )
    parser.add_argument(
        "--advantage-method",
        choices=("value_avg", "gae"),
        default="value_avg",
        help=(
            "Method used when --advantage-obs-horizon is set. value_avg keeps the current future-value "
            "average behavior; gae computes chunk-level generalized advantage estimation."
        ),
    )
    parser.add_argument(
        "--advantage-source",
        choices=("model", "value_prediction"),
        default="model",
        help=(
            "Use model inference or existing value_prediction parquet columns. value_prediction approximates "
            "next_obs value with the value_prediction at the aligned future dataset frame."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local LeRobot dataset path override. Only supported with --advantage-source=value_prediction.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda used with --advantage-method=gae.",
    )
    parser.add_argument(
        "--advantage-action-fps",
        type=float,
        default=100.0,
        help=(
            "Action/control frequency for aligning next_obs to dataset frames in GAE. "
            "Default 100 matches 100 Hz action chunks."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing advantage column.")
    args = parser.parse_args(argv)

    run_annotate_advantage(
        config_name=args.config_name,
        ckpt_path=args.ckpt_path,
        percentile=args.percentile,
        inference_batch_size=args.inference_batch_size,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        prefetch_factor=args.prefetch_factor,
        fsdp_devices=args.fsdp_devices,
        pin_memory=True,
        persistent_workers=True,
        overwrite=args.overwrite,
        advantage_obs_horizon=args.advantage_obs_horizon,
        advantage_method=args.advantage_method,
        gae_lambda=args.gae_lambda,
        advantage_action_fps=args.advantage_action_fps,
        advantage_source=args.advantage_source,
        dataset_path=args.dataset_path,
    )


if __name__ == "__main__":
    main()
