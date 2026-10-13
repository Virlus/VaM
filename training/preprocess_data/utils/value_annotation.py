from __future__ import annotations

# ruff: noqa: I001

import argparse
import dataclasses
import logging
from pathlib import Path
import platform
import queue
import sys
import threading
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm_loggable.auto as tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import vam.models.model as _model
import vam.models.vam_config as vam_config
import vam.training.config as _config
import vam.training.data_loader as _data_loader
import vam.training.sharding as sharding
import vam.training.utils as training_utils
from preprocess_data.utils.lerobot_ops_common import ensure_dataset_exists
from preprocess_data.utils.lerobot_ops_common import ensure_annotation_column_writable
from preprocess_data.utils.lerobot_ops_common import find_episode_parquet_files
from preprocess_data.utils.lerobot_ops_common import init_logging
from preprocess_data.utils.lerobot_ops_common import read_parquet_allow_duplicate_columns
from preprocess_data.utils.lerobot_ops_common import resolve_dataset_root
from preprocess_data.utils.lerobot_ops_common import write_info_json_features
from vam.training import value_inference as _value_inference

logger = logging.getLogger("vam")


def _annotation_loader_config(config: _config.TrainConfig) -> _config.TrainConfig:
    """Build dataloader config for annotation before the target column exists."""
    model_config = config.model
    if isinstance(model_config, vam_config.VAMConfig):
        value_horizon = int(model_config.value_observation_horizon)
        if model_config.action_obs_encoder is not None:
            action_obs_encoder = dataclasses.replace(
                model_config.action_obs_encoder,
                observation_horizon=value_horizon,
                history_tokens=vam_config.HistoryTokenConfig(),
            )
            model_config = dataclasses.replace(
                model_config,
                action_obs_encoder=action_obs_encoder,
            )
    elif hasattr(model_config, "value_observation_horizon"):
        value_horizon = int(model_config.value_observation_horizon)
        field_names = {field.name for field in dataclasses.fields(model_config)}
        replace_kwargs = {
            name: value
            for name, value in {
                "observation_horizon": value_horizon,
                "action_observation_horizon": value_horizon,
                "value_observation_horizon": value_horizon,
                "use_history_state_tokens": False,
                "use_history_value_tokens": False,
            }.items()
            if name in field_names
        }
        if replace_kwargs:
            model_config = dataclasses.replace(model_config, **replace_kwargs)

    data_factory = config.data
    observation_sequence_keys = getattr(data_factory, "observation_sequence_keys", None)
    data_factory_fields = {field.name for field in dataclasses.fields(data_factory)}
    data_factory_updates: dict[str, Any] = {}
    if observation_sequence_keys is not None:
        data_factory_updates["observation_sequence_keys"] = tuple(
            key for key in observation_sequence_keys if key != "value_prediction"
        )
    if "include_advantage" in data_factory_fields:
        data_factory_updates["include_advantage"] = False
    if data_factory_updates:
        data_factory = dataclasses.replace(data_factory, **data_factory_updates)

    return dataclasses.replace(config, model=model_config, data=data_factory)


def write_value_predictions_to_parquet(
    dataset_root: Path,
    value_predictions: np.ndarray,
    *,
    column_name: str = "value_prediction",
    overwrite: bool = False,
) -> int:
    ensure_dataset_exists(dataset_root)
    ensure_annotation_column_writable(dataset_root, column_name, overwrite=overwrite)
    parquet_files = find_episode_parquet_files(dataset_root)
    values = np.asarray(value_predictions, dtype=np.float32).reshape(-1)

    row_counts = [pq.ParquetFile(parquet_path).metadata.num_rows for parquet_path in parquet_files]
    total_rows = int(sum(row_counts))
    if values.shape[0] != total_rows:
        raise ValueError(f"Expected {total_rows} predictions for {dataset_root}, got {values.shape[0]}")

    offset = 0
    for parquet_path, row_count in zip(parquet_files, row_counts, strict=True):
        episode_values = values[offset : offset + row_count]
        offset += row_count

        table = read_parquet_allow_duplicate_columns(parquet_path)
        if column_name in table.column_names:
            table = table.select([idx for idx, name in enumerate(table.column_names) if name != column_name])
        column = pa.array([[float(value)] for value in episode_values], type=pa.list_(pa.float32()))
        table = table.append_column(column_name, column)

        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)

    write_info_json_features(
        dataset_root,
        add_or_update={column_name: {"dtype": "float32", "shape": [1], "names": [column_name]}},
    )
    return len(parquet_files)


def _pad_indexed_batch(
    item: tuple[np.ndarray, dict[str, Any]],
    target_size: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    indices, batch = item
    actual_size = int(indices.shape[0])

    def _is_jax_compatible(value: Any) -> bool:
        if isinstance(value, str | bytes):
            return False
        return not (isinstance(value, np.ndarray) and (np.issubdtype(value.dtype, np.str_) or value.dtype == object))

    batch = {key: value for key, value in batch.items() if _is_jax_compatible(value)}
    if actual_size >= target_size:
        return indices, batch, actual_size

    pad_count = target_size - actual_size
    padded_indices = np.concatenate([indices, np.repeat(indices[-1:], pad_count, axis=0)], axis=0)

    def _pad(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return np.concatenate([value, np.repeat(value[-1:], pad_count, axis=0)], axis=0)
        return value

    return padded_indices, jax.tree.map(_pad, batch), actual_size


def _to_device_batch(
    item: tuple[np.ndarray, dict[str, Any]],
    *,
    target_batch_size: int,
    data_sharding: jax.sharding.Sharding | None,
) -> tuple[np.ndarray, dict[str, Any], int]:
    indices, batch, actual_size = _pad_indexed_batch(item, target_batch_size)
    if data_sharding is None:
        batch_dev = jax.tree.map(jnp.asarray, batch)
    else:
        batch_dev = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
    return indices, batch_dev, actual_size


def _collect_value_predictions_fast(
    *,
    predict_value_fn,
    dataloader,
    data_sharding: jax.sharding.Sharding | None,
    prefetch_size: int,
    inference_batch_size: int,
    total_frames: int,
    start_frame: int,
    value_decode_config: _value_inference.ValueDecodeConfig,
) -> tuple[np.ndarray, int]:
    """Single-forward annotation loop without evaluation/video episode bookkeeping."""
    value_by_frame = np.full(total_frames, np.nan, dtype=np.float32)
    frames_processed = np.zeros(total_frames, dtype=np.bool_)
    decode_queue: queue.Queue[tuple[np.ndarray, int, Any] | None] = queue.Queue(maxsize=max(prefetch_size, 1))
    decode_errors: list[BaseException] = []

    def _decode_worker() -> None:
        try:
            while True:
                item = decode_queue.get()
                if item is None:
                    break
                indices, actual_size, pred_future = item
                current_indices = indices[:actual_size]
                local_indices = current_indices - start_frame
                valid_mask = (local_indices >= 0) & (local_indices < total_frames)
                if not np.all(valid_mask):
                    bad_indices = current_indices[~valid_mask]
                    raise IndexError(
                        "Received out-of-range frame indices during value annotation: "
                        f"{bad_indices[:8].tolist()} (start_frame={start_frame}, total_frames={total_frames})"
                    )

                values = _value_inference.decode_value_predictions(pred_future, value_decode_config)
                value_by_frame[local_indices] = values[:actual_size]
                frames_processed[local_indices] = True
        except Exception as exc:
            decode_errors.append(exc)
            logger.exception("Value prediction decode worker failed")

    decode_thread = threading.Thread(target=_decode_worker, daemon=True)
    decode_thread.start()

    iterator = training_utils.threaded_prefetch(
        iter(dataloader),
        prefetch_size=prefetch_size,
        transform=lambda item: _to_device_batch(
            item,
            target_batch_size=inference_batch_size,
            data_sharding=data_sharding,
        ),
        error_message="Host/device prefetch worker failed",
    )

    processed_frames = 0
    t0 = time.perf_counter()
    for indices, batch, actual_size in tqdm.tqdm(iterator, total=len(dataloader), desc="Inference", dynamic_ncols=True):
        if decode_errors:
            raise RuntimeError("Value prediction decode worker failed") from decode_errors[0]

        obs = _model.Observation.from_dict(batch)
        pred = predict_value_fn(obs, batch["actions"])
        if pred.ndim == 3:
            pred = pred[:, 0, :]

        while True:
            if decode_errors:
                raise RuntimeError("Value prediction decode worker failed") from decode_errors[0]
            try:
                decode_queue.put((indices, actual_size, pred), timeout=1.0)
                break
            except queue.Full:
                logger.warning("Value prediction decode queue is full; waiting for decode worker")
        processed_frames += actual_size

    decode_queue.put(None)
    decode_thread.join()
    if decode_errors:
        raise RuntimeError("Value prediction decode worker failed") from decode_errors[0]

    missing_count = int((~frames_processed).sum())
    if missing_count:
        missing = np.flatnonzero(~frames_processed)[:8]
        raise RuntimeError(
            f"Missing value predictions for {missing_count} frames; first missing indices: {missing.tolist()}"
        )

    elapsed = max(time.perf_counter() - t0, 1e-6)
    logger.info(
        "Inference stage done: %d frames in %.2fs (%.2f frames/s)",
        processed_frames,
        elapsed,
        processed_frames / elapsed,
    )
    return value_by_frame, processed_frames


def annotate_value_predictions_with_config(
    *,
    config: _config.TrainConfig,
    ckpt_path: str,
    inference_batch_size: int = 64,
    num_workers: int = 16,
    prefetch_size: int = 16,
    prefetch_factor: int = 8,
    fsdp_devices: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    overwrite: bool = False,
) -> int:
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if prefetch_size < 0:
        raise ValueError("prefetch_size must be >= 0")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")
    if fsdp_devices is not None and fsdp_devices <= 0:
        raise ValueError("fsdp_devices must be > 0")

    loader_config = _annotation_loader_config(config)

    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))

    data_config = _data_loader.resolve_data_config_override(
        loader_config,
        repo_id=None,
    )
    if data_config.repo_id is None:
        raise ValueError("Value annotation requires a single LeRobot repo_id.")
    root = resolve_dataset_root(repo_id=data_config.repo_id)
    ensure_annotation_column_writable(root, "value_prediction", overwrite=overwrite)
    norm_stats = data_config.norm_stats if config.use_norm_stats else {}
    if norm_stats is None:
        raise ValueError(
            "Normalization stats not found. "
            "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
        )

    resolved_fsdp_devices = fsdp_devices if fsdp_devices is not None else config.fsdp_devices

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    shard_cfg = sharding.choose_inference_sharding(
        disable_fsdp=False,
        requested_fsdp_devices=resolved_fsdp_devices,
        default_fsdp_devices=config.fsdp_devices,
    )
    logger.info("Inference sharding mode: %s (devices=%d)", shard_cfg.mode, jax.device_count())

    model = _value_inference.load_model_fsdp(config, init_rng, ckpt_path, shard_cfg.mesh)
    jax.block_until_ready(nnx.state(model))
    model.eval()
    predict_value_fn = _value_inference.make_sharded_predict_value(
        model,
        shard_cfg.mesh,
    )

    dataset_selection = _data_loader.create_inference_data_loader(
        loader_config,
        data_config=data_config,
        norm_stats=norm_stats,
        inference_batch_size=inference_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        start_index=0,
        num_frames=None,
        episode_ids=None,
    )
    value_decode_config = _value_inference.get_value_decode_config(config.model)
    value_by_frame, processed_frames = _collect_value_predictions_fast(
        predict_value_fn=predict_value_fn,
        dataloader=dataset_selection.dataloader,
        data_sharding=shard_cfg.data_sharding,
        prefetch_size=prefetch_size,
        inference_batch_size=inference_batch_size,
        total_frames=dataset_selection.total_frames,
        start_frame=dataset_selection.start_frame,
        value_decode_config=value_decode_config,
    )
    if processed_frames != dataset_selection.total_frames:
        raise RuntimeError(f"Processed {processed_frames}/{dataset_selection.total_frames} frames")
    if np.isnan(value_by_frame).any():
        raise RuntimeError("Value annotation produced NaN predictions.")

    return write_value_predictions_to_parquet(root, value_by_frame, overwrite=overwrite)


def annotate_value_predictions(
    *,
    config_name: str,
    ckpt_path: str,
    inference_batch_size: int = 64,
    num_workers: int = 16,
    prefetch_size: int = 16,
    prefetch_factor: int = 8,
    fsdp_devices: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    overwrite: bool = False,
) -> int:
    config = _config.get_config(config_name)
    return annotate_value_predictions_with_config(
        config=config,
        ckpt_path=ckpt_path,
        inference_batch_size=inference_batch_size,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        prefetch_factor=prefetch_factor,
        fsdp_devices=fsdp_devices,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        overwrite=overwrite,
    )


def main(argv=None) -> None:
    init_logging()
    logger.info("Running on: %s", platform.node())
    parser = argparse.ArgumentParser(description="Annotate every LeRobot frame with pretrained value predictions.")
    parser.add_argument("config_name", type=str)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-size", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing value_prediction column.")
    args = parser.parse_args(argv)

    modified = annotate_value_predictions(
        config_name=args.config_name,
        ckpt_path=args.ckpt_path,
        inference_batch_size=args.inference_batch_size,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        prefetch_factor=args.prefetch_factor,
        fsdp_devices=args.fsdp_devices,
        pin_memory=True,
        persistent_workers=True,
        overwrite=args.overwrite,
    )
    print(f"Annotated value predictions in {modified} parquet files.")


if __name__ == "__main__":
    main()
