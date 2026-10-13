# Synchronous training-time validation helpers: episode selection, manifest persistence,
# in-process value inference, and video visualization.

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

import cv2
import flax.nnx as nnx
import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vam.models.model as _model
import vam.training.config as _config
import vam.training.data_loader as _data_loader
import vam.training.sharding as sharding
import vam.training.utils as training_utils
import vam.transforms as _transforms

VIDEO_FRAME_STRIDE = 10
VIDEO_OUTPUT_FPS = 30


@dataclasses.dataclass(frozen=True)
class ValidationEpisodeRef:
    repo_id: str
    episode_id: int


@dataclasses.dataclass(frozen=True)
class VisualizationImageSpec:
    raw_key: str
    display_name: str
    target_key: str | None = None


@dataclasses.dataclass(frozen=True)
class ValueDecodeConfig:
    discrete_value: bool
    n_bins: int
    vmin: float
    vmax: float


@dataclasses.dataclass(frozen=True)
class ValidationLoader:
    repo_id: str
    selection: _data_loader.DatasetSelection
    image_specs: list[VisualizationImageSpec]


def _validation_root(config: _config.TrainConfig) -> pathlib.Path:
    return pathlib.Path(config.checkpoint_dir) / "validation"


def _validation_manifest_path(config: _config.TrainConfig) -> pathlib.Path:
    return _validation_root(config) / "selected_episodes.json"


def _repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _load_validation_episode_refs_from_manifest(manifest_path: pathlib.Path) -> list[ValidationEpisodeRef]:
    payload = json.loads(manifest_path.read_text())
    if "selections" in payload:
        return [
            ValidationEpisodeRef(repo_id=entry["repo_id"], episode_id=int(entry["episode_id"]))
            for entry in payload["selections"]
        ]
    return [ValidationEpisodeRef(repo_id=payload["repo_id"], episode_id=int(ep)) for ep in payload["episode_ids"]]


def _validation_manifest_matches_config(
    payload: dict[str, Any],
    validation_config: _config.ValidationConfig,
    *,
    seed: int,
) -> bool:
    if payload.get("num_episodes") != validation_config.num_episodes:
        return False
    if payload.get("start_index") != validation_config.start_index:
        return False
    if payload.get("seed") != seed:
        return False

    manifest_repo_ids: tuple[str, ...]
    if "selections" in payload:
        manifest_repo_ids = tuple(payload.get("repo_ids", ()))
    else:
        repo_id = payload.get("repo_id")
        manifest_repo_ids = () if repo_id is None else (repo_id,)
    return manifest_repo_ids == validation_config.effective_repo_ids()


def _collect_validation_candidates(validation_config: _config.ValidationConfig) -> dict[str, np.ndarray]:
    repo_candidates: dict[str, np.ndarray] = {}
    for repo_id in validation_config.effective_repo_ids():
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        total_episodes = int(dataset_meta.total_episodes)
        if validation_config.start_index >= total_episodes:
            raise ValueError(
                f"validation.start_index={validation_config.start_index} must be smaller than "
                f"dataset episodes={total_episodes} for repo_id={repo_id}"
            )
        candidate_episode_ids = np.arange(validation_config.start_index, total_episodes, dtype=np.int64)
        if len(candidate_episode_ids) == 0:
            raise ValueError(f"No validation episodes available for repo_id={repo_id}")
        repo_candidates[repo_id] = candidate_episode_ids
    return repo_candidates


def _sample_single_repo_validation_refs(
    *,
    repo_id: str,
    candidate_episode_ids: np.ndarray,
    num_episodes: int,
    seed: int,
) -> list[ValidationEpisodeRef]:
    if num_episodes > len(candidate_episode_ids):
        raise ValueError(
            f"validation.num_episodes={num_episodes} exceeds available episodes={len(candidate_episode_ids)}"
        )
    rng = np.random.default_rng(seed)
    episode_ids = sorted(int(ep) for ep in rng.choice(candidate_episode_ids, size=num_episodes, replace=False))
    return [ValidationEpisodeRef(repo_id=repo_id, episode_id=episode_id) for episode_id in episode_ids]


def _sample_multi_repo_validation_refs(
    *,
    repo_candidates: dict[str, np.ndarray],
    num_episodes: int,
    start_index: int,
    seed: int,
) -> list[ValidationEpisodeRef]:
    repo_ids = tuple(repo_candidates)
    if num_episodes < len(repo_ids):
        raise ValueError("validation.num_episodes must be >= number of validation repos when using validation.repo_ids")

    total_candidates = sum(len(candidate_ids) for candidate_ids in repo_candidates.values())
    if num_episodes > total_candidates:
        raise ValueError(
            f"validation.num_episodes={num_episodes} exceeds available episodes={total_candidates} "
            f"from start_index={start_index}"
        )

    rng = np.random.default_rng(seed)
    selected_refs: list[ValidationEpisodeRef] = []
    remaining_refs: list[ValidationEpisodeRef] = []

    for repo_id, candidate_ids in repo_candidates.items():
        chosen_episode = int(rng.choice(candidate_ids, size=1, replace=False)[0])
        selected_refs.append(ValidationEpisodeRef(repo_id=repo_id, episode_id=chosen_episode))
        remaining_refs.extend(
            ValidationEpisodeRef(repo_id=repo_id, episode_id=int(ep))
            for ep in candidate_ids
            if int(ep) != chosen_episode
        )

    remaining_slots = num_episodes - len(selected_refs)
    if remaining_slots > 0:
        chosen_indices = rng.choice(len(remaining_refs), size=remaining_slots, replace=False)
        selected_refs.extend(remaining_refs[int(idx)] for idx in np.atleast_1d(chosen_indices))

    return sorted(selected_refs, key=lambda ref: (ref.repo_id, ref.episode_id))


def resolve_validation_episode_refs(config: _config.TrainConfig) -> list[ValidationEpisodeRef]:
    """Load fixed validation episode refs from the manifest, regenerating if stale."""
    validation_config = config.validation
    manifest_path = _validation_manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        if _validation_manifest_matches_config(payload, validation_config, seed=config.seed):
            episode_refs = _load_validation_episode_refs_from_manifest(manifest_path)
            logging.info("Loaded %d fixed validation episodes from %s", len(episode_refs), manifest_path)
            return episode_refs
        logging.info("Validation manifest at %s is stale for the current config; regenerating.", manifest_path)

    repo_candidates = _collect_validation_candidates(validation_config)
    repo_ids = tuple(repo_candidates)
    if len(repo_ids) == 1:
        selected_refs = _sample_single_repo_validation_refs(
            repo_id=repo_ids[0],
            candidate_episode_ids=repo_candidates[repo_ids[0]],
            num_episodes=validation_config.num_episodes,
            seed=config.seed,
        )
    else:
        selected_refs = _sample_multi_repo_validation_refs(
            repo_candidates=repo_candidates,
            num_episodes=validation_config.num_episodes,
            start_index=validation_config.start_index,
            seed=config.seed,
        )

    payload: dict[str, Any] = {
        "num_episodes": validation_config.num_episodes,
        "start_index": validation_config.start_index,
        "seed": config.seed,
    }
    if len(repo_ids) == 1:
        payload["repo_id"] = repo_ids[0]
        payload["episode_ids"] = [ref.episode_id for ref in selected_refs]
    else:
        payload["repo_ids"] = list(repo_ids)
        payload["selections"] = [{"repo_id": ref.repo_id, "episode_id": ref.episode_id} for ref in selected_refs]
    manifest_path.write_text(json.dumps(payload, indent=2))
    logging.info("Persisted %d fixed validation episodes to %s", len(selected_refs), manifest_path)
    return selected_refs


def prepare_validation_loaders(
    config: _config.TrainConfig,
    episode_refs: list[ValidationEpisodeRef],
) -> list[ValidationLoader]:
    if not config.validation.enabled:
        return []

    training_data_config = config.data.create(config.assets_dirs, config.model)
    use_norm_stats = (
        config.use_norm_stats if config.validation.use_norm_stats is None else config.validation.use_norm_stats
    )
    if use_norm_stats:
        norm_stats = training_data_config.norm_stats
        if norm_stats is None:
            raise ValueError("Training data norm_stats are required for validation normalization but are missing.")
        logging.info(
            "Validation normalization uses training data norm stats from asset_id=%s", training_data_config.asset_id
        )
    else:
        norm_stats = {}
        logging.info("Validation normalization disabled.")

    repo_to_episode_ids: dict[str, list[int]] = {}
    for ref in episode_refs:
        repo_to_episode_ids.setdefault(ref.repo_id, []).append(ref.episode_id)

    loaders: list[ValidationLoader] = []
    for repo_id, episode_ids in repo_to_episode_ids.items():
        data_config = _data_loader.resolve_data_config_override(
            config,
            repo_id=repo_id,
        )
        image_specs = _derive_visualization_image_specs(data_config)
        selection = _data_loader.create_inference_data_loader(
            config,
            data_config=data_config,
            norm_stats=norm_stats,
            inference_batch_size=config.validation.inference_batch_size or config.batch_size,
            num_workers=config.validation.num_workers
            if config.validation.num_workers is not None
            else config.num_workers,
            pin_memory=config.validation.pin_memory if config.validation.pin_memory is not None else config.pin_memory,
            prefetch_factor=(
                config.validation.prefetch_factor
                if config.validation.prefetch_factor is not None
                else config.prefetch_factor
            ),
            persistent_workers=(
                config.validation.persistent_workers
                if config.validation.persistent_workers is not None
                else config.persistent_workers
            ),
            episode_ids=sorted(episode_ids),
        )
        logging.info(
            "Prepared validation loader repo_id=%s episodes=%s frames=%d batch_size=%d",
            repo_id,
            sorted(episode_ids),
            selection.total_frames,
            config.validation.inference_batch_size or config.batch_size,
        )
        loaders.append(ValidationLoader(repo_id=repo_id, selection=selection, image_specs=image_specs))

    return loaders


def _get_value_decode_config(model_config: _model.BaseModelConfig) -> ValueDecodeConfig:
    discrete_value = bool(getattr(model_config, "discrete_value", False))
    n_bins = int(getattr(model_config, "n_bins", 1))
    vmin = float(getattr(model_config, "vmin", 0.0))
    vmax = float(getattr(model_config, "vmax", 1.0))
    return ValueDecodeConfig(discrete_value=discrete_value, n_bins=n_bins, vmin=vmin, vmax=vmax)


def _decode_value_predictions(predictions: Any, value_decode_config: ValueDecodeConfig) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim == 0:
        raise ValueError("Expected batched value predictions, got scalar output.")

    if value_decode_config.discrete_value:
        support = np.linspace(
            value_decode_config.vmin,
            value_decode_config.vmax,
            value_decode_config.n_bins,
            dtype=np.float32,
        )
        return np.sum(support[np.newaxis, :] * values, axis=-1, dtype=np.float32)

    while values.ndim > 1 and values.shape[-1] == 1:
        values = np.squeeze(values, axis=-1)

    return values.astype(np.float32, copy=False)


def _pad_batch(indices: np.ndarray, batch: dict[str, Any], target_size: int) -> tuple[np.ndarray, dict[str, Any], int]:
    actual_size = int(indices.shape[0])

    def _filter_strings(kv):
        _, val = kv
        if isinstance(val, str | bytes):
            return False
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.str_):
            return False
        return not (isinstance(val, np.ndarray) and val.dtype == object)

    batch = {key: value for key, value in batch.items() if _filter_strings((key, value))}
    if actual_size >= target_size:
        return indices, batch, actual_size

    pad_count = target_size - actual_size
    indices = np.concatenate([indices, np.repeat(indices[-1:], pad_count, axis=0)], axis=0)

    def _pad(x):
        if isinstance(x, np.ndarray) and x.ndim >= 1:
            return np.concatenate([x, np.repeat(x[-1:], pad_count, axis=0)], axis=0)
        return x

    return indices, jax.tree.map(_pad, batch), actual_size


def _to_device_batch(
    item: tuple[np.ndarray, dict[str, Any]],
    *,
    target_batch_size: int,
    data_sharding: jax.sharding.Sharding,
) -> tuple[np.ndarray, dict[str, Any], int]:
    indices, batch = item
    idx_pad, batch_pad, actual = _pad_batch(indices, batch, target_batch_size)
    batch_dev = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch_pad)
    return idx_pad, batch_dev, actual


def _make_predict_value_fn(
    train_state: training_utils.TrainState,
):
    graphdef = train_state.model_def

    def _predict_value(params: nnx.State, batch: dict[str, Any]):
        model = nnx.merge(graphdef, params)
        model.eval()
        observation = _model.Observation.from_dict(batch)
        pred = model.predict_value(observation, batch["actions"])
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim == 3:
            pred = pred[:, 0, :]
        return pred

    return jax.jit(_predict_value)


def _lookup_sample_value(sample: Any, *candidates: str) -> Any:
    for key in candidates:
        if hasattr(sample, "get"):
            value = sample.get(key)
            if value is not None:
                return value

        for separator in ("/", "."):
            current = sample
            found = True
            for part in key.split(separator):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    found = False
                    break
            if found:
                return current

    return None


def _to_numpy(x: Any) -> Any:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _image_to_hwc_uint8(image: Any, *, image_name: str) -> np.ndarray:
    image = np.asarray(_to_numpy(image))
    if image.ndim == 4:
        if image.shape[-1] in (1, 3) or image.shape[1] in (1, 3):
            image = image[-1]
        elif image.shape[0] in (1, 3):
            image = image[:, -1]
        else:
            raise ValueError(f"Could not infer temporal axis for {image_name} image history with shape {image.shape}")

    if image.ndim != 3:
        raise ValueError(f"Expected {image_name} image to be rank-3, got shape {image.shape}")

    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected {image_name} image to have 3 channels, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        finite_min = float(np.nanmin(image)) if image.size else 0.0
        if finite_min >= 0.0 and finite_max <= 1.0:
            image = image * 255.0
        elif finite_min >= -1.0 and finite_max <= 1.0:
            image = (image + 1.0) * 127.5
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _fallback_image() -> np.ndarray:
    return np.zeros((*_model.IMAGE_RESOLUTION, 3), dtype=np.uint8)


def _format_image_label(name: str) -> str:
    return name.replace("_", " ").replace("/", " ").strip().title()


def _camera_sort_key(name: str) -> tuple[int, str]:
    lowered = name.lower()
    if any(token in lowered for token in ("primary", "base", "high", "top")):
        return (0, lowered)
    if "left" in lowered:
        return (1, lowered)
    if "right" in lowered:
        return (2, lowered)
    if "wrist" in lowered:
        return (3, lowered)
    return (4, lowered)


def _is_visualization_image_target(target_key: str, raw_key: str) -> bool:
    target_leaf = target_key.split("/")[-1]
    raw_leaf = raw_key.split("/")[-1].split(".")[-1]
    skipped_tokens = ("is_pad", "mask", "valid")
    if any(token in target_leaf or token in raw_leaf for token in skipped_tokens):
        return False
    if target_leaf.startswith("next_") or raw_leaf.startswith("next_"):
        return False
    return target_key.startswith(("observation/images/", "images/")) or (
        target_key.startswith("observation/") and "image" in target_leaf
    )


def _derive_visualization_image_specs(data_config: _config.DataConfig) -> list[VisualizationImageSpec]:
    specs: list[VisualizationImageSpec] = []
    seen_raw_keys: set[str] = set()

    for transform in data_config.repack_transforms.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue

        for target_key, raw_key in _transforms.flatten_dict(transform.structure).items():
            if not isinstance(raw_key, str):
                continue
            if not _is_visualization_image_target(target_key, raw_key):
                continue
            if raw_key in seen_raw_keys:
                continue

            display_source = target_key.split("/")[-1]
            specs.append(
                VisualizationImageSpec(
                    raw_key=raw_key,
                    display_name=_format_image_label(display_source),
                    target_key=target_key,
                )
            )
            seen_raw_keys.add(raw_key)

    specs.sort(key=lambda spec: _camera_sort_key(spec.target_key or spec.display_name))
    return specs


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _add_image_label(image: np.ndarray, label: str | None) -> np.ndarray:
    if not label:
        return image

    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (min(labeled.shape[1], 150), 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _compose_image_panel(images: list[np.ndarray], image_labels: list[str] | None = None) -> np.ndarray:
    if not images:
        images = [_fallback_image()]

    image_labels = image_labels or [""] * len(images)
    base_height, base_width = images[0].shape[:2]
    tile_size = (base_width, base_height)

    tiles = [
        _add_image_label(
            _resize_image(_image_to_hwc_uint8(image, image_name=f"image_{idx}"), tile_size),
            image_labels[idx] if idx < len(image_labels) else None,
        )
        for idx, image in enumerate(images)
    ]

    if len(tiles) <= 3:
        return np.vstack(tiles)

    rows: list[np.ndarray] = []
    for idx in range(0, len(tiles), 2):
        row_tiles = tiles[idx : idx + 2]
        if len(row_tiles) == 1:
            row_tiles.append(np.zeros_like(row_tiles[0]))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def _extract_visualization_images(sample: Any, image_specs: list[VisualizationImageSpec]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for spec in image_specs:
        image = _lookup_sample_value(sample, spec.raw_key)
        if image is None:
            images.append(_fallback_image())
            continue
        images.append(_image_to_hwc_uint8(image, image_name=spec.display_name))
    return images


def _resolve_value_plot_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if np.isclose(vmin, vmax):
        pad = 0.05 if np.isfinite(vmin) else 1.0
        return float(vmin - pad), float(vmax + pad)
    return float(min(vmin, vmax)), float(max(vmin, vmax))


def _resolve_plot_xmax(episode_length: int) -> int:
    sampled_episode_length = max((max(episode_length, 1) - 1) // VIDEO_FRAME_STRIDE + 1, 1)
    return max(sampled_episode_length - 1, 10)


def create_frame(
    *,
    images: list[np.ndarray],
    expected_value_history: list[float],
    episode_length: int,
    value_plot_limits: tuple[float, float],
    image_labels: list[str] | None = None,
) -> np.ndarray:
    stacked_images = _compose_image_panel(images, image_labels)
    plot_ymin, plot_ymax = _resolve_value_plot_limits(*value_plot_limits)

    plot_dpi = 170
    fig, ax = plt.subplots(figsize=(4.6, 7), dpi=plot_dpi)
    fig.patch.set_facecolor("white")

    timesteps = list(range(len(expected_value_history)))
    current_value = expected_value_history[-1] if expected_value_history else 0.0
    ax.set_title(f"Expected Value: {current_value:.4f}", fontsize=15, pad=10)
    ax.plot(timesteps, expected_value_history, "b-", linewidth=2.8)
    ax.fill_between(timesteps, expected_value_history, alpha=0.3)

    ax.set_xlim(0, _resolve_plot_xmax(episode_length))
    ax.set_ylim(plot_ymin, plot_ymax)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Value", fontsize=13)
    ax.grid(visible=True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout(pad=1.2)
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if plot_img.shape[0] != stacked_images.shape[0]:
        scale = stacked_images.shape[0] / plot_img.shape[0]
        new_width = int(plot_img.shape[1] * scale)
        plot_img = cv2.resize(plot_img, (new_width // 2 * 2, stacked_images.shape[0]), interpolation=cv2.INTER_LINEAR)

    return np.hstack([stacked_images, plot_img])


def _iter_sampled_frame_pairs(start: int, end: int, *, stride: int = VIDEO_FRAME_STRIDE) -> list[tuple[int, int]]:
    sampled = list(range(start, end, stride))
    if not sampled or sampled[-1] != end - 1:
        sampled.append(end - 1)
    return [(global_idx, global_idx - start) for global_idx in sampled]


def create_video_from_frames(frames: list[np.ndarray], output_path: pathlib.Path, fps: int = VIDEO_OUTPUT_FPS) -> bool:
    if not frames:
        logging.warning("No frames provided for %s", output_path)
        return False

    logging.info("Creating video %s from %d frames at %d fps", output_path, len(frames), fps)
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    filelist_path = temp_dir / "filelist.txt"

    try:
        for i, frame in enumerate(frames):
            frame_path = temp_dir / f"frame_{i:04d}.png"
            frame_bgr = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)

        with open(filelist_path, "w") as f:
            for i in range(len(frames)):
                frame_path = str((temp_dir / f"frame_{i:04d}.png").absolute()).replace("'", "'\\''")
                f.write(f"file '{frame_path}'\n")

        temp_video = temp_dir / "output.mp4"
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),
            "-i",
            str(filelist_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(temp_video),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_video, "rb") as src, open(output_path, "wb") as dst:
                dst.write(src.read())
            logging.info("Created video: %s", output_path)
            return True

        logging.error("Failed to create video %s: %s", output_path, result.stderr)
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _reap_video_threads(active_threads: list[threading.Thread]) -> list[threading.Thread]:
    remaining: list[threading.Thread] = []
    for thread in active_threads:
        if thread.is_alive():
            remaining.append(thread)
            continue
        thread.join()
    return remaining


def join_validation_video_threads(active_threads: list[threading.Thread]) -> None:
    for thread in active_threads:
        logging.info("Waiting for validation video writer %s to finish...", thread.name)
        thread.join()


def _episode_infos(selection: _data_loader.DatasetSelection) -> list[dict[str, int]]:
    episode_data_index = selection.raw_dataset.episode_data_index
    selected_episode_ids = selection.selected_episode_ids
    if selected_episode_ids is None:
        raise ValueError("Training-time validation expects explicit selected episode ids.")

    episode_info: list[dict[str, int]] = []
    running_local_start = 0
    for ep_idx in selected_episode_ids:
        global_start = int(_to_numpy(episode_data_index["from"][ep_idx]).item())
        global_end = int(_to_numpy(episode_data_index["to"][ep_idx]).item())
        if global_end <= global_start:
            continue
        episode_len = global_end - global_start
        episode_info.append(
            {
                "ep_idx": int(ep_idx),
                "global_start": global_start,
                "global_end": global_end,
                "local_start": running_local_start,
                "local_end": running_local_start + episode_len,
            }
        )
        running_local_start += episode_len
    return episode_info


def _write_validation_videos(
    *,
    value_by_frame: np.ndarray,
    selection: _data_loader.DatasetSelection,
    image_specs: list[VisualizationImageSpec],
    output_dir: pathlib.Path,
    value_plot_limits: tuple[float, float],
) -> int:
    videos_created = 0
    image_labels = [spec.display_name for spec in image_specs]
    for info in _episode_infos(selection):
        frames: list[np.ndarray] = []
        value_history: list[float] = []
        episode_length = info["global_end"] - info["global_start"]

        for global_frame_idx, rel_offset in _iter_sampled_frame_pairs(info["global_start"], info["global_end"]):
            local_frame_idx = info["local_start"] + rel_offset
            sample = selection.raw_dataset[global_frame_idx]
            images = _extract_visualization_images(sample, image_specs)
            value_history.append(float(value_by_frame[local_frame_idx]))
            frames.append(
                create_frame(
                    images=images,
                    image_labels=image_labels,
                    expected_value_history=value_history,
                    episode_length=episode_length,
                    value_plot_limits=value_plot_limits,
                )
            )

        if create_video_from_frames(frames, output_dir / f"{info['ep_idx']}.mp4"):
            videos_created += 1
    return videos_created


def _json_float(value: float | np.floating) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _write_validation_outputs(
    *,
    output_dir: pathlib.Path,
    value_by_frame: np.ndarray,
    episode_ids: list[int],
    elapsed_sec: float,
    video_status: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    finite_values = value_by_frame[np.isfinite(value_by_frame)]
    metrics = {
        "num_frames": int(value_by_frame.shape[0]),
        "num_episodes": len(episode_ids),
        "episode_ids": episode_ids,
        "mean_value": _json_float(np.mean(finite_values)) if finite_values.size else None,
        "min_value": _json_float(np.min(finite_values)) if finite_values.size else None,
        "max_value": _json_float(np.max(finite_values)) if finite_values.size else None,
        "nan_count": int(np.isnan(value_by_frame).sum()),
        "elapsed_sec": float(elapsed_sec),
        "video_status": video_status,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
        tmp_path = pathlib.Path(tmp_file.name)
    try:
        np.savez(tmp_path, value_by_frame=value_by_frame, episode_ids=np.asarray(episode_ids))
        shutil.copyfile(tmp_path, output_dir / "values.npz")
    finally:
        tmp_path.unlink(missing_ok=True)


def _start_validation_video_writer(
    *,
    value_by_frame: np.ndarray,
    selection: _data_loader.DatasetSelection,
    image_specs: list[VisualizationImageSpec],
    output_dir: pathlib.Path,
    value_plot_limits: tuple[float, float],
    repo_id: str,
    step: int,
) -> threading.Thread:
    def _worker() -> None:
        start_time = time.perf_counter()
        try:
            videos_created = _write_validation_videos(
                value_by_frame=value_by_frame,
                selection=selection,
                image_specs=image_specs,
                output_dir=output_dir,
                value_plot_limits=value_plot_limits,
            )
            elapsed_sec = max(time.perf_counter() - start_time, 1e-6)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "video_metrics.json").write_text(
                json.dumps(
                    {
                        "repo_id": repo_id,
                        "step": int(step),
                        "videos_created": int(videos_created),
                        "elapsed_sec": float(elapsed_sec),
                    },
                    indent=2,
                )
            )
            logging.info(
                "Validation videos finished repo_id=%s step=%s videos=%d elapsed=%.2fs output=%s",
                repo_id,
                step,
                videos_created,
                elapsed_sec,
                output_dir,
            )
        except Exception:
            logging.exception("Validation video writer failed repo_id=%s step=%s output=%s", repo_id, step, output_dir)

    thread = threading.Thread(target=_worker, name=f"validation-video-{step}-{_repo_slug(repo_id)}", daemon=False)
    thread.start()
    return thread


def _run_single_validation_loader(
    *,
    config: _config.TrainConfig,
    step: int,
    validation_loader: ValidationLoader,
    predict_value_fn,
    params: nnx.State,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding,
    output_dir: pathlib.Path,
    schedule_videos: bool,
) -> threading.Thread | None:
    selection = validation_loader.selection
    value_decode_config = _get_value_decode_config(config.model)
    value_by_frame = np.full(selection.total_frames, np.nan, dtype=np.float32)
    inference_batch_size = config.validation.inference_batch_size or config.batch_size
    prefetch_size = (
        config.validation.prefetch_size if config.validation.prefetch_size is not None else config.prefetch_size
    )
    start_time = time.perf_counter()

    iterator = training_utils.threaded_prefetch(
        iter(selection.dataloader),
        prefetch_size=prefetch_size,
        transform=lambda item: _to_device_batch(
            item,
            target_batch_size=inference_batch_size,
            data_sharding=data_sharding,
        ),
        error_message="Validation host/device prefetch worker failed",
    )

    logging.info(
        "Running validation inference repo_id=%s frames=%d batch_size=%d",
        validation_loader.repo_id,
        selection.total_frames,
        inference_batch_size,
    )
    for indices, batch, actual_size in iterator:
        with sharding.set_mesh(mesh):
            pred = predict_value_fn(params, batch)
            jax.block_until_ready(pred)
        values = _decode_value_predictions(pred, value_decode_config)
        local_indices = indices[:actual_size]
        valid_mask = (local_indices >= 0) & (local_indices < selection.total_frames)
        if not np.all(valid_mask):
            bad_indices = local_indices[~valid_mask]
            raise IndexError(
                "Received out-of-range validation frame indices: "
                f"{bad_indices[:8].tolist()} (total_frames={selection.total_frames})"
            )
        value_by_frame[local_indices] = values[:actual_size]

    elapsed_sec = max(time.perf_counter() - start_time, 1e-6)
    episode_ids = selection.selected_episode_ids or []
    video_thread = None
    video_status = "skipped_active_video_writer"
    if schedule_videos:
        video_status = "scheduled"
    _write_validation_outputs(
        output_dir=output_dir,
        value_by_frame=value_by_frame,
        episode_ids=episode_ids,
        elapsed_sec=elapsed_sec,
        video_status=video_status,
    )
    if schedule_videos:
        video_thread = _start_validation_video_writer(
            value_by_frame=value_by_frame.copy(),
            selection=selection,
            image_specs=validation_loader.image_specs,
            output_dir=output_dir,
            value_plot_limits=(value_decode_config.vmin, value_decode_config.vmax),
            repo_id=validation_loader.repo_id,
            step=step,
        )
    logging.info(
        "Validation inference finished repo_id=%s frames=%d elapsed=%.2fs video_status=%s output=%s",
        validation_loader.repo_id,
        selection.total_frames,
        elapsed_sec,
        video_status,
        output_dir,
    )
    return video_thread


def run_in_process_validation(
    config: _config.TrainConfig,
    *,
    step: int,
    train_state: training_utils.TrainState,
    validation_loaders: list[ValidationLoader] | None,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding,
    active_video_threads: list[threading.Thread] | None = None,
) -> None:
    if not config.validation.enabled or not validation_loaders:
        return

    active_threads = _reap_video_threads(active_video_threads or [])
    schedule_videos = not active_threads
    if active_threads:
        logging.warning(
            "Skipping validation video scheduling for step %s because %d previous video writer(s) are still running.",
            step,
            len(active_threads),
        )

    params = train_state.ema_params if train_state.ema_params is not None else train_state.params
    predict_value_fn = _make_predict_value_fn(
        train_state,
    )
    output_root = _validation_root(config) / str(step)
    multiple_repos = len(validation_loaders) > 1
    logging.info("Starting in-process validation for step %s", step)
    new_video_threads: list[threading.Thread] = []

    for validation_loader in validation_loaders:
        output_dir = output_root / _repo_slug(validation_loader.repo_id) if multiple_repos else output_root
        video_thread = _run_single_validation_loader(
            config=config,
            step=step,
            validation_loader=validation_loader,
            predict_value_fn=predict_value_fn,
            params=params,
            mesh=mesh,
            data_sharding=data_sharding,
            output_dir=output_dir,
            schedule_videos=schedule_videos,
        )
        if video_thread is not None:
            new_video_threads.append(video_thread)

    if active_video_threads is not None:
        active_video_threads[:] = [*active_threads, *new_video_threads]
