"""Online DAgger/Learn from Experience: train action/value experts from streaming datasets."""

import dataclasses
import functools
import gc
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

from etils import epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vam.models.model as _model
from vam.shared.logging_utils import init_cli_logging
from vam.training import async_control
import vam.training.checkpoints as _checkpoints
import vam.training.config as _config
import vam.training.data_loader as _data_loader
import vam.training.sharding as sharding
import vam.training.utils as training_utils
import vam.training.validation as _validation
from scripts import train as sync_train


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _debug_every(name: str, default: int = 10) -> int:
    return max(1, int(os.environ.get(name, str(default))))


class AsyncTrainStream:
    def __init__(
        self,
        *,
        name: str,
        default_train_type_values: tuple[str, ...],
        train_type_by_target_policy_id: dict[int, tuple[str, ...]],
        train_types: set[_model.TrainingType],
        data_config: _config.DataConfig,
        watcher: async_control.DatasetViewWatcher,
        data_loader: _data_loader.DataLoader,
        data_iter: Any,
        batch: Any,
        steps_per_cycle: int,
        ptrain_step: Any,
        episode_replay_state: _data_loader.EpisodeReplayState,
        view_id: str | None = None,
        target_policy_id: int | None = None,
    ):
        self.name = name
        self.default_train_type_values = tuple(default_train_type_values)
        self.train_type_by_target_policy_id = dict(train_type_by_target_policy_id)
        self.train_types = train_types
        self.data_config = data_config
        self.watcher = watcher
        self.data_loader = data_loader
        self.data_iter = data_iter
        self.batch = batch
        self.steps_per_cycle = max(1, int(steps_per_cycle))
        self.ptrain_step = ptrain_step
        self.episode_replay_state = episode_replay_state
        self.view_id = view_id
        self.target_policy_id = target_policy_id
        self.last_view_update_step: int | None = 0 if view_id is not None else None
        self.last_policy_update_step: int | None = 0 if target_policy_id is not None else None
        self.skip_next_batch_prefetch_once = False
        self.enable_device_prefetch_after_step = False
        self._next_batch_thread: threading.Thread | None = None
        self._next_batch: Any | None = None
        self._next_batch_error: BaseException | None = None
        self.last_next_batch_fetch_seconds: float | None = None
        self.last_next_batch_join_seconds: float | None = None
        # Set by the reload machinery to (reload_step, stall_start_monotonic) so the training loop can
        # measure the main-thread stall between a dataset-view swap and the next completed training step.
        # Cleared after logging.
        self.reload_bench_pending: tuple[int, float] | None = None
        # Background dataset-view reload state. The expensive part of a reload (building the new dataloader
        # and fetching its first batch — dominated by worker spawn + video decode) runs in this thread while
        # the training loop keeps stepping on the current pipeline. The finished pipeline is swapped in by the
        # main thread. See _maybe_reload_stream.
        self._reload_thread: threading.Thread | None = None
        self._reload_lock = threading.Lock()
        self._reload_result: dict[str, Any] | None = None
        self.reload_build_t0: float | None = None

    def reload_build_in_flight(self) -> bool:
        return self._reload_thread is not None and self._reload_thread.is_alive()

    def start_reload_build(self, build_fn) -> None:
        """Run build_fn (returns a result dict) in a background thread; result is picked up by the main loop."""
        self._reload_result = None

        def _run() -> None:
            try:
                result = build_fn()
            except Exception as exc:
                result = {"error": exc}
            with self._reload_lock:
                self._reload_result = result

        self._reload_thread = threading.Thread(target=_run, name=f"{self.name}-reload-build", daemon=True)
        self.reload_build_t0 = time.monotonic()
        self._reload_thread.start()

    def take_reload_result(self) -> dict[str, Any] | None:
        with self._reload_lock:
            result = self._reload_result
            self._reload_result = None
        if result is not None:
            self._reload_thread = None
        return result

    def train_type_values_for_policy(self, target_policy_id: int | None) -> tuple[str, ...]:
        if target_policy_id is not None and int(target_policy_id) in self.train_type_by_target_policy_id:
            return tuple(self.train_type_by_target_policy_id[int(target_policy_id)])
        return self.default_train_type_values

    def start_next_batch_fetch(self) -> None:
        if self._next_batch_thread is not None:
            return
        self._next_batch = None
        self._next_batch_error = None
        self.last_next_batch_fetch_seconds = None
        self.last_next_batch_join_seconds = None

        def _fetch() -> None:
            fetch_t0 = time.monotonic()
            try:
                self._next_batch = next(self.data_iter)
            except BaseException as exc:  # noqa: BLE001
                self._next_batch_error = exc
            finally:
                self.last_next_batch_fetch_seconds = time.monotonic() - fetch_t0

        self._next_batch_thread = threading.Thread(target=_fetch, name=f"{self.name}-next-batch", daemon=True)
        self._next_batch_thread.start()

    def finish_next_batch_fetch(self) -> None:
        if self._next_batch_thread is None:
            fetch_t0 = time.monotonic()
            self.batch = next(self.data_iter)
            elapsed = time.monotonic() - fetch_t0
            self.last_next_batch_fetch_seconds = elapsed
            self.last_next_batch_join_seconds = elapsed
            return
        join_t0 = time.monotonic()
        self._next_batch_thread.join()
        self.last_next_batch_join_seconds = time.monotonic() - join_t0
        self._next_batch_thread = None
        if self._next_batch_error is not None:
            raise RuntimeError(
                f"Async stream '{self.name}' failed to fetch the next batch."
            ) from self._next_batch_error
        self.batch = self._next_batch
        self._next_batch = None

    def wait_for_pending_batch(self) -> None:
        if self._next_batch_thread is not None:
            self.finish_next_batch_fetch()

    def release_data_buffers(self, *, clear_batch: bool, context: str, clear_jax_caches: bool = False) -> None:
        if self._next_batch_thread is not None:
            self._next_batch_thread.join()
            self._next_batch_thread = None
        close = getattr(self.data_iter, "close", None)
        if close is not None:
            close()
        self.data_iter = None
        self._next_batch = None
        self._next_batch_error = None
        if clear_batch:
            self.batch = None
        gc.collect()
        # jax.clear_caches() evicts the compiled ptrain_step executable and forces a full recompile on
        # the next step. Dataset-view reloads keep batch shapes static and reuse the same ptrain_step, so
        # clearing is pure waste there. Only clear when the caller rebuilds ptrain_step (train_type change),
        # where the old executable is genuinely garbage.
        if clear_jax_caches:
            jax.clear_caches()
        logging.info("Released async stream '%s' data buffers after %s.", self.name, context)

    def replace_data_pipeline(
        self,
        *,
        data_config: _config.DataConfig,
        data_loader: _data_loader.DataLoader,
        data_iter: Any,
        batch: Any,
        skip_next_batch_prefetch_once: bool,
        enable_device_prefetch_after_step: bool,
        context: str,
        clear_jax_caches: bool = False,
        episode_replay_state: _data_loader.EpisodeReplayState | None = None,
    ) -> None:
        self.release_data_buffers(clear_batch=False, context=context, clear_jax_caches=clear_jax_caches)
        self.data_config = data_config
        self.data_loader = data_loader
        self.data_iter = data_iter
        self.batch = batch
        if episode_replay_state is not None:
            self.episode_replay_state = episode_replay_state
        self.skip_next_batch_prefetch_once = skip_next_batch_prefetch_once
        self.enable_device_prefetch_after_step = enable_device_prefetch_after_step


class AsyncCheckpointPublishJob:
    def __init__(
        self,
        *,
        step: int,
        checkpoint_manager,
        checkpoint_publisher: async_control.CheckpointPublisher,
        source_checkpoint_dir: Path,
        dataset_view_id: str | None,
        dataset_view_ids: dict[str, str | None] | None,
        policy_id: int,
        base_checkpoint_id: str | None,
    ):
        self.step = int(step)
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_publisher = checkpoint_publisher
        self._source_checkpoint_dir = source_checkpoint_dir
        self._dataset_view_id = dataset_view_id
        self._dataset_view_ids = dataset_view_ids
        self._policy_id = int(policy_id)
        self._base_checkpoint_id = base_checkpoint_id
        self._done_event = threading.Event()
        self._published_checkpoint: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"checkpoint-publish-{step}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._checkpoint_manager.wait_until_finished()
            self._published_checkpoint = self._checkpoint_publisher.publish(
                source_checkpoint_dir=self._source_checkpoint_dir,
                step=self.step,
                dataset_view_id=self._dataset_view_id,
                dataset_view_ids=self._dataset_view_ids,
                policy_id=self._policy_id,
                base_checkpoint_id=self._base_checkpoint_id,
            )
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
        finally:
            self._done_event.set()

    def done(self) -> bool:
        return self._done_event.is_set()

    def result(self) -> dict[str, Any] | None:
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Async checkpoint publish failed at step {self.step}.") from self._error
        return self._published_checkpoint

    def wait(self) -> dict[str, Any] | None:
        self._thread.join()
        return self.result()


class GpuUtilizationMonitor:
    def __init__(self, config: _config.GpuUtilizationMonitorConfig):
        self.config = config
        self.samples: list[float] = []
        self._stop_event = threading.Event()
        self._active_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = False
        self._collection_enabled = False
        self._visible_device_tokens = self._parse_visible_device_tokens()
        self._monitored_device_ids: tuple[str, ...] = ()

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self.config.sample_interval_s <= 0:
            raise ValueError(f"GPU monitor sample_interval_s must be > 0, got {self.config.sample_interval_s}")
        if not self._query_once():
            logging.warning("GPU utilization monitor is enabled but nvidia-smi is unavailable; skipping GPU monitor.")
            return
        self._available = True
        if self._monitored_device_ids:
            logging.info("GPU utilization monitor tracking GPU indices: %s", ",".join(self._monitored_device_ids))
        self._thread = threading.Thread(target=self._run, name="gpu-utilization-monitor", daemon=True)
        self._thread.start()

    def _parse_visible_device_tokens(self) -> set[str] | None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is None:
            return None
        tokens = {token.strip() for token in visible_devices.split(",") if token.strip() and token.strip() != "-1"}
        return tokens

    def _query_gpu_rows(self) -> list[tuple[str, str, float]]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            index, uuid, utilization = parts
            rows.append((index, uuid, float(utilization)))
        return rows

    def _filter_visible_rows(self, rows: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        if self._visible_device_tokens is None:
            return rows
        if not self._visible_device_tokens:
            return []
        filtered = [
            row for row in rows if row[0] in self._visible_device_tokens or row[1] in self._visible_device_tokens
        ]
        if not filtered and rows:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s did not match nvidia-smi GPU indices or UUIDs; monitoring all GPUs.",
                ",".join(sorted(self._visible_device_tokens)),
            )
            return rows
        return filtered

    def _query_once(self) -> bool:
        try:
            rows = self._filter_visible_rows(self._query_gpu_rows())
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
        self._monitored_device_ids = tuple(row[0] for row in rows)
        return True

    def _sample(self) -> list[float]:
        rows = self._filter_visible_rows(self._query_gpu_rows())
        return [row[2] for row in rows]

    def _run(self) -> None:
        while not self._stop_event.wait(self.config.sample_interval_s):
            if not self._collection_enabled or not self._active_event.is_set():
                continue
            try:
                values = self._sample()
            except subprocess.SubprocessError as exc:
                logging.warning("GPU utilization sample failed: %s", exc)
                continue
            if values:
                self.samples.append(float(np.mean(values)))

    def begin_collection(self, *, reset: bool = True) -> None:
        if not self._available:
            return
        if reset:
            self.reset()
        self._collection_enabled = True

    def resume(self) -> None:
        if self._available and self._collection_enabled:
            self._active_event.set()

    def pause(self) -> None:
        self._active_event.clear()

    def reset(self) -> None:
        self.samples.clear()

    def metrics(self) -> dict[str, float]:
        if not self.samples:
            return {}
        samples = np.asarray(self.samples, dtype=np.float32)
        nonzero_samples = samples[samples > 0]
        metrics = {
            "gpu_utilization/mean": float(np.mean(samples)),
            "gpu_utilization/min": float(np.min(samples)),
            "gpu_utilization/max": float(np.max(samples)),
            "gpu_utilization/samples": float(len(samples)),
        }
        if len(nonzero_samples) > 0:
            metrics.update(
                {
                    "gpu_utilization/nonzero_mean": float(np.mean(nonzero_samples)),
                    "gpu_utilization/nonzero_p10": float(np.percentile(nonzero_samples, 10)),
                }
            )
        return metrics

    def stop(self) -> dict[str, float]:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(1.0, self.config.sample_interval_s + 1.0))
        metrics = self.metrics()
        if metrics:
            logging.info(
                "GPU utilization summary: mean=%.2f%% min=%.2f%% max=%.2f%% samples=%d target=%.2f%%",
                metrics["gpu_utilization/mean"],
                metrics["gpu_utilization/min"],
                metrics["gpu_utilization/max"],
                int(metrics["gpu_utilization/samples"]),
                self.config.target_percent,
            )
            if self.config.warn_below_target and metrics["gpu_utilization/mean"] < self.config.target_percent:
                logging.warning(
                    "Average GPU utilization %.2f%% is below target %.2f%%.",
                    metrics["gpu_utilization/mean"],
                    self.config.target_percent,
                )
        elif self.config.enabled and self._available:
            logging.warning("GPU utilization monitor collected no post-warmup samples.")
        return metrics


def _repo_policy_id(repo_id: str) -> float | None:
    """Read a repo's constant policy_id from meta/policy_metadata_map.jsonl (first row). None if unknown."""
    path = _data_loader.lerobot_dataset.HF_LEROBOT_HOME / repo_id / "meta" / "policy_metadata_map.jsonl"
    try:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    return float(json.loads(line).get("policy_id"))
    except (OSError, ValueError, TypeError):
        return None
    return None


def _resolve_current_repo_id(data_config: _config.DataConfig) -> str | None:
    """The growing current-policy repo = the view's repo whose policy_id == sampler.current_policy_id.

    Returns None when the dual-source split is not applicable (no repo_specs, no current_policy_id, or the
    current policy's repo is not present / not identifiable) — callers then fall back to the normal loader.
    """
    specs = data_config.repo_specs
    sampler = data_config.sampler
    if not specs or sampler is None or getattr(sampler, "current_policy_id", None) is None:
        return None
    target = int(sampler.current_policy_id)
    matches = [spec.repo_id for spec in specs if _repo_policy_id(spec.repo_id) == float(target)]
    if len(matches) != 1:
        # 0 -> current policy has no repo yet; >1 -> ambiguous. Both fall back to the full-rebuild loader.
        return None
    if len(specs) < 2:
        # Only the current repo in the view: nothing "stable" to keep alive, so the dual split has no benefit.
        return None
    return matches[0]


def _make_data_loader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    data_sharding: jax.sharding.Sharding,
    *,
    skip_norm_stats: bool,
    force_nonpersistent_workers: bool = False,
    episode_replay_state: _data_loader.EpisodeReplayState | None = None,
    episode_replay_scope: str = "default",
) -> _data_loader.DataLoader:
    action_horizon = (
        config.model.async_action_horizon
        if getattr(config.model, "async_action_horizon", -1) > 0
        else config.model.action_horizon
    )
    persistent_workers = False if force_nonpersistent_workers else config.persistent_workers
    if config.async_training.dual_source_loader:
        current_repo_id = _resolve_current_repo_id(data_config)
        if current_repo_id is not None:
            logging.info("Building dual-source dataloader; current (live-growing) repo = %s", current_repo_id)
            return _data_loader.create_dual_source_data_loader(
                data_config,
                model_config=config.model,
                action_horizon=action_horizon,
                batch_size=config.batch_size,
                current_repo_id=current_repo_id,
                sharding=data_sharding,
                skip_norm_stats=skip_norm_stats,
                num_workers=config.num_workers,
                seed=config.seed,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=persistent_workers,
                reuse_datasets=True,
                episode_replay_state=episode_replay_state,
                episode_replay_scope=episode_replay_scope,
            )
    return _data_loader.create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=action_horizon,
        batch_size=config.batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=persistent_workers,
        skip_norm_stats=skip_norm_stats,
        reuse_datasets=True,
        episode_replay_state=episode_replay_state,
        episode_replay_scope=episode_replay_scope,
    )


def _is_transient_parquet_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in {"ArrowInvalid", "DatasetGenerationError"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _make_stream_data_loader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    data_sharding: jax.sharding.Sharding,
    *,
    attempts: int,
    episode_replay_state: _data_loader.EpisodeReplayState | None = None,
    episode_replay_scope: str = "default",
) -> _data_loader.DataLoader:
    for attempt in range(1, attempts + 1):
        try:
            return _make_data_loader(
                config,
                data_config,
                data_sharding,
                skip_norm_stats=not config.use_norm_stats,
                force_nonpersistent_workers=config.async_training.disable_persistent_workers_on_reload,
                episode_replay_state=episode_replay_state,
                episode_replay_scope=episode_replay_scope,
            )
        except Exception as exc:
            if not _is_transient_parquet_error(exc) or attempt == attempts:
                raise
            logging.warning(
                "Streaming parquet snapshot was incomplete while building a dataloader; retrying in 2 seconds "
                "(attempt %d/%d).",
                attempt,
                attempts,
            )
            time.sleep(2)
    raise AssertionError("unreachable")


def _prefetch_iter(
    config: _config.TrainConfig,
    data_loader: _data_loader.DataLoader,
    *,
    prefetch_size: int | None = None,
):
    return training_utils.threaded_prefetch(
        iter(data_loader),
        prefetch_size=config.prefetch_size if prefetch_size is None else prefetch_size,
        transform=(lambda item: jax.tree.map(jax.device_put, item)) if config.device_prefetch else None,
        error_message="Async training prefetch worker failed",
        generation=(lambda: data_loader.generation)
        if isinstance(data_loader, _data_loader.DualSourceDataLoaderImpl)
        else None,
        debug_name=f"{config.name}:{type(data_loader).__name__}",
    )


class WandbAsyncControlLogger:
    def __init__(self, config: _config.TrainConfig):
        self.config = config
        self.role = self._role()
        self.view_indices: dict[str, int] = {}
        self.view_table = wandb.Table(
            columns=[
                "step",
                "role",
                "stream",
                "view_index",
                "view_id",
                "target_policy_id",
                "train_type",
                "repo_ids",
                "episode_counts",
                "repo_count",
                "episode_count",
            ]
        )

    def _role(self) -> str:
        merge = self.config.async_training.checkpoint_publisher.merge
        if merge is not None:
            return merge.role
        return "async"

    def stream_alias(self, stream: AsyncTrainStream) -> str:
        if self.role in {"action", "value"}:
            return self.role
        return stream.name

    def log_dataset_view(self, *, stream: AsyncTrainStream, view: async_control.DatasetView, step: int) -> None:
        stream_key = stream.name
        view_index = self.view_indices.get(stream_key, -1) + 1
        self.view_indices[stream_key] = view_index
        repo_ids = [spec.repo_id for spec in view.repo_specs]
        episode_counts = [-1 if spec.episode_ids is None else len(spec.episode_ids) for spec in view.repo_specs]
        known_episode_count = sum(count for count in episode_counts if count >= 0)
        alias = self.stream_alias(stream)
        train_type = ",".join(sorted(train_type.value for train_type in stream.train_types))
        metrics = {
            f"online/{alias}/target_policy_id": -1 if view.target_policy_id is None else int(view.target_policy_id),
            f"online/{alias}/view_index": view_index,
            f"online/{alias}/view_update": 1,
            f"online/{alias}/repo_count": len(repo_ids),
            f"online/{alias}/episode_count": known_episode_count,
            f"online/{alias}/train_type_code": _train_type_code(stream.train_types),
        }
        if stream.last_policy_update_step is not None:
            metrics[f"online/{alias}/stage_step"] = max(0, step - int(stream.last_policy_update_step))
        self.view_table.add_data(
            step,
            self.role,
            stream.name,
            view_index,
            view.view_id,
            view.target_policy_id,
            train_type,
            ",".join(repo_ids),
            ",".join(str(count) for count in episode_counts),
            len(repo_ids),
            known_episode_count,
        )
        try:
            wandb.log({**metrics, "online/dataset_views": self.view_table}, step=step)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not log dataset-view metadata to W&B: %s", exc)

    def log_checkpoint_event(
        self,
        *,
        step: int,
        event: str,
        checkpoint_ref: dict[str, Any] | None = None,
        policy_id: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            f"checkpoint/{self.role}/{event}": 1,
        }
        resolved_policy_id = policy_id
        if resolved_policy_id is None and checkpoint_ref is not None and checkpoint_ref.get("policy_id") is not None:
            resolved_policy_id = int(checkpoint_ref["policy_id"])
        if resolved_policy_id is not None:
            payload[f"checkpoint/{self.role}/policy_id"] = int(resolved_policy_id)
        try:
            wandb.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not log checkpoint metadata to W&B: %s", exc)


def _train_type_code(train_types: set[_model.TrainingType]) -> int:
    if train_types == {_model.TrainingType.BC}:
        return 0
    if train_types == {_model.TrainingType.VALUE_MC_LEARNING}:
        return 1
    if train_types == {_model.TrainingType.TD_LEARNING}:
        return 2
    return -1


def _train_types_from_values(values: tuple[str, ...]) -> set[_model.TrainingType]:
    return {_model.TrainingType(value) for value in values}


def _log_startup_previews(
    config: _config.TrainConfig,
    batch,
) -> None:
    if not config.startup_previews:
        return

    observation, _, _, _, _, _, _, _, _, _ = batch

    def _preview_frames(image):
        image = np.array(image)
        if image.ndim == 4:
            return [image[t] for t in range(image.shape[0])]
        return [image]

    images_to_log = []
    for sample_idx in range(min(5, len(next(iter(observation.images.values()))))):
        camera_histories = [_preview_frames(img[sample_idx]) for img in observation.images.values()]
        history_len = max(len(history) for history in camera_histories)
        for history_idx in range(history_len):
            camera_frames = [history[min(history_idx, len(history) - 1)] for history in camera_histories]
            caption = (
                f"sample={sample_idx}, history={history_idx - history_len + 1}"
                if history_len > 1
                else f"sample={sample_idx}"
            )
            images_to_log.append(wandb.Image(np.concatenate(camera_frames, axis=1), caption=caption))
    wandb.log({"camera_views": images_to_log}, step=0)


def _checkpoint_step_dir(checkpoint_dir: epath.Path, step: int) -> Path:
    return Path(str(checkpoint_dir)) / str(int(step))


def _latest_local_checkpoint_step(checkpoint_dir: epath.Path) -> int | None:
    path = Path(str(checkpoint_dir))
    if not path.exists():
        return None
    steps = [int(child.name) for child in path.iterdir() if child.is_dir() and child.name.isdigit()]
    return max(steps) if steps else None


def _set_train_state_step(train_state, step: int):
    return dataclasses.replace(train_state, step=jnp.asarray(step, dtype=train_state.step.dtype))


def _published_checkpoint_train_step(checkpoint_ref: dict[str, Any]) -> int | None:
    train_step = checkpoint_ref.get("train_step")
    if train_step is None:
        return None
    try:
        return int(train_step)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Published checkpoint has invalid train_step={train_step!r}.") from exc


def _split_next_policy_id(streams: list[AsyncTrainStream]) -> int | None:
    if not all(stream.target_policy_id is not None for stream in streams):
        return None
    target_policy_ids = {int(stream.target_policy_id) for stream in streams}
    if len(target_policy_ids) != 1:
        raise ValueError(
            "Split checkpoint merge requires all streams to use the same target_policy_id, got "
            f"{sorted(target_policy_ids)}."
        )
    return next(iter(target_policy_ids)) + 1


def _should_publish_split_round_checkpoint(
    config: _config.TrainConfig, streams: list[AsyncTrainStream], step: int
) -> bool:
    if config.async_training.checkpoint_publisher.merge is None:
        return False
    round_steps = config.async_training.split_round_steps
    if round_steps is None:
        return False
    policy_id = _split_next_policy_id(streams)
    if policy_id is None:
        return False
    update_steps = [stream.last_policy_update_step for stream in streams]
    if not all(update_step is not None for update_step in update_steps):
        return False
    policy_boundary_step = max(int(update_step) for update_step in update_steps if update_step is not None)
    publish_step = policy_boundary_step + max(1, int(round_steps))
    should_publish = step == publish_step
    if should_publish:
        logging.info(
            "Split round boundary reached for policy_id=%d at step %d (policy boundary=%d, round_steps=%d).",
            policy_id,
            step,
            policy_boundary_step,
            round_steps,
        )
    return should_publish


def _create_ptrain_step(
    config: _config.TrainConfig,
    train_types: set[_model.TrainingType],
    *,
    replicated_sharding: jax.sharding.Sharding,
    train_state_sharding: Any,
    data_sharding: jax.sharding.Sharding,
):
    return jax.jit(
        functools.partial(sync_train.train_step, config, train_types),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )


def _wait_for_initial_dataset_view(
    watcher: async_control.DatasetViewWatcher,
    *,
    stream_name: str,
    target_policy_id: int | None,
    poll_interval_s: float = 5.0,
) -> async_control.DatasetView | None:
    initial_view = watcher.poll(0, target_policy_id=target_policy_id)
    if initial_view is not None or target_policy_id is None:
        return initial_view

    last_log_time = 0.0
    while initial_view is None:
        now = time.monotonic()
        if now - last_log_time >= 30:
            logging.info(
                "Async stream '%s' waiting for initial dataset view target_policy_id=%d before building dataloader.",
                stream_name,
                target_policy_id,
            )
            last_log_time = now
        time.sleep(poll_interval_s)
        initial_view = watcher.poll(0, force=True, target_policy_id=target_policy_id)
    return initial_view


def _create_stream(
    config: _config.TrainConfig,
    *,
    name: str,
    data_factory: Any,
    dataset_view_config: _config.DatasetViewConfig,
    train_type_values: tuple[str, ...],
    train_type_by_target_policy_id: dict[int, tuple[str, ...]],
    steps_per_cycle: int,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    train_state_sharding: Any,
    wandb_logger: WandbAsyncControlLogger,
    initial_target_policy_id: int | None = None,
    defer_initial_device_prefetch: bool = False,
) -> AsyncTrainStream:
    data_config = data_factory.create(config.assets_dirs, config.model)
    watcher = async_control.DatasetViewWatcher(dataset_view_config)
    initial_view = _wait_for_initial_dataset_view(
        watcher,
        stream_name=name,
        target_policy_id=initial_target_policy_id,
    )
    view_id = None
    target_policy_id = None
    if initial_view is not None:
        data_config = async_control.apply_dataset_view(data_config, initial_view)
        view_id = initial_view.view_id
        target_policy_id = initial_view.target_policy_id

    episode_replay_state = _data_loader.EpisodeReplayState(seed=config.seed)
    data_loader = _make_stream_data_loader(
        config,
        data_config,
        data_sharding,
        attempts=10,
        episode_replay_state=episode_replay_state,
        episode_replay_scope=name,
    )
    data_iter = _prefetch_iter(config, data_loader, prefetch_size=0 if defer_initial_device_prefetch else None)
    batch = next(data_iter)
    effective_train_type_values = (
        tuple(train_type_by_target_policy_id[int(target_policy_id)])
        if target_policy_id is not None and int(target_policy_id) in train_type_by_target_policy_id
        else train_type_values
    )
    train_types = _train_types_from_values(effective_train_type_values)
    ptrain_step = _create_ptrain_step(
        config,
        train_types,
        replicated_sharding=replicated_sharding,
        train_state_sharding=train_state_sharding,
        data_sharding=data_sharding,
    )
    logging.info(
        "Initialized async stream '%s' train_type=%s view=%s:\n%s",
        name,
        sorted(train_type.value for train_type in train_types),
        view_id,
        training_utils.array_tree_to_info(batch),
    )
    stream = AsyncTrainStream(
        name=name,
        default_train_type_values=train_type_values,
        train_type_by_target_policy_id=train_type_by_target_policy_id,
        train_types=train_types,
        data_config=data_config,
        watcher=watcher,
        data_loader=data_loader,
        data_iter=data_iter,
        batch=batch,
        steps_per_cycle=steps_per_cycle,
        ptrain_step=ptrain_step,
        episode_replay_state=episode_replay_state,
        view_id=view_id,
        target_policy_id=target_policy_id,
    )
    if defer_initial_device_prefetch:
        stream.skip_next_batch_prefetch_once = True
        stream.enable_device_prefetch_after_step = True
        logging.info(
            "Async stream '%s' deferring background device prefetch until after the first resumed step.",
            stream.name,
        )
    if initial_view is not None:
        wandb_logger.log_dataset_view(stream=stream, view=initial_view, step=0)
    return stream


def _stream_schedule(streams: list[AsyncTrainStream]) -> list[AsyncTrainStream]:
    schedule = []
    for stream in streams:
        schedule.extend([stream] * stream.steps_per_cycle)
    if not schedule:
        raise ValueError("Async training requires at least one train stream.")
    return schedule


def _latest_published_checkpoint(config: _config.TrainConfig) -> dict[str, Any] | None:
    output_dir = config.async_training.checkpoint_publisher.output_dir
    if not output_dir:
        return None
    latest_path = Path(output_dir).expanduser() / "checkpoints" / "latest.json"
    if not latest_path.exists():
        return None
    with latest_path.open() as f:
        latest = json.load(f)
    async_control.validate_checkpoint_ref(latest)
    return latest


def _maybe_reload_stream(
    config: _config.TrainConfig,
    stream: AsyncTrainStream,
    *,
    step: int,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    train_state_sharding: Any,
    wandb_logger: WandbAsyncControlLogger,
    force_poll: bool = False,
    target_policy_id: int | None = None,
) -> None:
    """Non-blocking dataset-view reload.

    The costly part of a reload — building the new dataloader and fetching its first batch (dominated by
    torch worker spawn + video decode, ~2 minutes on a large view) — runs in a background thread while the
    training loop keeps stepping on the current pipeline. This function is called every step: it (1) swaps
    in a finished background build (fast, main thread), then (2) if idle, polls for a new view and kicks off
    the next background build. The main loop therefore never stalls for the whole build; it only pays the
    cheap swap. Newer views published while a build is in flight are coalesced: the next poll picks the latest.
    """
    # (1) A background build finished -> swap it in on the main thread (cheap).
    if not stream.reload_build_in_flight():
        result = stream.take_reload_result()
        if result is not None:
            _finish_stream_reload(
                config,
                stream,
                result,
                step=step,
                replicated_sharding=replicated_sharding,
                train_state_sharding=train_state_sharding,
                data_sharding=data_sharding,
                wandb_logger=wandb_logger,
            )

    # (2) Idle -> poll for a new view and start building it in the background.
    if not stream.reload_build_in_flight():
        new_view = stream.watcher.poll(step, force=force_poll, target_policy_id=target_policy_id)
        if new_view is not None:
            # Dual-source fast path: a new view within the SAME generation (same target_policy_id and same
            # repo set) means only the current repo grew. Grow the live source in place — no rebuild, no
            # background build, no first-batch stall — and keep training on the same pipeline.
            if _try_grow_dual_source_in_place(stream, new_view, step=step, wandb_logger=wandb_logger):
                return
            _start_stream_reload_build(config, stream, new_view, step=step, data_sharding=data_sharding)


def _try_grow_dual_source_in_place(
    stream: AsyncTrainStream,
    view: async_control.DatasetView,
    *,
    step: int,
    wandb_logger: WandbAsyncControlLogger,
) -> bool:
    """If the stream uses a dual-source loader and the view stayed within the current generation, grow the
    live current-repo source in place and return True (skip the rebuild). Otherwise return False."""
    loader = getattr(stream, "data_loader", None)
    if not isinstance(loader, _data_loader.DualSourceDataLoaderImpl):
        return False
    same_generation = view.target_policy_id == stream.target_policy_id
    same_repos = {s.repo_id for s in view.repo_specs} == {s.repo_id for s in stream.data_config.repo_specs}
    if not (same_generation and same_repos):
        return False  # generation/repo-set changed -> fall through to a full rebuild
    live = loader.dual.live_current_repo()
    if live is None:
        return False
    old_specs = {s.repo_id: s for s in stream.data_config.repo_specs}
    new_specs = {s.repo_id: s for s in view.repo_specs}
    if any(old_specs[k] != new_specs[k] for k in old_specs if k != live.repo_id):
        return False  # Changed historical data needs new stable workers.
    spec = new_specs[live.repo_id]
    if spec.missing_feature_defaults != old_specs[live.repo_id].missing_feature_defaults:
        return False
    if view.raw.get("annotation_revision") is not None and spec.advantage_snapshot is None:
        return False  # Legacy mutable labels have no pinned revision.
    t0 = time.monotonic()
    generation = loader.generation
    n_new = loader.poll_and_grow_current(episode_ids=spec.episode_ids, advantage_snapshot=spec.advantage_snapshot)
    stream.data_config = async_control.apply_dataset_view(stream.data_config, view)
    if loader.generation != generation:
        # Drop the already-fetched old batch too; threaded_prefetch discards queued old generations.
        stream.wait_for_pending_batch()
        stream.batch = next(stream.data_iter)
    grow_ms = (time.monotonic() - t0) * 1e3
    stream.view_id = view.view_id
    stream.last_view_update_step = step
    logging.info(
        "DUAL_GROW stream=%s view=%s step=%d added_episodes=%d grow=%.0fms (no rebuild)",
        stream.name,
        view.view_id,
        step,
        n_new,
        grow_ms,
    )
    wandb_logger.log_dataset_view(stream=stream, view=view, step=step)
    return True


def _start_stream_reload_build(
    config: _config.TrainConfig,
    stream: AsyncTrainStream,
    view: async_control.DatasetView,
    *,
    step: int,
    data_sharding: jax.sharding.Sharding,
) -> None:
    logging.info(
        "Reloading dataloader for stream '%s' dataset view %s at step %d (background build; training continues)",
        stream.name,
        view.view_id,
        step,
    )
    base_data_config = stream.data_config
    new_episode_replay_state = stream.episode_replay_state.snapshot()

    def _build() -> dict[str, Any]:
        t0 = time.monotonic()
        new_data_config = async_control.apply_dataset_view(base_data_config, view)
        t_apply = time.monotonic()
        new_data_iter = None
        try:
            new_data_loader = _make_stream_data_loader(
                config,
                new_data_config,
                data_sharding,
                attempts=3,
                episode_replay_state=new_episode_replay_state,
                episode_replay_scope=stream.name,
            )
            t_loader = time.monotonic()
            # Build with full device prefetch directly in the background. The synchronous-reload path used a
            # deferred (prefetch_size=0) iterator plus a post-step "reload warmup" that re-created the iterator
            # and re-decoded the first batch on the main thread — a second ~2-minute stall. Because this build
            # runs off the training loop, we warm the full prefetch pipeline here and skip the warmup entirely.
            new_data_iter = _prefetch_iter(config, new_data_loader)
            new_batch = next(new_data_iter)
            t_batch = time.monotonic()
        except BaseException:
            close = getattr(new_data_iter, "close", None)
            if close is not None:
                close()
            raise
        return {
            "view": view,
            "data_config": new_data_config,
            "data_loader": new_data_loader,
            "data_iter": new_data_iter,
            "batch": new_batch,
            "episode_replay_state": new_episode_replay_state,
            "apply_ms": (t_apply - t0) * 1e3,
            "loader_ms": (t_loader - t_apply) * 1e3,
            "first_batch_ms": (t_batch - t_loader) * 1e3,
        }

    stream.start_reload_build(_build)


def _finish_stream_reload(
    config: _config.TrainConfig,
    stream: AsyncTrainStream,
    result: dict[str, Any],
    *,
    step: int,
    replicated_sharding: jax.sharding.Sharding,
    train_state_sharding: Any,
    data_sharding: jax.sharding.Sharding,
    wandb_logger: WandbAsyncControlLogger,
) -> None:
    view = result["view"]
    if "error" in result:
        exc = result["error"]
        close = getattr(result.get("data_iter"), "close", None)
        if close is not None:
            close()
        if not _is_transient_parquet_error(exc):
            raise exc
        logging.warning(
            "Deferring dataset view %s for stream '%s' because its parquet snapshot is still being written. "
            "Continuing with view %s.",
            view.view_id,
            stream.name,
            stream.view_id,
        )
        stream.watcher.retry_last_view()
        return

    build_ms = (time.monotonic() - stream.reload_build_t0) * 1e3 if stream.reload_build_t0 is not None else -1.0
    # Decide whether the swap rebuilds ptrain_step (train_type change); only then clear the jax compile cache.
    new_train_type_values = stream.train_type_values_for_policy(view.target_policy_id)
    new_train_types = _train_types_from_values(new_train_type_values)
    train_type_changed = new_train_types != stream.train_types

    swap_t0 = time.monotonic()
    stream.replace_data_pipeline(
        data_config=result["data_config"],
        data_loader=result["data_loader"],
        data_iter=result["data_iter"],
        batch=result["batch"],
        skip_next_batch_prefetch_once=True,
        enable_device_prefetch_after_step=False,
        context=f"dataset view reload to {view.view_id}",
        clear_jax_caches=train_type_changed,
        episode_replay_state=result["episode_replay_state"],
    )
    stream.view_id = view.view_id
    stream.last_view_update_step = step
    if train_type_changed:
        logging.info(
            "Async stream '%s' switching train_type from %s to %s for target_policy_id=%s at step %d",
            stream.name,
            sorted(train_type.value for train_type in stream.train_types),
            sorted(train_type.value for train_type in new_train_types),
            view.target_policy_id,
            step,
        )
        stream.train_types = new_train_types
        stream.ptrain_step = _create_ptrain_step(
            config,
            stream.train_types,
            replicated_sharding=replicated_sharding,
            train_state_sharding=train_state_sharding,
            data_sharding=data_sharding,
        )
    if view.target_policy_id != stream.target_policy_id:
        policy_boundary_step = max(0, int(step) - 1)
        logging.info(
            "Async stream '%s' advanced target_policy_id from %s to %s at step %d; policy boundary step=%d.",
            stream.name,
            stream.target_policy_id,
            view.target_policy_id,
            step,
            policy_boundary_step,
        )
        stream.target_policy_id = view.target_policy_id
        stream.last_policy_update_step = policy_boundary_step
    swap_ms = (time.monotonic() - swap_t0) * 1e3
    logging.info(
        "RELOAD_BENCH stream=%s view=%s step=%d bg_build_total=%.0fms (apply=%.0f loader=%.0f first_batch=%.0f) "
        "main_swap=%.0fms train_type_changed=%s",
        stream.name,
        view.view_id,
        step,
        build_ms,
        result["apply_ms"],
        result["loader_ms"],
        result["first_batch_ms"],
        swap_ms,
        train_type_changed,
    )
    # Measure the main-thread stall (swap -> next completed training step). This is the latency the training
    # loop actually experiences per reload; the background build above overlapped ongoing training.
    stream.reload_bench_pending = (step, swap_t0)
    wandb_logger.log_dataset_view(stream=stream, view=view, step=step)


def _load_split_stage_checkpoint(
    config: _config.TrainConfig,
    *,
    checkpoint_publisher: async_control.CheckpointPublisher,
    policy_id: int,
    train_state,
    train_state_sharding,
    step: int,
    wandb_logger: WandbAsyncControlLogger,
):
    logging.info("Split stage view advanced; waiting for latest merged policy_id=%d before training it.", policy_id)
    wandb_logger.log_checkpoint_event(step=step, event="waiting_for_stage_checkpoint", policy_id=policy_id)
    merged_checkpoint = checkpoint_publisher.wait_for_latest_merged_policy(policy_id=policy_id)
    wandb_logger.log_checkpoint_event(
        step=step,
        event="stage_checkpoint_ready",
        checkpoint_ref=merged_checkpoint,
        policy_id=policy_id,
    )
    logging.info(
        "Loading merged checkpoint %s before entering target_policy_id=%d.",
        merged_checkpoint.get("checkpoint_id"),
        policy_id,
    )
    train_state = async_control.switch_train_state_checkpoint(
        config,
        train_state,
        merged_checkpoint,
        reset_optimizer=config.async_training.checkpoint_switch.reset_optimizer,
        reset_ema=config.async_training.checkpoint_switch.reset_ema,
        preserve_global_step=config.async_training.checkpoint_switch.preserve_global_step,
        state_sharding=train_state_sharding,
    )
    train_state = async_control.finalize_checkpoint_switch(
        train_state,
        context="split merged checkpoint load",
        reclaim_memory=config.async_training.checkpoint_publisher.merge is not None,
    )
    logging.info("Split stage handoff complete: loaded merged policy_id=%d.", policy_id)
    return train_state, str(merged_checkpoint.get("checkpoint_id"))


def _wait_for_final_split_merge(
    *,
    checkpoint_publisher: async_control.CheckpointPublisher,
    role_checkpoint: dict[str, Any],
    step: int,
    wandb_logger: WandbAsyncControlLogger,
) -> None:
    policy_id = int(role_checkpoint["policy_id"])
    train_step = int(role_checkpoint["train_step"])
    logging.info(
        "Final split checkpoint %s published for policy_id=%d step=%d; waiting for merged checkpoint before exit.",
        role_checkpoint.get("checkpoint_id"),
        policy_id,
        train_step,
    )
    wandb_logger.log_checkpoint_event(
        step=step,
        event="waiting_for_final_merged_checkpoint",
        checkpoint_ref=role_checkpoint,
        policy_id=policy_id,
    )
    merged_checkpoint = checkpoint_publisher.wait_for_merged_policy(
        policy_id=policy_id,
        role_checkpoint=role_checkpoint,
    )
    wandb_logger.log_checkpoint_event(
        step=step,
        event="final_merged_checkpoint_ready",
        checkpoint_ref=merged_checkpoint,
        policy_id=policy_id,
    )
    logging.info(
        "Final merged checkpoint %s is ready for policy_id=%d step=%d.",
        merged_checkpoint.get("checkpoint_id"),
        policy_id,
        train_step,
    )


def _wait_for_split_stage_views(
    config: _config.TrainConfig,
    *,
    streams: list[AsyncTrainStream],
    pending_next_policy_id: int,
    step: int,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    train_state_sharding: Any,
    wandb_logger: WandbAsyncControlLogger,
    poll_interval_s: float = 5.0,
) -> None:
    """Block (without training) until every stream's dataset view advances to pending_next_policy_id.

    Called after the trainer publishes the merged snapshot that drives the next stage handoff. Training
    is paused here so the EMA buffer does not drift away from the published snapshot; on resume the loaded
    snapshot params match the (undrifted) EMA. This does not advance the global step counter or run any
    device compute, so a slow rollout/convert cycle cannot exhaust the num_train_steps budget while waiting.
    """
    last_log_time = 0.0
    while not all(stream.target_policy_id == pending_next_policy_id for stream in streams):
        for stream in streams:
            if stream.target_policy_id == pending_next_policy_id:
                continue
            _maybe_reload_stream(
                config,
                stream,
                step=step,
                data_sharding=data_sharding,
                replicated_sharding=replicated_sharding,
                train_state_sharding=train_state_sharding,
                wandb_logger=wandb_logger,
                force_poll=True,
                target_policy_id=pending_next_policy_id,
            )
        if all(stream.target_policy_id == pending_next_policy_id for stream in streams):
            break
        now = time.monotonic()
        if now - last_log_time >= 30:
            current = {stream.name: stream.target_policy_id for stream in streams}
            logging.info(
                "Paused training after publishing split snapshot for policy_id=%d; waiting for dataset views to "
                "advance, current targets=%s.",
                pending_next_policy_id,
                current,
            )
            last_log_time = now
        time.sleep(poll_interval_s)


def _maybe_reset_gpu_monitor_warmup(
    config: _config.TrainConfig,
    *,
    step: int,
    start_step: int,
    gpu_monitor: GpuUtilizationMonitor,
    last_log_time: float,
) -> float:
    """Begin GPU-utilization collection once the warmup window ends; return the (possibly reset) log clock."""
    if (
        config.async_training.gpu_monitor.enabled
        and config.async_training.gpu_monitor.warmup_steps > 0
        and step == start_step + config.async_training.gpu_monitor.warmup_steps
    ):
        gpu_monitor.begin_collection()
        last_log_time = time.monotonic()
        logging.info(
            "Reset GPU utilization monitor after %d warmup steps.", config.async_training.gpu_monitor.warmup_steps
        )
    return last_log_time


def _reload_all_streams(
    config: _config.TrainConfig,
    *,
    streams: list[AsyncTrainStream],
    step: int,
    data_sharding,
    replicated_sharding,
    train_state_sharding,
    wandb_logger: WandbAsyncControlLogger,
) -> None:
    """Non-blocking dataset-view reload poll for every stream (mutates stream objects in place)."""
    for stream in streams:
        target_policy_id = (
            stream.target_policy_id if config.async_training.checkpoint_publisher.merge is not None else None
        )
        _maybe_reload_stream(
            config,
            stream,
            step=step,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            train_state_sharding=train_state_sharding,
            wandb_logger=wandb_logger,
            target_policy_id=target_policy_id,
        )


def _log_periodic_metrics(
    config: _config.TrainConfig,
    *,
    step: int,
    streams: list[AsyncTrainStream],
    streams_by_name: dict[str, AsyncTrainStream],
    infos_by_stream: dict[str, list[dict[str, Any]]],
    last_log_time: float,
    gpu_monitor: GpuUtilizationMonitor,
    wandb_logger: WandbAsyncControlLogger,
    pbar,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    """Reduce/log per-stream metrics; return a fresh infos buffer and reset log clock."""
    reduced_info = {}
    num_logged_infos = 0
    for stream_name, stream_infos in infos_by_stream.items():
        if not stream_infos:
            continue
        stacked_infos = common_utils.stack_forest(stream_infos)
        stream_reduced = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
        reduced_info.update({f"{stream_name}/{key}": value for key, value in stream_reduced.items()})
        if stream_name in streams_by_name:
            alias = wandb_logger.stream_alias(streams_by_name[stream_name])
            reduced_info.update({f"train/{alias}/{key}": value for key, value in stream_reduced.items()})
            reduced_info[f"train/{alias}/train_type_code"] = _train_type_code(streams_by_name[stream_name].train_types)
        num_logged_infos += len(stream_infos)
    elapsed = max(time.monotonic() - last_log_time, 1e-6)
    reduced_info["steps_per_second"] = num_logged_infos / elapsed
    reduced_info.update(gpu_monitor.metrics())
    info_str = ", ".join(f"{key}={value:.4f}" for key, value in reduced_info.items())
    pbar.write(f"Step {step}: {info_str}")
    wandb.log(reduced_info, step=step)
    infos_by_stream = {stream.name: [] for stream in streams}
    last_log_time = time.monotonic()
    if step != config.num_train_steps - 1:
        gpu_monitor.reset()
    return infos_by_stream, last_log_time


def _run_training_loop(
    *,
    config: _config.TrainConfig,
    checkpoint_manager,
    checkpoint_publisher: async_control.CheckpointPublisher,
    checkpoint_switch_watcher: async_control.CheckpointSwitchWatcher,
    train_state,
    train_state_sharding,
    full_checkpoint_template,
    train_rng,
    mesh,
    data_sharding,
    replicated_sharding,
    streams: list[AsyncTrainStream],
    schedule: list[AsyncTrainStream],
    primary_stream: AsyncTrainStream,
    start_step: int,
    gpu_monitor: GpuUtilizationMonitor,
    wandb_logger: WandbAsyncControlLogger,
    validation_loaders,
    validation_video_threads: list[threading.Thread],
):
    if config.async_training.compute_burst_steps <= 0:
        raise ValueError(
            f"async_training.compute_burst_steps must be > 0, got {config.async_training.compute_burst_steps}"
        )
    active_base_checkpoint_id: str | None = None
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    infos_by_stream: dict[str, list[dict[str, Any]]] = {stream.name: [] for stream in streams}
    streams_by_name = {stream.name: stream for stream in streams}
    last_log_time = time.monotonic()
    publish_jobs: list[AsyncCheckpointPublishJob] = []
    saved_view_update_keys: set[tuple[tuple[str, str], ...]] = set()
    view_update_targets: dict[tuple[tuple[str, str], ...], int] = {}
    scheduled_checkpoint_keys: set[tuple[int, int]] = set()
    pending_next_policy_id: int | None = None
    if config.async_training.gpu_monitor.enabled and config.async_training.gpu_monitor.warmup_steps == 0:
        gpu_monitor.begin_collection()

    def _consume_finished_publish_jobs(*, wait: bool = False) -> None:
        nonlocal train_state, active_base_checkpoint_id, pending_next_policy_id
        remaining_jobs = []
        for job in publish_jobs:
            if not wait and not job.done():
                remaining_jobs.append(job)
                continue
            published_checkpoint = job.wait() if wait else job.result()
            if published_checkpoint is not None:
                wandb_logger.log_checkpoint_event(
                    step=job.step,
                    event="published",
                    checkpoint_ref=published_checkpoint,
                )
            if published_checkpoint is not None and config.async_training.checkpoint_publisher.reload_after_publish:
                if config.async_training.checkpoint_publisher.merge is not None:
                    pending_next_policy_id = int(published_checkpoint["policy_id"])
                else:
                    train_state = async_control.switch_train_state_checkpoint(
                        config,
                        train_state,
                        published_checkpoint,
                        reset_optimizer=config.async_training.checkpoint_switch.reset_optimizer,
                        reset_ema=config.async_training.checkpoint_switch.reset_ema,
                        preserve_global_step=config.async_training.checkpoint_switch.preserve_global_step,
                        state_sharding=train_state_sharding,
                    )
                    active_base_checkpoint_id = str(published_checkpoint.get("checkpoint_id"))
                    train_state = async_control.finalize_checkpoint_switch(
                        train_state,
                        context="async published checkpoint reload",
                        reclaim_memory=config.async_training.checkpoint_publisher.merge is not None,
                    )
        publish_jobs[:] = remaining_jobs

    for step in pbar:
        _consume_finished_publish_jobs()
        last_log_time = _maybe_reset_gpu_monitor_warmup(
            config, step=step, start_step=start_step, gpu_monitor=gpu_monitor, last_log_time=last_log_time
        )

        _reload_all_streams(
            config,
            streams=streams,
            step=step,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            train_state_sharding=train_state_sharding,
            wandb_logger=wandb_logger,
        )
        if pending_next_policy_id is not None:
            # Pause training after publishing the merged snapshot that drives the next stage handoff.
            # Blocking here (rather than continuing ptrain_step on stale policy-N data) keeps the EMA buffer
            # from drifting away from the published snapshot, so on resume params == EMA == snapshot. The wait
            # does not advance `step` or run device compute, so it cannot consume the num_train_steps budget.
            _wait_for_split_stage_views(
                config,
                streams=streams,
                pending_next_policy_id=pending_next_policy_id,
                step=step,
                data_sharding=data_sharding,
                replicated_sharding=replicated_sharding,
                train_state_sharding=train_state_sharding,
                wandb_logger=wandb_logger,
            )
            logging.info(
                "Split stage dataset views advanced to target_policy_id=%d; resuming training from published snapshot.",
                pending_next_policy_id,
            )
            if config.async_training.checkpoint_publisher.reload_after_publish:
                train_state, active_base_checkpoint_id = _load_split_stage_checkpoint(
                    config,
                    checkpoint_publisher=checkpoint_publisher,
                    policy_id=pending_next_policy_id,
                    train_state=train_state,
                    train_state_sharding=train_state_sharding,
                    step=step,
                    wandb_logger=wandb_logger,
                )
            pending_next_policy_id = None

        checkpoint_ref = checkpoint_switch_watcher.poll(step)
        if checkpoint_ref is not None:
            train_state = async_control.switch_train_state_checkpoint(
                config,
                train_state,
                checkpoint_ref,
                reset_optimizer=config.async_training.checkpoint_switch.reset_optimizer,
                reset_ema=config.async_training.checkpoint_switch.reset_ema,
                preserve_global_step=config.async_training.checkpoint_switch.preserve_global_step,
                state_sharding=train_state_sharding,
            )
            active_base_checkpoint_id = str(checkpoint_ref.get("checkpoint_id"))
            train_state = async_control.finalize_checkpoint_switch(
                train_state,
                context="checkpoint switch watcher reload",
                reclaim_memory=config.async_training.checkpoint_publisher.merge is not None,
            )

        step_start_time = time.monotonic()
        gpu_monitor.resume()
        compute_start_time = time.monotonic()
        batch_wait_elapsed = 0.0
        try:
            burst_start = (step - start_step) * config.async_training.compute_burst_steps
            burst_streams = [
                schedule[(burst_start + burst_idx) % len(schedule)]
                for burst_idx in range(config.async_training.compute_burst_steps)
            ]
            pending_infos = []
            streams_to_enable_prefetch: list[AsyncTrainStream] = []
            for stream in burst_streams:
                skip_next_batch_prefetch = stream.skip_next_batch_prefetch_once
                stream.skip_next_batch_prefetch_once = False
                if skip_next_batch_prefetch:
                    logging.info(
                        "Async stream '%s' using synchronous next-batch fetch for the first step after reload.",
                        stream.name,
                    )
                else:
                    stream.start_next_batch_fetch()
                with sharding.set_mesh(mesh):
                    train_state, info = stream.ptrain_step(train_rng, train_state, stream.batch)
                batch_wait_start_time = time.monotonic()
                stream.finish_next_batch_fetch()
                stream_batch_wait_elapsed = time.monotonic() - batch_wait_start_time
                batch_wait_elapsed += stream_batch_wait_elapsed
                if (
                    _env_flag("VAM_ASYNC_DATALOADER_DEBUG")
                    and step % _debug_every("VAM_ASYNC_DATALOADER_DEBUG_EVERY") == 0
                ):
                    logging.info(
                        "DATALOADER_TRAIN_LOOP stream=%s step=%d skip_prefetch=%s fetch_ms=%.0f "
                        "main_join_ms=%.0f stream_wait_ms=%.0f",
                        stream.name,
                        step,
                        skip_next_batch_prefetch,
                        (stream.last_next_batch_fetch_seconds or 0.0) * 1e3,
                        (stream.last_next_batch_join_seconds or 0.0) * 1e3,
                        stream_batch_wait_elapsed * 1e3,
                    )
                pending_infos.append(info)
                infos_by_stream[stream.name].append(info)
                if stream.enable_device_prefetch_after_step:
                    streams_to_enable_prefetch.append(stream)
            jax.block_until_ready((train_state, pending_infos))
            for stream in burst_streams:
                pending = stream.reload_bench_pending
                if pending is not None:
                    reload_step, reload_t0 = pending
                    logging.info(
                        "RELOAD_TO_STEP stream=%s reload_step=%d next_step=%d view_update_to_next_step_done=%.0fms",
                        stream.name,
                        reload_step,
                        step,
                        (time.monotonic() - reload_t0) * 1e3,
                    )
                    stream.reload_bench_pending = None
            for stream in streams_to_enable_prefetch:
                logging.info("Async stream '%s' restoring background device prefetch after reload warmup.", stream.name)
                new_data_iter = _prefetch_iter(config, stream.data_loader)
                new_batch = next(new_data_iter)
                stream.replace_data_pipeline(
                    data_config=stream.data_config,
                    data_loader=stream.data_loader,
                    data_iter=new_data_iter,
                    batch=new_batch,
                    skip_next_batch_prefetch_once=False,
                    enable_device_prefetch_after_step=False,
                    context="reload warmup",
                )
        finally:
            gpu_monitor.pause()
        compute_elapsed = time.monotonic() - compute_start_time
        if config.async_training.compute_burst_steps > 1:
            infos_by_stream.setdefault("burst", []).append(
                {
                    "step_compute_seconds": jnp.asarray(compute_elapsed, dtype=jnp.float32),
                    "next_batch_wait_seconds": jnp.asarray(batch_wait_elapsed, dtype=jnp.float32),
                    "step_wall_seconds": jnp.asarray(time.monotonic() - step_start_time, dtype=jnp.float32),
                    "micro_steps": jnp.asarray(config.async_training.compute_burst_steps, dtype=jnp.float32),
                }
            )
        else:
            infos_by_stream[stream.name][-1] = {
                **infos_by_stream[stream.name][-1],
                "step_compute_seconds": jnp.asarray(compute_elapsed, dtype=jnp.float32),
                "next_batch_wait_seconds": jnp.asarray(batch_wait_elapsed, dtype=jnp.float32),
                "step_wall_seconds": jnp.asarray(time.monotonic() - step_start_time, dtype=jnp.float32),
            }

        if step % config.log_interval == 0:
            infos_by_stream, last_log_time = _log_periodic_metrics(
                config,
                step=step,
                streams=streams,
                streams_by_name=streams_by_name,
                infos_by_stream=infos_by_stream,
                last_log_time=last_log_time,
                gpu_monitor=gpu_monitor,
                wandb_logger=wandb_logger,
                pbar=pbar,
            )

        should_save_interval = step % config.save_interval == 0 and step > start_step
        should_publish_split_round = _should_publish_split_round_checkpoint(config, streams, step)
        should_save_after_view_update = False
        if (
            config.async_training.checkpoint_publisher.merge is None
            and config.async_training.save_steps_after_view_update is not None
        ):
            delay = max(0, int(config.async_training.save_steps_after_view_update))
            if all(stream.view_id is not None for stream in streams):
                view_key = tuple(sorted((stream.name, str(stream.view_id)) for stream in streams))
                if view_key not in saved_view_update_keys:
                    update_steps = [stream.last_view_update_step for stream in streams]
                    if all(update_step is not None for update_step in update_steps):
                        target_step = (
                            max(int(update_step) for update_step in update_steps if update_step is not None) + delay
                        )
                        view_update_targets.setdefault(view_key, target_step)
                        if step >= view_update_targets[view_key]:
                            should_save_after_view_update = True
                            saved_view_update_keys.add(view_key)
        should_save_final = config.async_training.save_final_checkpoint and step == config.num_train_steps - 1
        should_publish_checkpoint = (
            (config.async_training.publish_on_save_interval and should_save_interval)
            or should_save_after_view_update
            or should_publish_split_round
            or should_save_final
        )
        if should_save_interval or should_save_after_view_update or should_publish_split_round or should_save_final:
            checkpoint_params_override = training_utils.build_checkpoint_params_override(
                train_state, full_checkpoint_template
            )
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                primary_stream.data_loader,
                step,
                params_override=checkpoint_params_override,
            )
            source_checkpoint_dir = _checkpoint_step_dir(config.checkpoint_dir, step)
            dataset_view_id = (
                ",".join(f"{stream.name}:{stream.view_id}" for stream in streams if stream.view_id is not None) or None
            )
            dataset_view_ids = {stream.name: stream.view_id for stream in streams}
            if should_publish_checkpoint:
                target_policy_ids = {stream.target_policy_id for stream in streams}
                policy_id = checkpoint_publisher.resolve_policy_id(target_policy_ids)
                checkpoint_key = (int(policy_id), int(step))
                if checkpoint_key in scheduled_checkpoint_keys:
                    logging.info(
                        "Skipping duplicate checkpoint publication for policy_id=%d step=%d.",
                        policy_id,
                        step,
                    )
                    should_publish_checkpoint = False
                else:
                    scheduled_checkpoint_keys.add(checkpoint_key)
            if should_publish_checkpoint:
                if config.async_training.checkpoint_publisher.async_publish:
                    publish_jobs.append(
                        AsyncCheckpointPublishJob(
                            step=step,
                            checkpoint_manager=checkpoint_manager,
                            checkpoint_publisher=checkpoint_publisher,
                            source_checkpoint_dir=source_checkpoint_dir,
                            dataset_view_id=dataset_view_id,
                            dataset_view_ids=dataset_view_ids,
                            policy_id=policy_id,
                            base_checkpoint_id=active_base_checkpoint_id,
                        )
                    )
                else:
                    checkpoint_manager.wait_until_finished()
                    published_checkpoint = checkpoint_publisher.publish(
                        source_checkpoint_dir=source_checkpoint_dir,
                        step=step,
                        dataset_view_id=dataset_view_id,
                        dataset_view_ids=dataset_view_ids,
                        policy_id=policy_id,
                        base_checkpoint_id=active_base_checkpoint_id,
                    )
                    if published_checkpoint is not None:
                        wandb_logger.log_checkpoint_event(
                            step=step,
                            event="published",
                            checkpoint_ref=published_checkpoint,
                            policy_id=policy_id,
                        )
                    if (
                        published_checkpoint is not None
                        and config.async_training.checkpoint_publisher.reload_after_publish
                    ):
                        if config.async_training.checkpoint_publisher.merge is not None:
                            if step < config.num_train_steps - 1:
                                pending_next_policy_id = int(policy_id)
                                logging.info(
                                    "Published split checkpoint snapshot for policy_id=%d at step %d; trainer will "
                                    "pause training until matching dataset views arrive (keeps EMA == snapshot).",
                                    policy_id,
                                    step,
                                )
                            else:
                                _wait_for_final_split_merge(
                                    checkpoint_publisher=checkpoint_publisher,
                                    role_checkpoint=published_checkpoint,
                                    step=step,
                                    wandb_logger=wandb_logger,
                                )
                        else:
                            train_state = async_control.switch_train_state_to_current_ema(
                                config,
                                train_state,
                                checkpoint_id=str(published_checkpoint.get("checkpoint_id")),
                                reset_optimizer=config.async_training.checkpoint_switch.reset_optimizer,
                                reset_ema=config.async_training.checkpoint_switch.reset_ema,
                                preserve_global_step=config.async_training.checkpoint_switch.preserve_global_step,
                                state_sharding=train_state_sharding,
                            )
                            active_base_checkpoint_id = str(published_checkpoint.get("checkpoint_id"))
                            train_state = async_control.finalize_checkpoint_switch(
                                train_state,
                                context="current EMA rebase",
                                reclaim_memory=config.async_training.checkpoint_publisher.merge is not None,
                            )
            _validation.run_in_process_validation(
                config,
                step=step,
                train_state=train_state,
                validation_loaders=validation_loaders,
                mesh=mesh,
                data_sharding=data_sharding,
                active_video_threads=validation_video_threads,
            )
    _consume_finished_publish_jobs(wait=True)
    return train_state


def main(config: _config.TrainConfig):
    init_cli_logging()
    logging.info("Running async trainer on: %s", platform.node())

    # Boot the DataLoader forkserver now, before any JAX GPU op (jax.device_count below) initializes CUDA.
    # This lets dataset-view reload workers fork from a clean pre-CUDA server instead of spawning and
    # re-importing jax/torch/lerobot (~6x faster first batch). Falls back to spawn on failure.
    _data_loader.ensure_dataloader_forkserver()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    split_merge_enabled = config.async_training.checkpoint_publisher.merge is not None
    requested_resume = bool(config.resume)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    resume_published_checkpoint = (
        _latest_published_checkpoint(config)
        if requested_resume and config.async_training.checkpoint_publisher.reload_after_publish
        else None
    )
    lightweight_split_resume = split_merge_enabled and resume_published_checkpoint is not None
    effective_resuming = resuming or lightweight_split_resume
    if lightweight_split_resume:
        logging.info(
            "Using lightweight split resume from merged checkpoint %s; local role checkpoint is optional.",
            resume_published_checkpoint.get("checkpoint_id"),
        )
    training_utils.init_wandb_run(
        config,
        resuming=effective_resuming,
        enabled=config.wandb_enabled,
        log_code_path=None if split_merge_enabled else epath.Path(__file__).parent.parent,
        clear_proxy_env=True,
    )
    training_utils.log_wandb_startup(config)

    validation_loaders = None
    validation_video_threads: list[threading.Thread] = []
    if config.validation.enabled:
        validation_episode_refs = _validation.resolve_validation_episode_refs(config)
        validation_loaders = _validation.prepare_validation_loaders(config, validation_episode_refs)

    checkpoint_switch_watcher = async_control.CheckpointSwitchWatcher(config.async_training.checkpoint_switch)
    checkpoint_publisher = async_control.CheckpointPublisher(config.async_training.checkpoint_publisher)
    wandb_logger = WandbAsyncControlLogger(config)

    train_state, train_state_sharding = sync_train.init_train_state(
        config,
        init_rng,
        mesh,
        resume=resuming and not lightweight_split_resume,
    )
    jax.block_until_ready(train_state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))

    stream_configs = async_control.validate_stream_configs(config)
    initial_target_policy_id = None
    if split_merge_enabled:
        if resume_published_checkpoint is not None:
            initial_target_policy_id = int(resume_published_checkpoint["policy_id"])
        else:
            initial_target_policy_id = config.async_training.checkpoint_publisher.policy_id
    streams = [
        _create_stream(
            config,
            name=stream_config.name,
            data_factory=stream_config.data or config.data,
            dataset_view_config=stream_config.dataset_view,
            train_type_values=stream_config.train_type,
            train_type_by_target_policy_id=stream_config.train_type_by_target_policy_id,
            steps_per_cycle=stream_config.steps_per_cycle,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            train_state_sharding=train_state_sharding,
            wandb_logger=wandb_logger,
            initial_target_policy_id=initial_target_policy_id,
            defer_initial_device_prefetch=split_merge_enabled and effective_resuming,
        )
        for stream_config in stream_configs
    ]
    schedule = _stream_schedule(streams)
    primary_stream = streams[0]
    _log_startup_previews(config, primary_stream.batch)

    if effective_resuming:
        if lightweight_split_resume:
            local_checkpoint_step = _latest_local_checkpoint_step(config.checkpoint_dir)
            published_train_step = _published_checkpoint_train_step(resume_published_checkpoint)
            resume_source_step = local_checkpoint_step
            if published_train_step is not None:
                resume_source_step = (
                    max(resume_source_step, published_train_step)
                    if resume_source_step is not None
                    else published_train_step
                )
            if resume_source_step is None:
                raise FileNotFoundError(
                    "Cannot lightweight-resume split training because neither a local role checkpoint nor "
                    "published checkpoint train_step is available."
                )
            resume_step = resume_source_step + 1
            if local_checkpoint_step is None:
                logging.info(
                    "Lightweight split resume using published checkpoint train_step=%d to set TrainState.step=%d.",
                    resume_source_step,
                    resume_step,
                )
            else:
                logging.info(
                    "Lightweight split resume using checkpoint step %d to set TrainState.step=%d "
                    "(local=%d, published=%s).",
                    resume_source_step,
                    resume_step,
                    local_checkpoint_step,
                    "none" if published_train_step is None else str(published_train_step),
                )
            train_state = _set_train_state_step(train_state, resume_step)
        else:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, primary_stream.data_loader)
        if config.async_training.checkpoint_publisher.reload_after_publish:
            latest_published_checkpoint = resume_published_checkpoint
            if latest_published_checkpoint is not None:
                logging.info(
                    "Rebasing resumed train state to latest published checkpoint %s",
                    latest_published_checkpoint.get("checkpoint_id"),
                )
                train_state = async_control.switch_train_state_checkpoint(
                    config,
                    train_state,
                    latest_published_checkpoint,
                    reset_optimizer=config.async_training.checkpoint_switch.reset_optimizer,
                    reset_ema=config.async_training.checkpoint_switch.reset_ema,
                    preserve_global_step=config.async_training.checkpoint_switch.preserve_global_step,
                    state_sharding=train_state_sharding,
                )
                train_state = async_control.finalize_checkpoint_switch(
                    train_state,
                    context="resume latest published checkpoint rebase",
                    reclaim_memory=config.async_training.checkpoint_publisher.merge is not None,
                )

    full_checkpoint_template = training_utils.build_full_value_expert_checkpoint_template(config)

    start_step = int(train_state.step)
    if start_step > 0 and config.async_training.checkpoint_publisher.merge is not None:
        policy_boundary_step = max(0, start_step - 1)
        for stream in streams:
            if stream.target_policy_id is not None:
                stream.last_policy_update_step = policy_boundary_step
        logging.info(
            "Resumed split policy timing from train step %d with policy boundary step %d; active target policies "
            "will train for the configured delay before publication.",
            start_step,
            policy_boundary_step,
        )
    gpu_monitor = GpuUtilizationMonitor(config.async_training.gpu_monitor)
    gpu_monitor.start()
    try:
        train_state = _run_training_loop(
            config=config,
            checkpoint_manager=checkpoint_manager,
            checkpoint_publisher=checkpoint_publisher,
            checkpoint_switch_watcher=checkpoint_switch_watcher,
            train_state=train_state,
            train_state_sharding=train_state_sharding,
            full_checkpoint_template=full_checkpoint_template,
            train_rng=train_rng,
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            streams=streams,
            schedule=schedule,
            primary_stream=primary_stream,
            start_step=start_step,
            gpu_monitor=gpu_monitor,
            wandb_logger=wandb_logger,
            validation_loaders=validation_loaders,
            validation_video_threads=validation_video_threads,
        )
    finally:
        logging.info("Waiting for checkpoint manager to finish...")
        checkpoint_manager.wait_until_finished()
        if validation_video_threads:
            _validation.join_validation_video_threads(validation_video_threads)
        gpu_metrics = gpu_monitor.stop()
        if gpu_metrics:
            try:
                wandb.log(gpu_metrics, step=max(config.num_train_steps - 1, 0))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not log final GPU metrics to W&B: %s", exc)


if __name__ == "__main__":
    main(_config.cli())
