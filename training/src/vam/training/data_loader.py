from collections.abc import Iterator, Sequence
import copy
import dataclasses
import logging
import math
import multiprocessing
import os
import queue as queue_module
import random
import threading
import time
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow as pa
import torch

import vam.models.model as _model
import vam.training.config as _config
from vam.training.episode_replay import EpisodeKey
from vam.training.episode_replay import EpisodeReplayState
from vam.training.episode_replay import FrameRef
import vam.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


# DataLoader worker start method. Workers are forced off the default `fork` because the trainer's main
# process has JAX/CUDA initialized and forking that is unsafe. `spawn` is safe but makes every worker
# re-import jax/torch/lerobot from scratch (~85s to produce the first batch with 16 workers). `forkserver`
# is both safe and cheap: a server process is started once BEFORE CUDA init (see ensure_dataloader_forkserver),
# preloads the heavy modules, and workers fork from it without re-importing (~6x faster first batch). We only
# switch to forkserver once that pre-CUDA server has been booted; otherwise we keep spawn.
_DATALOADER_FORKSERVER_READY = False
_FORKSERVER_PRELOAD = (
    "numpy",
    "torch",
    "lerobot.common.datasets.lerobot_dataset",
    "vam.training.data_loader",
)


def _forkserver_noop() -> None:  # module-level so the forkserver can import it to boot the server
    pass


def ensure_dataloader_forkserver() -> None:
    """Boot the DataLoader forkserver BEFORE JAX/CUDA is initialized.

    Must be called at trainer startup before any JAX GPU op (e.g. jax.device_count()). Starting a trivial
    process forces Python's forkserver server to fork now, from a clean pre-CUDA main process, and to preload
    the heavy modules. Subsequent DataLoader workers then fork from that server cheaply instead of spawning
    and re-importing. On any failure we log and fall back to spawn (unchanged behavior).
    """
    global _DATALOADER_FORKSERVER_READY  # noqa: PLW0603
    try:
        multiprocessing.set_forkserver_preload(list(_FORKSERVER_PRELOAD))
        ctx = multiprocessing.get_context("forkserver")
        proc = ctx.Process(target=_forkserver_noop)
        proc.start()
        proc.join()
        _DATALOADER_FORKSERVER_READY = True
        logging.info("DataLoader forkserver started pre-CUDA; workers will fork from it (not spawn).")
    except Exception:
        logging.exception("Failed to start DataLoader forkserver; falling back to spawn workers.")
        _DATALOADER_FORKSERVER_READY = False


def _dataloader_mp_context(num_workers: int):
    """Multiprocessing context for DataLoader workers: forkserver if booted pre-CUDA, else spawn."""
    if num_workers <= 0:
        return None
    method = "forkserver" if _DATALOADER_FORKSERVER_READY else "spawn"
    return multiprocessing.get_context(method)


# Process-level cache of constructed per-repo LeRobotDataset objects, keyed by repo_id.
# Async streaming republishes a dataset view whenever the current-policy deployment repo grows by a
# few episodes, but the offline demo + historical repos in the view are immutable. Rebuilding every
# repo (and its O(num_frames) timestamp materialization) on each reload costs minutes; caching lets
# us reuse the unchanged repos and only reconstruct the one that actually grew. Only used on the
# async path (reuse_datasets=True); train.py / norm-stats build once and never touch this.
_LEROBOT_DATASET_CACHE: dict[str, tuple[typing.Any, "Dataset"]] = {}
# repo_ids that have been constructed (and timestamp-validated) at least once in this process.
_VALIDATED_REPO_IDS: set[str] = set()


def clear_lerobot_dataset_cache() -> None:
    """Drop all cached per-repo datasets (test hook; not used in the training loop)."""
    _LEROBOT_DATASET_CACHE.clear()
    _VALIDATED_REPO_IDS.clear()


def _hashable(value: typing.Any) -> typing.Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(val)) for key, val in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_hashable(item) for item in value)
    return value


def _dataset_signature(
    repo_id: str,
    episodes: Sequence[int] | None,
    delta_timestamps: dict[str, list[float]],
    defaults: dict[str, typing.Any],
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
) -> tuple:
    """Cache key capturing repo identity, how it is windowed, and its on-disk content.

    The content component (total_frames/total_episodes + episodes.jsonl mtime) changes whenever the
    converter appends episodes, so a grown repo is a cache miss and gets rebuilt; unchanged repos hit.
    Direct-upload p70 relabeling does not change episode counts, so also track its annotation sidecar.
    """
    episodes_key = tuple(episodes) if episodes is not None else None
    delta_key = tuple(sorted((key, tuple(vals)) for key, vals in delta_timestamps.items()))
    defaults_key = _hashable(defaults)
    try:
        episodes_mtime = (dataset_meta.root / "meta" / "episodes.jsonl").stat().st_mtime_ns
    except OSError:
        episodes_mtime = None
    try:
        annotation_mtime = (dataset_meta.root / "meta" / "advantage_annotation.json").stat().st_mtime_ns
    except FileNotFoundError:
        annotation_mtime = None
    return (
        repo_id,
        episodes_key,
        delta_key,
        defaults_key,
        int(dataset_meta.total_frames),
        int(dataset_meta.total_episodes),
        episodes_mtime,
        annotation_mtime,
    )


def _build_lerobot_dataset(
    repo_id: str,
    *,
    delta_timestamps: dict[str, list[float]],
    episodes: Sequence[int] | None,
    validate_timestamps: bool,
) -> lerobot_dataset.LeRobotDataset:
    """Construct a LeRobotDataset, optionally skipping the O(num_frames) timestamp-sync check.

    ``validate_timestamps=True`` calls the vendored constructor unchanged (full validation).
    ``validate_timestamps=False`` faithfully replicates the vendored ``__init__`` attribute setup but
    skips the ``torch.stack(hf_dataset["timestamp"]) + check_timestamps_sync`` block, which forces the
    whole memory-mapped Arrow column to be materialized into torch tensors on every construction. We
    only skip it when rebuilding a repo that was already validated once in this process (i.e. it grew),
    so trusted async data is not re-validated on every reload while first-time loads still are.
    """
    if validate_timestamps:
        return lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=episodes)

    dataset = lerobot_dataset.LeRobotDataset.__new__(lerobot_dataset.LeRobotDataset)
    # Mirror the vendored LeRobotDataset.__init__ attribute assignments (see the pinned
    # lerobot/common/datasets/lerobot_dataset.py) minus the timestamp-sync block.
    torch.utils.data.Dataset.__init__(dataset)
    dataset.repo_id = repo_id
    dataset.root = lerobot_dataset.HF_LEROBOT_HOME / repo_id
    dataset.image_transforms = None
    dataset.delta_timestamps = delta_timestamps
    dataset.episodes = list(episodes) if episodes is not None else None
    dataset.tolerance_s = 1e-4
    dataset.revision = lerobot_dataset.CODEBASE_VERSION
    dataset.video_backend = lerobot_dataset.get_safe_default_codec()
    dataset.delta_indices = None
    dataset.image_writer = None
    dataset.episode_buffer = None

    dataset.root.mkdir(exist_ok=True, parents=True)
    dataset.meta = lerobot_dataset.LeRobotDatasetMetadata(dataset.repo_id, dataset.root, dataset.revision)
    if dataset.episodes is not None and dataset.meta._version >= lerobot_dataset.packaging.version.parse("v2.1"):  # noqa: SLF001
        episodes_stats = [dataset.meta.episodes_stats[ep_idx] for ep_idx in dataset.episodes]
        dataset.stats = lerobot_dataset.aggregate_stats(episodes_stats)

    try:
        assert all((dataset.root / fpath).is_file() for fpath in dataset.get_episodes_file_paths())
        dataset.hf_dataset = dataset.load_hf_dataset()
    except (AssertionError, FileNotFoundError, NotADirectoryError):
        dataset.revision = lerobot_dataset.get_safe_version(dataset.repo_id, dataset.revision)
        dataset.download_episodes(True)  # noqa: FBT003
        dataset.hf_dataset = dataset.load_hf_dataset()

    dataset.episode_data_index = lerobot_dataset.get_episode_data_index(dataset.meta.episodes, dataset.episodes)

    # Intentionally skipped: torch.stack(hf_dataset["timestamp"]) + check_timestamps_sync (O(num_frames)).
    if dataset.delta_timestamps is not None:
        lerobot_dataset.check_delta_timestamps(dataset.delta_timestamps, dataset.fps, dataset.tolerance_s)
        dataset.delta_indices = lerobot_dataset.get_delta_indices(dataset.delta_timestamps, dataset.fps)
    return dataset


def _apply_advantage_snapshot(dataset, snapshot: str | None) -> None:
    """Install precomputed labels for exactly this view, even if disk parquets were relabeled later."""
    if snapshot is None:
        return
    import datasets as hf_datasets

    path = dataset.root / snapshot
    if not path.resolve().is_relative_to(dataset.root.resolve()):
        raise ValueError(f"Advantage snapshot escapes repo: {snapshot}")
    episodes = dataset.episodes if dataset.episodes is not None else sorted(dataset.meta.episodes)
    with np.load(path, allow_pickle=False) as archive:
        labels = []
        for ep in episodes:
            values = np.asarray(archive[str(ep)], dtype=np.float32)
            if values.shape != (dataset.meta.episodes[ep]["length"],) or not np.isin(values, [0, 1]).all():
                raise ValueError(f"Invalid advantage snapshot for episode {ep}: {path}")
            labels.extend(values[:, None].tolist())
    columns = dataset.hf_dataset.column_names
    table = dataset.hf_dataset.with_format(None)
    if "advantage" in columns:
        table = table.remove_columns("advantage")
    else:
        columns = [*columns, "advantage"]
    label_table = hf_datasets.Dataset.from_dict(
        {"advantage": labels},
        features=hf_datasets.Features({"advantage": hf_datasets.Sequence(hf_datasets.Value("float32"))}),
    )
    # Horizontal Arrow concatenation preserves mmap blocks for all large action/state columns.
    table = hf_datasets.concatenate_datasets([table, label_table], axis=1).select_columns(columns)
    table.set_transform(lerobot_dataset.hf_transform_to_torch)
    dataset.hf_dataset = table


class LiveCurrentRepoDataset:
    """A single LeRobot repo that grows in place as episodes are appended — no dataloader rebuild.

    VAM's Online DAgger/Learn from Experience loop (HybridDataset.append_episodes) keeps one dataset
    alive and refreshes episode_data_index / delta_indices as new data arrives. We do the same for the current-policy
    deployment repo (the only thing that grows within a policy generation). Because the converter writes one
    parquet per episode, growing the repo means reading the newly-appeared parquet files and concatenating
    them into the in-memory hf_dataset — the offline demo + history repos are NOT touched (they live in the
    separate stable multiprocess loader). This dataset is read on the MAIN thread (no torch workers), so an
    in-place grow is immediately visible to the next batch; the current repo is small (a few dozen episodes
    per generation) so single-thread decode of a batch is a few seconds.

    Reuses LeRobotDataset.__getitem__ unchanged (delta-timestamp windowing + video decode + transforms).
    """

    def __init__(
        self,
        repo_id: str,
        *,
        delta_timestamps: dict[str, list[float]],
        missing_feature_defaults: dict[str, typing.Any] | None = None,
        episode_ids: Sequence[int] | None = None,
        advantage_snapshot: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self._delta_timestamps = delta_timestamps
        self._defaults = dict(missing_feature_defaults or {})
        # Pin the initial build to the authoritative meta.episodes list rather than letting lerobot glob the
        # whole data/ dir (episodes=None). The converter writes an episode's parquet BEFORE appending it to
        # meta/episodes.jsonl, so a directory glob can catch a not-yet-registered parquet whose frames carry an
        # episode_index beyond episode_data_index's range -> "index N out of bounds" mid-decode. Loading exactly
        # the files for meta.episodes keeps hf_dataset and episode_data_index built from the same snapshot.
        initial_episodes = (
            list(episode_ids)
            if episode_ids is not None
            else sorted(int(ep) for ep in lerobot_dataset.LeRobotDatasetMetadata(repo_id).episodes)
        )
        # Skip the O(num_frames) check_timestamps_sync, exactly like create_torch_dataset does for these
        # async converter repos: their legitimate per-episode timestamp reset (last frame of ep N ~= 33s,
        # first frame of ep N+1 == 0.0) trips the vendored check even though the data is well-formed.
        self._dataset = _build_lerobot_dataset(
            repo_id, delta_timestamps=delta_timestamps, episodes=initial_episodes, validate_timestamps=False
        )
        _add_missing_hf_dataset_columns(self._dataset, self._defaults)
        _apply_advantage_snapshot(self._dataset, advantage_snapshot)
        self._advantage_snapshot = advantage_snapshot
        self.revision = 0
        # Track exactly the episodes materialized into hf_dataset above (NOT a re-read of meta, which may have
        # grown in the meantime) so poll_and_grow won't mistake a not-yet-loaded episode for an already-loaded
        # one and silently skip it.
        self._loaded_episode_ids = list(initial_episodes)
        # poll_and_grow (main thread) publishes a fresh dataset object; prefetch threads read self._dataset.
        # A single __getitem__/__len__ reads the ref once (atomic in CPython), so it always sees a consistent
        # (hf_dataset, episode_data_index) pair. _grow_lock only serializes concurrent grow calls.
        self._grow_lock = threading.Lock()

    # --- LeRobot passthrough so the sampler / __getitem__ path sees a normal dataset ---
    # Each reads self._dataset into a local first, so a concurrent poll_and_grow swap can't tear a read.
    @property
    def hf_dataset(self):
        return self._dataset.hf_dataset

    @property
    def meta(self):
        return self._dataset.meta

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex):
        return self._dataset[index]

    def _current_episode_ids(self) -> list[int]:
        return sorted(int(ep) for ep in self._dataset.meta.episodes)

    def poll_and_grow(self, *, episode_ids: Sequence[int] | None = None, advantage_snapshot: str | None = None) -> int:
        """Load only published new episodes; atomically replace labels from an immutable revision."""
        import datasets as hf_datasets

        with self._grow_lock:
            current = self._dataset
            meta = lerobot_dataset.LeRobotDatasetMetadata(self.repo_id)
            current_ids = list(episode_ids) if episode_ids is not None else sorted(meta.episodes)
            if current_ids[: len(self._loaded_episode_ids)] != self._loaded_episode_ids:
                raise ValueError("Live repo updates must preserve the existing episode prefix")
            new_ids = current_ids[len(self._loaded_episode_ids) :]
            labels_changed = advantage_snapshot != self._advantage_snapshot
            if not new_ids and not labels_changed:
                return 0
            grown = current.hf_dataset
            if new_ids:
                new_files = [str(current.root / meta.get_data_file_path(ep)) for ep in new_ids]
                new_hf = hf_datasets.load_dataset("parquet", data_files=new_files, split="train")
                for key, value in self._defaults.items():
                    if key in grown.column_names and key not in new_hf.column_names:
                        new_hf = new_hf.add_column(key, _default_hf_column(len(new_hf), value))
                grown = hf_datasets.concatenate_datasets([grown, new_hf])
            new_dataset = copy.copy(current)
            new_dataset.meta = meta
            new_dataset.hf_dataset = grown
            new_dataset.episodes = current_ids
            new_dataset.episode_data_index = lerobot_dataset.get_episode_data_index(meta.episodes, current_ids)
            _apply_advantage_snapshot(new_dataset, advantage_snapshot)
            new_dataset.hf_dataset.set_transform(lerobot_dataset.hf_transform_to_torch)
            self._dataset = new_dataset
            self._loaded_episode_ids = current_ids
            self._advantage_snapshot = advantage_snapshot
            self.revision += 1
            logging.info(
                "Live repo %s updated: +%d episodes, labels_changed=%s (no worker rebuild).",
                self.repo_id,
                len(new_ids),
                labels_changed,
            )
            return len(new_ids)


@dataclasses.dataclass(frozen=True)
class DatasetSelection:
    dataloader: torch.utils.data.DataLoader
    raw_dataset: lerobot_dataset.LeRobotDataset
    total_frames: int
    start_frame: int
    selected_episode_ids: list[int] | None


def _default_hf_column(length: int, value: typing.Any):
    scalar = value
    if isinstance(value, list | tuple | np.ndarray):
        array = np.asarray(value)
        if array.ndim == 0:
            scalar = array.item()
        else:
            return [array.astype(np.float32).tolist() for _ in range(length)]
    if isinstance(scalar, bool):
        return [scalar for _ in range(length)]
    if isinstance(scalar, int):
        return [int(scalar) for _ in range(length)]
    if isinstance(scalar, float):
        return [[float(scalar)] for _ in range(length)]
    return [scalar for _ in range(length)]


def _add_missing_hf_dataset_columns(dataset: lerobot_dataset.LeRobotDataset, defaults: dict[str, typing.Any]) -> None:
    if not defaults or not hasattr(dataset, "hf_dataset"):
        return
    missing = {key: value for key, value in defaults.items() if key not in dataset.hf_dataset.column_names}
    if not missing:
        return
    for key, value in missing.items():
        dataset.hf_dataset = dataset.hf_dataset.add_column(key, _default_hf_column(len(dataset.hf_dataset), value))
        feature_spec = {"dtype": "float32", "shape": [1], "names": [key]} if isinstance(value, float) else None
        if feature_spec is not None:
            dataset.meta.info.setdefault("features", {}).setdefault(key, feature_spec)
    logging.info(
        "Added in-memory default columns for %s: %s",
        getattr(dataset, "repo_id", dataset),
        sorted(missing),
    )


class BalancedDataSourceSampler(torch.utils.data.Sampler):
    """
    Sampler that provides balanced sampling from configured data_source groups.

    Each batch will contain samples from each configured data_source value in a 1:1:...
    ratio when the batch size is divisible by the number of groups. Otherwise, the
    remainder is rotated across groups so sampling stays balanced over consecutive batches.

    Args:
        dataset: The dataset to sample from
        batch_size: Total batch size (will be split across the configured groups)
        data_source_values: data_source values to balance, e.g. (-1.0, 0.0) or (-1.0, 0.0, 1.0). If None, online
            balancing keys are applied to all data instead of only the data_source=1.0 quota.
        balance_online_success_failure: If true, split the data_source=1.0 quota 1:1 between success=1 and success=0.
        balance_online_termination: If true, split the data_source=1.0 quota 1:1 between terminate=1 and terminate=0.
            If both online balancing flags are true, split online samples across success/terminate combinations in a
            1:1:1:1 ratio.
        shuffle: Whether to shuffle indices within each category
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        data_source_values: Sequence[float] | None,
        balance_online_success_failure: bool = False,  # noqa: FBT001, FBT002
        balance_online_termination: bool = False,  # noqa: FBT001, FBT002
        shuffle: bool = True,  # noqa: FBT001, FBT002
        seed: int = 0,
    ):
        if data_source_values is None and not (balance_online_success_failure or balance_online_termination):
            raise ValueError("BalancedDataSourceSampler requires data_source values or online balancing.")
        if (
            data_source_values is not None
            and len(data_source_values) < 2
            and not (balance_online_success_failure or balance_online_termination)
        ):
            raise ValueError(
                "BalancedDataSourceSampler requires at least two data_source values unless online balancing is enabled."
            )
        if data_source_values is not None and len(set(data_source_values)) != len(data_source_values):
            raise ValueError(f"BalancedDataSourceSampler got duplicate data_source values: {data_source_values}.")
        if data_source_values is not None and batch_size < len(data_source_values):
            raise ValueError(
                "BalancedDataSourceSampler requires batch_size >= number of data_source groups "
                f"({len(data_source_values)}), got {batch_size}."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.balance_all_data = data_source_values is None
        self.data_source_values = (
            () if data_source_values is None else tuple(float(value) for value in data_source_values)
        )
        self.balance_online_success_failure = balance_online_success_failure
        self.balance_online_termination = balance_online_termination
        self.online_data_source_value = 1.0
        self.online_balance_keys = (("success",) if balance_online_success_failure else ()) + (
            ("terminate",) if balance_online_termination else ()
        )
        self.online_balance_values = self._online_balance_values()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Build index lists by scanning data_source lazily on first iteration to avoid slow initialization.
        self._group_indices = None
        self._online_balance_indices = None
        self._indices_built = False
        self._unmatched_data_sources = set()
        self._unmatched_online_balance_values = set()

    @staticmethod
    def _extract_scalar(value) -> float:
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, list | np.ndarray):
            array = np.asarray(value)
            return float(array.flatten()[0]) if array.size > 0 else 0.0
        return float(value)

    @classmethod
    def _extract_data_source(cls, data_source) -> float:
        return cls._extract_scalar(data_source)

    def _online_balance_values(self) -> tuple[float | tuple[float, ...], ...]:
        if self.balance_online_success_failure and self.balance_online_termination:
            return ((1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0))
        if self.balance_online_success_failure or self.balance_online_termination:
            return (1.0, 0.0)
        return ()

    def _format_online_balance_key(self, value: float | tuple[float, ...]) -> str:
        if len(self.online_balance_keys) == 1:
            return f"{self.online_balance_keys[0]}={value}"
        return ", ".join(
            f"{key}={field_value}" for key, field_value in zip(self.online_balance_keys, value, strict=True)
        )

    def _should_balance_online_binary_field(self) -> bool:
        return bool(self.online_balance_keys)

    def _append_index(
        self,
        idx: int,
        data_source: float,
        online_balance: float | tuple[float, ...] | None = None,
    ) -> None:
        if self.balance_all_data:
            self._append_balance_index(idx, online_balance)
            return
        for configured_value in self.data_source_values:
            if np.isclose(data_source, configured_value):
                self._group_indices[configured_value].append(idx)
                if self._should_balance_online_binary_field() and np.isclose(
                    configured_value, self.online_data_source_value
                ):
                    self._append_balance_index(idx, online_balance)
                return
        self._unmatched_data_sources.add(data_source)

    def _append_balance_index(self, idx: int, online_balance: float | tuple[float, ...] | None) -> None:
        if online_balance is None:
            self._unmatched_online_balance_values.add(None)
            return
        for balance_value in self.online_balance_values:
            if np.allclose(online_balance, balance_value):
                self._online_balance_indices[balance_value].append(idx)
                return
        self._unmatched_online_balance_values.add(online_balance)

    def _build_indices(self):
        """Build configured data_source index lists by scanning the dataset."""
        if self._indices_built:
            return

        logging.info("Building balanced sampler indices for data_source values %s...", self.data_source_values)
        self._group_indices = {value: [] for value in self.data_source_values}
        self._online_balance_indices = {value: [] for value in self.online_balance_values}

        # Access the underlying dataset if wrapped
        base_dataset = self.dataset
        while hasattr(base_dataset, "_dataset"):
            base_dataset = base_dataset._dataset  # noqa: SLF001

        # Try to read data_source from the dataset's hf_dataset (parquet) directly.
        if hasattr(base_dataset, "hf_dataset") and (
            self.balance_all_data or "data_source" in base_dataset.hf_dataset.column_names
        ):
            logging.info("Using fast parquet-based balanced sampler index building...")
            data_sources = (
                [0.0] * len(base_dataset.hf_dataset)
                if self.balance_all_data
                else base_dataset.hf_dataset["data_source"]
            )
            online_balance_columns = (
                [base_dataset.hf_dataset[key] for key in self.online_balance_keys]
                if self._should_balance_online_binary_field()
                and all(key in base_dataset.hf_dataset.column_names for key in self.online_balance_keys)
                else None
            )
            for idx, data_source in enumerate(data_sources):
                parsed_online_balance = None
                if online_balance_columns is not None:
                    parsed_values = tuple(self._extract_scalar(column[idx]) for column in online_balance_columns)
                    parsed_online_balance = parsed_values[0] if len(parsed_values) == 1 else parsed_values
                self._append_index(idx, self._extract_data_source(data_source), parsed_online_balance)
        else:
            # Fallback to slow method if hf_dataset not available.
            logging.info("Using slow sample-based balanced sampler index building (hf_dataset not available)...")
            for idx in range(len(base_dataset)):
                try:
                    sample = base_dataset[idx]
                    online_balance = None
                    if self._should_balance_online_binary_field() and all(
                        key in sample for key in self.online_balance_keys
                    ):
                        parsed_values = tuple(self._extract_scalar(sample[key]) for key in self.online_balance_keys)
                        online_balance = parsed_values[0] if len(parsed_values) == 1 else parsed_values
                    self._append_index(
                        idx,
                        0.0
                        if self.balance_all_data
                        else self._extract_data_source(sample.get("data_source", np.array([0.0]))),
                        online_balance,
                    )
                except Exception as e:
                    logging.warning(f"Error reading sample {idx}: {e}")
                    # Default to the first configured group if we can't read.
                    if self.balance_all_data:
                        self._unmatched_online_balance_values.add(None)
                    else:
                        self._group_indices[self.data_source_values[0]].append(idx)

        self._indices_built = True
        if not self.balance_all_data:
            group_counts = ", ".join(
                f"data_source={value}: {len(indices)}" for value, indices in self._group_indices.items()
            )
            logging.info("Balanced data_source sampler indices built: %s", group_counts)
        if self._should_balance_online_binary_field():
            online_balance_counts = ", ".join(
                f"{self._format_online_balance_key(value)}: {len(indices)}"
                for value, indices in self._online_balance_indices.items()
            )
            logging.info(
                "Balanced %s sampler indices built for %s: %s",
                "/".join(self.online_balance_keys),
                "all data" if self.balance_all_data else f"data_source={self.online_data_source_value}",
                online_balance_counts,
            )
        if self._unmatched_data_sources:
            logging.warning(
                "Balanced data_source sampler ignored values not listed in data_source_values: %s",
                sorted(self._unmatched_data_sources),
            )
        if self._unmatched_online_balance_values:
            logging.warning(
                "Balanced online %s sampler ignored values outside %s: %s",
                "/".join(self.online_balance_keys),
                self.online_balance_values,
                sorted(self._unmatched_online_balance_values, key=lambda value: str(value)),
            )

    def _batch_counts(self, batch_index: int) -> list[int]:
        num_groups = len(self.data_source_values)
        base_count = self.batch_size // num_groups
        remainder = self.batch_size % num_groups
        counts = [base_count] * num_groups
        for offset in range(remainder):
            counts[(batch_index + offset) % num_groups] += 1
        return counts

    def _num_complete_batches(self) -> int:
        if self.balance_all_data:
            max_samples = max(len(indices) for indices in self._online_balance_indices.values())
            max_balance_quota = max(1, math.ceil(self.batch_size / len(self.online_balance_values)))
            return max(1, math.ceil(max_samples / max_balance_quota))

        num_groups = len(self.data_source_values)
        base_count = self.batch_size // num_groups
        max_samples = max(len(indices) for indices in self._group_indices.values())
        num_batches = max(1, math.ceil(max_samples / base_count))
        if self._should_balance_online_binary():
            online_group_index = self.data_source_values.index(self.online_data_source_value)
            max_online_count = max(
                self._batch_counts(batch_index)[online_group_index] for batch_index in range(num_groups)
            )
            max_balance_quota = max(1, math.ceil(max_online_count / len(self.online_balance_values)))
            max_online_balance_samples = max(len(indices) for indices in self._online_balance_indices.values())
            num_batches = max(num_batches, math.ceil(max_online_balance_samples / max_balance_quota))
        if self.batch_size % num_groups:
            # Rotating the remainder is exactly balanced every `num_groups` batches.
            num_batches = math.ceil(num_batches / num_groups) * num_groups
        return num_batches

    @staticmethod
    def _extend_to_length(indices: list[int], target_len: int) -> list[int]:
        if len(indices) >= target_len:
            return indices[:target_len]

        repeated = indices.copy()
        while len(repeated) < target_len:
            repeated.extend(indices.copy())
        return repeated[:target_len]

    def _should_balance_online_binary(self) -> bool:
        if not self._should_balance_online_binary_field():
            return False
        if self.balance_all_data:
            if any(len(self._online_balance_indices[value]) == 0 for value in self.online_balance_values):
                logging.warning(
                    "Balanced %s sampling requested for all data, but at least one group is empty. "
                    "Falling back to all available indices.",
                    "/".join(self.online_balance_keys),
                )
                return False
            return True
        if self.online_data_source_value not in self.data_source_values:
            logging.warning(
                "Balanced online %s sampling requested, but data_source=%s is not configured in %s. "
                "Falling back to data_source-only balancing.",
                "/".join(self.online_balance_keys),
                self.online_data_source_value,
                self.data_source_values,
            )
            return False
        if any(len(self._online_balance_indices[value]) == 0 for value in self.online_balance_values):
            logging.warning(
                "Balanced online %s sampling requested, but at least one online group is empty. "
                "Falling back to data_source-only balancing.",
                "/".join(self.online_balance_keys),
            )
            return False
        return True

    def _online_balance_counts(self, online_count: int, batch_index: int) -> list[int]:
        base_count = online_count // len(self.online_balance_values)
        remainder = online_count % len(self.online_balance_values)
        counts = [base_count] * len(self.online_balance_values)
        for offset in range(remainder):
            counts[(batch_index + offset) % len(self.online_balance_values)] += 1
        return counts

    def __iter__(self):
        self._build_indices()

        balance_online_binary = self._should_balance_online_binary()
        if self.balance_all_data:
            if not balance_online_binary:
                return
            yield from self._iter_balance_groups(random.Random(self.seed + self.epoch))
            self.epoch += 1
            return

        groups = [self._group_indices[value].copy() for value in self.data_source_values]
        online_balance_groups = [self._online_balance_indices[value].copy() for value in self.online_balance_values]

        # Set random seed based on epoch for reproducibility.
        rng = random.Random(self.seed + self.epoch)

        if self.shuffle:
            for group in groups:
                rng.shuffle(group)
            for group in online_balance_groups:
                rng.shuffle(group)

        if any(len(group) == 0 for group in groups):
            logging.warning(
                "Balanced data_source sampling requested, but at least one configured group is empty. "
                "Falling back to all available indices."
            )
            all_indices = [idx for group in groups for idx in group]
            if self.shuffle:
                rng.shuffle(all_indices)
            yield from all_indices
            return

        num_complete_batches = self._num_complete_batches()
        base_count = self.batch_size // len(groups)
        max_group_quota = base_count + int(self.batch_size % len(groups) > 0)
        target_group_len = num_complete_batches * max_group_quota
        groups = [self._extend_to_length(group, target_group_len) for group in groups]
        cursors = [0] * len(groups)
        if balance_online_binary:
            online_group_index = self.data_source_values.index(self.online_data_source_value)
            max_online_quota = max(
                self._batch_counts(batch_index)[online_group_index] for batch_index in range(len(groups))
            )
            max_balance_quota = max(1, math.ceil(max_online_quota / len(self.online_balance_values)))
            target_balance_len = num_complete_batches * max_balance_quota
            online_balance_groups = [
                self._extend_to_length(group, target_balance_len) for group in online_balance_groups
            ]
            online_balance_cursors = [0] * len(online_balance_groups)

        for batch_index in range(num_complete_batches):
            counts = self._batch_counts(batch_index)
            batch_indices = []
            for group_index, count in enumerate(counts):
                if balance_online_binary and group_index == online_group_index:
                    balance_counts = self._online_balance_counts(count, batch_index)
                    for balance_group_index, balance_count in enumerate(balance_counts):
                        start = online_balance_cursors[balance_group_index]
                        end = start + balance_count
                        batch_indices.extend(online_balance_groups[balance_group_index][start:end])
                        online_balance_cursors[balance_group_index] = end
                    continue
                start = cursors[group_index]
                end = start + count
                batch_indices.extend(groups[group_index][start:end])
                cursors[group_index] = end

            # Shuffle within batch for additional randomness.
            rng.shuffle(batch_indices)

            yield from batch_indices

        self.epoch += 1

    def _iter_balance_groups(self, rng: random.Random):
        online_balance_groups = [self._online_balance_indices[value].copy() for value in self.online_balance_values]
        if self.shuffle:
            for group in online_balance_groups:
                rng.shuffle(group)

        if any(len(group) == 0 for group in online_balance_groups):
            logging.warning(
                "Balanced %s sampling requested for all data, but at least one group is empty. "
                "Falling back to all matched indices.",
                "/".join(self.online_balance_keys),
            )
            all_indices = [idx for group in online_balance_groups for idx in group]
            if self.shuffle:
                rng.shuffle(all_indices)
            yield from all_indices
            return

        num_complete_batches = self._num_complete_batches()
        max_balance_quota = max(1, math.ceil(self.batch_size / len(self.online_balance_values)))
        target_balance_len = num_complete_batches * max_balance_quota
        online_balance_groups = [self._extend_to_length(group, target_balance_len) for group in online_balance_groups]
        online_balance_cursors = [0] * len(online_balance_groups)

        for batch_index in range(num_complete_batches):
            balance_counts = self._online_balance_counts(self.batch_size, batch_index)
            batch_indices = []
            for balance_group_index, balance_count in enumerate(balance_counts):
                start = online_balance_cursors[balance_group_index]
                end = start + balance_count
                batch_indices.extend(online_balance_groups[balance_group_index][start:end])
                online_balance_cursors[balance_group_index] = end
            rng.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        self._build_indices()
        if self.balance_all_data:
            if any(len(indices) == 0 for indices in self._online_balance_indices.values()):
                return sum(len(indices) for indices in self._online_balance_indices.values())
            return self._num_complete_batches() * self.batch_size
        groups = list(self._group_indices.values())
        if any(len(group) == 0 for group in groups):
            return sum(len(group) for group in groups)

        return self._num_complete_batches() * self.batch_size

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch


class WeightedColumnGroupSampler(torch.utils.data.Sampler):
    """Weighted batch sampler over simple column predicates.

    This sampler is intentionally storage-agnostic: it unwraps transformed/concat datasets and reads columns from
    the underlying LeRobot hf_dataset when available, so sampler-only metadata does not need to pass through repack
    transforms.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        sampler_config: _config.SamplerConfig,
        *,
        shuffle: bool = True,
        seed: int = 0,
        episode_replay_state: EpisodeReplayState | None = None,
        episode_replay_scope: str = "default",
    ):
        if sampler_config.type != "weighted_column_groups":
            raise ValueError(f"Unsupported sampler type: {sampler_config.type!r}")
        if not sampler_config.groups:
            raise ValueError("WeightedColumnGroupSampler requires at least one sampler group.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = sampler_config
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.episode_replay_state = episode_replay_state or EpisodeReplayState(seed=seed)
        self.episode_replay_scope = episode_replay_scope
        self._group_indices: dict[str, list[int]] | None = None
        self._group_episode_candidates: dict[str, dict[EpisodeKey, list[FrameRef]]] | None = None
        self._active_groups: tuple[_config.SamplerGroupConfig, ...] | None = None

    @staticmethod
    def _extract_scalar(value) -> float | int | str:
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, list | np.ndarray):
            array = np.asarray(value)
            value = array.flatten()[0] if array.size > 0 else 0
        if isinstance(value, np.generic):
            value = value.item()
        return value

    @staticmethod
    def _coerce_expected(value, current_policy_id: int | None):
        if value == "$current_policy_id":
            if current_policy_id is None:
                raise ValueError("Sampler predicate uses $current_policy_id but current_policy_id is not set.")
            return current_policy_id
        if isinstance(value, list | tuple):
            return tuple(WeightedColumnGroupSampler._coerce_expected(v, current_policy_id) for v in value)
        return value

    @classmethod
    def _matches_predicate(cls, actual, predicate, current_policy_id: int | None) -> bool:
        actual = cls._extract_scalar(actual)
        if isinstance(predicate, dict):
            for op, raw_expected in predicate.items():
                if op not in {"eq", "==", "ne", "!=", "lt", "lte", "gt", "gte", "in"}:
                    raise ValueError(f"Unsupported sampler predicate operator: {op!r}")
                expected_value = cls._coerce_expected(raw_expected, current_policy_id)
                if op in {"eq", "=="} and actual != expected_value:
                    return False
                if op in {"ne", "!="} and actual == expected_value:
                    return False
                if op == "lt" and not actual < expected_value:
                    return False
                if op == "lte" and not actual <= expected_value:
                    return False
                if op == "gt" and not actual > expected_value:
                    return False
                if op == "gte" and not actual >= expected_value:
                    return False
                if op == "in" and actual not in expected_value:
                    return False
            return True

        expected = cls._coerce_expected(predicate, current_policy_id)
        if isinstance(expected, tuple):
            return actual in expected
        if isinstance(actual, float) or isinstance(expected, float):
            return bool(np.isclose(float(actual), float(expected)))
        return actual == expected

    @staticmethod
    def _column_as_numpy(hf_dataset, column: str) -> np.ndarray:
        """Read a whole hf_dataset column as a 1-D numpy array, bypassing the per-row torch transform.

        hf_dataset has a torch set_transform, so ``hf_dataset[column]`` materializes a list of tensors
        (O(num_frames) Python). We read the underlying Arrow table directly and normalize length-1 list
        columns (e.g. shape [1] scalars) down to 1-D.
        """
        chunked = hf_dataset.data.column(column)
        try:
            values = np.asarray(chunked.combine_chunks().to_numpy(zero_copy_only=False))
        except (ValueError, TypeError, pa.ArrowInvalid):  # pragma: no cover - exotic arrow layouts
            values = np.asarray(chunked.to_pylist())
        if values.dtype == object:
            values = np.asarray([np.asarray(item).reshape(-1)[0] for item in values])
        elif values.ndim >= 2:
            values = values[:, 0]
        return values

    @classmethod
    def _vectorized_predicate_mask(cls, values: np.ndarray, predicate, current_policy_id: int | None) -> np.ndarray:
        """Vectorized equivalent of _matches_predicate over a whole column."""
        values = np.asarray(values)
        is_float_col = np.issubdtype(values.dtype, np.floating)

        def eq_mask(expected) -> np.ndarray:
            if isinstance(expected, float) or is_float_col:
                return np.isclose(values.astype(np.float64), float(expected))
            return values == expected

        if isinstance(predicate, dict):
            mask = np.ones(values.shape[0], dtype=bool)
            for op, raw_expected in predicate.items():
                if op not in {"eq", "==", "ne", "!=", "lt", "lte", "gt", "gte", "in"}:
                    raise ValueError(f"Unsupported sampler predicate operator: {op!r}")
                expected = cls._coerce_expected(raw_expected, current_policy_id)
                if op in {"eq", "=="}:
                    mask &= eq_mask(expected)
                elif op in {"ne", "!="}:
                    mask &= ~eq_mask(expected)
                elif op == "lt":
                    mask &= values < expected
                elif op == "lte":
                    mask &= values <= expected
                elif op == "gt":
                    mask &= values > expected
                elif op == "gte":
                    mask &= values >= expected
                elif op == "in":
                    mask &= np.isin(values, np.asarray(expected))
            return mask

        expected = cls._coerce_expected(predicate, current_policy_id)
        if isinstance(expected, tuple):
            return np.isin(values, np.asarray(expected))
        return eq_mask(expected)

    @classmethod
    def _vectorized_group_mask(
        cls, columns: dict, where: dict, current_policy_id: int | None, num_rows: int
    ) -> np.ndarray:
        mask = np.ones(num_rows, dtype=bool)
        for column, predicate in where.items():
            mask &= cls._vectorized_predicate_mask(columns[column], predicate, current_policy_id)
        return mask

    @staticmethod
    def _split_operator_key(key: str) -> tuple[str, str | None]:
        for suffix, op in (
            ("_lt_current", "lt_current"),
            ("_eq_current", "eq_current"),
            ("_lte", "lte"),
            ("_gte", "gte"),
            ("_lt", "lt"),
            ("_gt", "gt"),
            ("_in", "in"),
        ):
            if key.endswith(suffix):
                return key[: -len(suffix)], op
        return key, None

    @classmethod
    def _normalize_where(cls, where: dict[str, typing.Any], current_policy_id: int | None) -> dict[str, typing.Any]:
        normalized = {}
        for raw_key, raw_predicate in where.items():
            column, op = cls._split_operator_key(raw_key)
            if op == "lt_current":
                normalized_predicate = {"lt": "$current_policy_id"}
            elif op == "eq_current":
                normalized_predicate = "$current_policy_id"
            elif op is not None:
                normalized_predicate = {op: raw_predicate}
            else:
                normalized_predicate = raw_predicate
            normalized[column] = normalized_predicate
        return normalized

    @staticmethod
    def _unwrap_transforms(dataset):
        while hasattr(dataset, "_dataset"):
            dataset = dataset._dataset  # noqa: SLF001
        return dataset

    @classmethod
    def _iter_hf_datasets(cls, dataset):
        base = cls._unwrap_transforms(dataset)
        if isinstance(base, torch.utils.data.ConcatDataset):
            start = 0
            for child in base.datasets:
                child_dataset = cls._unwrap_transforms(child)
                yield start, child_dataset
                start += len(child_dataset)
            return
        yield 0, base

    def _build_indices(self) -> None:
        if self._group_indices is not None:
            return

        normalized_groups = [
            (group, self._normalize_where(group.where, self.config.current_policy_id)) for group in self.config.groups
        ]
        group_indices = {group.name: [] for group in self.config.groups}
        group_episode_candidates: dict[str, dict[EpisodeKey, list[FrameRef]]] = {
            group.name: {} for group in self.config.groups if group.episode_replay is not None
        }

        logging.info(
            "Building weighted column sampler groups: %s",
            ", ".join(f"{group.name}(weight={group.weight}, where={group.where})" for group in self.config.groups),
        )

        for global_offset, base_dataset in self._iter_hf_datasets(self.dataset):
            if hasattr(base_dataset, "hf_dataset"):
                hf_dataset = base_dataset.hf_dataset
                column_names = set(hf_dataset.column_names)
                required_columns = {column for _, where in normalized_groups for column in where}
                if group_episode_candidates:
                    required_columns.update({"episode_index", "frame_index"})
                missing_columns = sorted(required_columns - column_names)
                if missing_columns:
                    logging.warning(
                        "Skipping fast sampler columns missing from %s: %s",
                        getattr(base_dataset, "repo_id", base_dataset),
                        missing_columns,
                    )
                    continue
                columns = {column: self._column_as_numpy(hf_dataset, column) for column in required_columns}
                num_rows = len(hf_dataset)
                for group, where in normalized_groups:
                    mask = self._vectorized_group_mask(columns, where, self.config.current_policy_id, num_rows)
                    matched = np.nonzero(mask)[0]
                    if matched.size:
                        group_indices[group.name].extend((matched + global_offset).tolist())
                        if group.episode_replay is not None:
                            repo_id = str(getattr(base_dataset, "repo_id", "unknown_repo"))
                            episode_values = columns["episode_index"][matched]
                            frame_values = columns["frame_index"][matched]
                            candidates = group_episode_candidates[group.name]
                            for local_idx, episode_index, frame_index in zip(
                                matched, episode_values, frame_values, strict=True
                            ):
                                episode_key = EpisodeKey(repo_id, int(episode_index))
                                candidates.setdefault(episode_key, []).append(
                                    FrameRef(
                                        frame_index=int(frame_index),
                                        sample_index=global_offset + int(local_idx),
                                    )
                                )
                continue

            logging.info("Using slow sample-based weighted sampler scan for %s", type(base_dataset).__name__)
            for local_idx in range(len(base_dataset)):
                sample = base_dataset[local_idx]
                for group, where in normalized_groups:
                    if all(
                        column in sample
                        and self._matches_predicate(sample[column], predicate, self.config.current_policy_id)
                        for column, predicate in where.items()
                    ):
                        sample_index = global_offset + local_idx
                        group_indices[group.name].append(sample_index)
                        if group.episode_replay is not None:
                            if "episode_index" not in sample or "frame_index" not in sample:
                                raise ValueError(
                                    "Episode replay requires LeRobot episode_index and frame_index columns."
                                )
                            repo_id = str(getattr(base_dataset, "repo_id", "unknown_repo"))
                            episode_key = EpisodeKey(repo_id, int(self._extract_scalar(sample["episode_index"])))
                            group_episode_candidates[group.name].setdefault(episode_key, []).append(
                                FrameRef(
                                    frame_index=int(self._extract_scalar(sample["frame_index"])),
                                    sample_index=sample_index,
                                )
                            )

        active_groups = []
        for group in self.config.groups:
            count = len(group_indices[group.name])
            logging.info("Weighted column sampler group %s: %d samples", group.name, count)
            if group.episode_replay is not None:
                candidates = group_episode_candidates[group.name]
                logging.info(
                    "Episode replay group %s: %d episodes, %d frames, mode=%s, frames_per_turn=%d",
                    group.name,
                    len(candidates),
                    sum(len(refs) for refs in candidates.values()),
                    group.episode_replay.mode,
                    group.episode_replay.frames_per_turn,
                )
            if count == 0:
                message = f"Weighted column sampler group {group.name!r} matched no samples."
                if self.config.drop_empty_groups:
                    logging.warning("%s Dropping it because drop_empty_groups=True.", message)
                    continue
                raise ValueError(message)
            active_groups.append(group)
        if not active_groups:
            raise ValueError("Weighted column sampler has no non-empty groups.")

        self._group_indices = group_indices
        self._group_episode_candidates = group_episode_candidates
        self._active_groups = tuple(active_groups)

    def _batch_counts(self, batch_index: int) -> list[int]:
        assert self._active_groups is not None
        weights = np.asarray([max(0.0, group.weight) for group in self._active_groups], dtype=np.float64)
        if np.sum(weights) <= 0:
            raise ValueError("Weighted column sampler group weights must sum to a positive value.")
        raw_counts = weights / np.sum(weights) * self.batch_size
        counts = np.floor(raw_counts).astype(np.int64)
        remainder = self.batch_size - int(np.sum(counts))
        fractional_order = np.argsort(-(raw_counts - counts))
        for offset in range(remainder):
            counts[int(fractional_order[(batch_index + offset) % len(fractional_order)])] += 1
        return counts.tolist()

    @staticmethod
    def _extend_to_length(indices: list[int], target_len: int) -> list[int]:
        if len(indices) >= target_len:
            return indices[:target_len]
        repeated = indices.copy()
        while len(repeated) < target_len:
            repeated.extend(indices)
        return repeated[:target_len]

    def _num_complete_batches(self) -> int:
        assert self._active_groups is not None
        assert self._group_indices is not None
        batches = 1
        for batch_index in range(len(self._active_groups)):
            counts = self._batch_counts(batch_index)
            for group, count in zip(self._active_groups, counts, strict=True):
                if count <= 0:
                    continue
                batches = max(batches, math.ceil(len(self._group_indices[group.name]) / count))
        return max(1, batches)

    def __iter__(self):
        self._build_indices()
        assert self._active_groups is not None
        assert self._group_indices is not None
        assert self._group_episode_candidates is not None
        rng = random.Random(self.seed + self.epoch)
        num_complete_batches = self._num_complete_batches()
        max_counts = [
            max(self._batch_counts(i)[j] for i in range(len(self._active_groups)))
            for j in range(len(self._active_groups))
        ]
        groups = []
        for group, max_count in zip(self._active_groups, max_counts, strict=True):
            if group.episode_replay is not None:
                groups.append(None)
                continue
            indices = self._group_indices[group.name].copy()
            if self.shuffle:
                rng.shuffle(indices)
            groups.append(self._extend_to_length(indices, num_complete_batches * max(1, max_count)))
        cursors = [0] * len(groups)

        for batch_index in range(num_complete_batches):
            batch_indices = []
            for group_index, (group, count) in enumerate(
                zip(self._active_groups, self._batch_counts(batch_index), strict=True)
            ):
                if group.episode_replay is not None:
                    policy_suffix = (
                        "none" if self.config.current_policy_id is None else str(self.config.current_policy_id)
                    )
                    batch_indices.extend(
                        self.episode_replay_state.sample(
                            pool_key=f"{self.episode_replay_scope}:{group.name}:policy={policy_suffix}",
                            candidates=self._group_episode_candidates[group.name],
                            count=count,
                            config=group.episode_replay,
                        )
                    )
                    continue
                start = cursors[group_index]
                end = start + count
                batch_indices.extend(groups[group_index][start:end])  # type: ignore[index]
                cursors[group_index] = end
            rng.shuffle(batch_indices)
            yield from batch_indices
        self.epoch += 1

    def __len__(self):
        self._build_indices()
        return self._num_complete_batches() * self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def _merge_sampler_where(
    base_where: dict[str, typing.Any], column: str, predicate: typing.Any
) -> dict[str, typing.Any]:
    if any(WeightedColumnGroupSampler._split_operator_key(key)[0] == column for key in base_where):
        raise ValueError(f"Sampler where already contains {column!r}; it cannot also be generated by sampler type.")
    return {**base_where, column: predicate}


def _column_unique_values(
    dataset,
    *,
    column: str,
    where: dict[str, typing.Any],
    current_policy_id: int | None,
) -> tuple[typing.Any, ...]:
    normalized_where = WeightedColumnGroupSampler._normalize_where(where, current_policy_id)
    values_by_key = {}
    for _, base_dataset in WeightedColumnGroupSampler._iter_hf_datasets(dataset):
        if hasattr(base_dataset, "hf_dataset"):
            hf_dataset = base_dataset.hf_dataset
            column_names = set(hf_dataset.column_names)
            required_columns = set(normalized_where) | {column}
            missing_columns = sorted(required_columns - column_names)
            if missing_columns:
                logging.warning(
                    "Skipping fast sampler value scan for %s because columns are missing: %s",
                    getattr(base_dataset, "repo_id", base_dataset),
                    missing_columns,
                )
                continue
            columns = {required_column: hf_dataset[required_column] for required_column in required_columns}
            for local_idx in range(len(hf_dataset)):
                if all(
                    WeightedColumnGroupSampler._matches_predicate(
                        columns[where_column][local_idx], predicate, current_policy_id
                    )
                    for where_column, predicate in normalized_where.items()
                ):
                    value = WeightedColumnGroupSampler._extract_scalar(columns[column][local_idx])
                    values_by_key.setdefault((type(value).__name__, repr(value)), value)
            continue

        logging.info("Using slow sample-based sampler value scan for %s", type(base_dataset).__name__)
        for local_idx in range(len(base_dataset)):
            sample = base_dataset[local_idx]
            if column not in sample:
                continue
            if all(
                where_column in sample
                and WeightedColumnGroupSampler._matches_predicate(sample[where_column], predicate, current_policy_id)
                for where_column, predicate in normalized_where.items()
            ):
                value = WeightedColumnGroupSampler._extract_scalar(sample[column])
                values_by_key.setdefault((type(value).__name__, repr(value)), value)

    return tuple(values_by_key[key] for key in sorted(values_by_key, key=lambda item: item[1]))


def _sampler_config_as_weighted_groups(dataset, sampler_config: _config.SamplerConfig) -> _config.SamplerConfig:
    if sampler_config.type == "weighted_column_groups":
        return sampler_config

    if sampler_config.type == "column_balanced":
        if sampler_config.column is None:
            raise ValueError("SamplerConfig(type='column_balanced') requires column.")
        if sampler_config.column in sampler_config.where:
            raise ValueError("SamplerConfig(type='column_balanced') where must not already filter the balanced column.")
        values = sampler_config.values
        if values is None:
            values = _column_unique_values(
                dataset,
                column=sampler_config.column,
                where=sampler_config.where,
                current_policy_id=sampler_config.current_policy_id,
            )
        if not values:
            raise ValueError(
                f"SamplerConfig(type='column_balanced') found no values for column {sampler_config.column!r}."
            )
        return dataclasses.replace(
            sampler_config,
            type="weighted_column_groups",
            groups=tuple(
                _config.SamplerGroupConfig(
                    name=f"{sampler_config.column}={value}",
                    weight=1.0,
                    where=_merge_sampler_where(sampler_config.where, sampler_config.column, value),
                )
                for value in values
            ),
        )

    if sampler_config.type == "recent_policy":
        if sampler_config.current_policy_id is None:
            raise ValueError("SamplerConfig(type='recent_policy') requires current_policy_id.")
        if sampler_config.recent_policy_window < 1:
            raise ValueError("SamplerConfig(type='recent_policy') requires recent_policy_window >= 1.")
        policy_column = sampler_config.policy_id_column
        if policy_column in sampler_config.where:
            raise ValueError("SamplerConfig(type='recent_policy') where must not already filter the policy column.")

        current_policy_id = sampler_config.current_policy_id
        recent_start = (
            current_policy_id - sampler_config.recent_policy_window + 1
            if sampler_config.include_current_policy
            else current_policy_id - sampler_config.recent_policy_window
        )
        groups = []
        if sampler_config.include_current_policy and sampler_config.current_policy_weight > 0:
            groups.append(
                _config.SamplerGroupConfig(
                    name="current_policy",
                    weight=sampler_config.current_policy_weight,
                    where=_merge_sampler_where(sampler_config.where, policy_column, current_policy_id),
                )
            )
        if sampler_config.recent_policy_weight > 0:
            groups.append(
                _config.SamplerGroupConfig(
                    name="recent_policies",
                    weight=sampler_config.recent_policy_weight,
                    where=_merge_sampler_where(
                        sampler_config.where,
                        policy_column,
                        {"gte": recent_start, "lt": current_policy_id},
                    ),
                )
            )
        if sampler_config.older_policy_weight > 0:
            groups.append(
                _config.SamplerGroupConfig(
                    name="older_policies",
                    weight=sampler_config.older_policy_weight,
                    where=_merge_sampler_where(sampler_config.where, policy_column, {"lt": recent_start}),
                )
            )
        if not groups:
            raise ValueError("SamplerConfig(type='recent_policy') produced no groups with positive weight.")
        return dataclasses.replace(
            sampler_config,
            type="weighted_column_groups",
            groups=tuple(groups),
        )

    raise ValueError(f"Unsupported sampler type: {sampler_config.type!r}")


def create_training_sampler(
    dataset,
    *,
    data_config: _config.DataConfig,
    batch_size: int,
    shuffle: bool,
    seed: int,
    episode_replay_state: EpisodeReplayState | None = None,
    episode_replay_scope: str = "default",
) -> torch.utils.data.Sampler | None:
    if data_config.sampler is not None:
        sampler_config = _sampler_config_as_weighted_groups(dataset, data_config.sampler)
        logging.info("Using WeightedColumnGroupSampler with config: %s", sampler_config)
        return WeightedColumnGroupSampler(
            dataset,
            batch_size=batch_size,
            sampler_config=sampler_config,
            shuffle=shuffle,
            seed=seed,
            episode_replay_state=episode_replay_state,
            episode_replay_scope=episode_replay_scope,
        )

    use_balanced_sampler = (
        data_config.balanced_data_source_values is not None
        or data_config.balance_online_success_failure
        or data_config.balance_online_termination
    )
    if not use_balanced_sampler:
        return None

    logging.info(
        "Using BalancedDataSourceSampler for balanced data_source sampling across values %s "
        "(balance_online_success_failure=%s, balance_online_termination=%s)",
        data_config.balanced_data_source_values,
        data_config.balance_online_success_failure,
        data_config.balance_online_termination,
    )
    return BalancedDataSourceSampler(
        dataset,
        batch_size=batch_size,
        data_source_values=data_config.balanced_data_source_values,
        balance_online_success_failure=data_config.balance_online_success_failure,
        balance_online_termination=data_config.balance_online_termination,
        shuffle=shuffle,
        seed=seed,
    )


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        specs = model_config.inputs_spec()
        self._observation_spec, self._action_spec = specs[:2]

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class SlicedIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, start_index: int, num_frames: int | None = None):
        self._dataset = dataset
        self._start_index = start_index
        total_length = len(dataset) - start_index
        self._length = min(num_frames, total_length) if num_frames is not None else total_length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return int(index + self._start_index), self._dataset[index + self._start_index]


class ExplicitFrameIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, frame_indices: np.ndarray):
        self._dataset = dataset
        self._frame_indices = np.asarray(frame_indices, dtype=np.int64)

    def __len__(self):
        return int(self._frame_indices.shape[0])

    def __getitem__(self, index):
        source_index = int(self._frame_indices[index])
        return int(index), self._dataset[source_index]


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    episodes: Sequence[int] | None = None,
    reuse_datasets: bool = False,
) -> Dataset:
    """Create a dataset for training.

    When ``reuse_datasets`` is True (async streaming reload path), per-repo LeRobotDataset objects are
    cached by repo_id and reused across calls if their on-disk content is unchanged, so only the grown
    or newly-added repo is reconstructed. Off by default so train.py / norm-stats are unaffected.
    """
    repo_ids = data_config.effective_repo_ids()
    if not repo_ids:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if len(repo_ids) == 1 and repo_ids[0] == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_episode_map = (
        {spec.repo_id: spec.episode_ids for spec in data_config.repo_specs}
        if data_config.repo_specs is not None
        else {}
    )
    repo_default_map = (
        {spec.repo_id: spec.missing_feature_defaults for spec in data_config.repo_specs}
        if data_config.repo_specs is not None
        else {}
    )

    snapshot_map = {s.repo_id: s.advantage_snapshot for s in data_config.repo_specs or ()}

    def _create_single_dataset(repo_id: str, repo_episodes: Sequence[int] | None = None) -> Dataset:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        defaults = repo_default_map.get(repo_id, {})
        validate_observation_sequence_features(
            data_config,
            model_config,
            dataset_meta,
            repo_id=repo_id,
            missing_feature_defaults=defaults,
        )
        delta_timestamps = _delta_timestamps_for(data_config, model_config, action_horizon, dataset_meta)
        episodes_arg = repo_episodes if repo_episodes is not None else episodes
        if reuse_datasets:
            signature = (
                *_dataset_signature(repo_id, episodes_arg, delta_timestamps, defaults, dataset_meta),
                snapshot_map.get(repo_id),
            )
            cached = _LEROBOT_DATASET_CACHE.get(repo_id)
            if cached is not None and cached[0] == signature:
                logging.info("Reusing cached dataset for %s (content unchanged).", repo_id)
                return cached[1]
        # First construction of a repo validates timestamps; a later rebuild (repo grew) skips the
        # O(num_frames) timestamp-sync scan. Non-async callers always validate.
        validate_timestamps = (not reuse_datasets) or (repo_id not in _VALIDATED_REPO_IDS)
        dataset = _build_lerobot_dataset(
            repo_id,
            delta_timestamps=delta_timestamps,
            episodes=episodes_arg,
            validate_timestamps=validate_timestamps,
        )
        _add_missing_hf_dataset_columns(dataset, defaults)
        _apply_advantage_snapshot(dataset, snapshot_map.get(repo_id))
        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        _VALIDATED_REPO_IDS.add(repo_id)
        if reuse_datasets:
            _LEROBOT_DATASET_CACHE[repo_id] = (signature, dataset)
        return dataset

    if len(repo_ids) == 1:
        return _create_single_dataset(repo_ids[0], repo_episode_map.get(repo_ids[0]))

    return torch.utils.data.ConcatDataset(
        [_create_single_dataset(repo_id, repo_episode_map.get(repo_id)) for repo_id in repo_ids]
    )


def _repack_source_key(data_config: _config.DataConfig, target_key: str) -> str:
    for transform in data_config.repack_transforms.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue
        flat_structure = _transforms.flatten_dict(transform.structure)
        source_key = flat_structure.get(target_key)
        if source_key is not None:
            return str(source_key)
    raise KeyError(f"Could not find source key for {target_key!r} in DataConfig repack transforms.")


def _remove_media_features(dataset: lerobot_dataset.LeRobotDataset) -> None:
    features = dataset.meta.info.get("features", {})
    media_keys = [key for key, feature in features.items() if feature.get("dtype") in {"image", "video"}]
    media_columns = [key for key in media_keys if key in dataset.hf_dataset.column_names]
    if media_columns:
        dataset.hf_dataset = dataset.hf_dataset.remove_columns(media_columns)
    for key in media_keys:
        features.pop(key, None)


def create_lowdim_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    episodes: Sequence[int] | None = None,
) -> Dataset:
    """Create a LeRobot dataset for normalization stats without decoding images/videos."""
    repo_ids = data_config.effective_repo_ids()
    if not repo_ids:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if len(repo_ids) == 1 and repo_ids[0] == "fake":
        return FakeDataset(model_config, num_samples=1024)

    state_key = _repack_source_key(data_config, "observation/state")
    next_state_key = _repack_source_key(data_config, "observation/next_state") if data_config.use_next_obs else None

    repo_episode_map = (
        {spec.repo_id: spec.episode_ids for spec in data_config.repo_specs}
        if data_config.repo_specs is not None
        else {}
    )
    repo_default_map = (
        {spec.repo_id: spec.missing_feature_defaults for spec in data_config.repo_specs}
        if data_config.repo_specs is not None
        else {}
    )

    def _create_single_dataset(repo_id: str, repo_episodes: Sequence[int] | None = None) -> Dataset:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        defaults = repo_default_map.get(repo_id, {})
        validate_observation_sequence_features(
            data_config,
            model_config,
            dataset_meta,
            repo_id=repo_id,
            missing_feature_defaults=defaults,
        )
        delta_timestamps = {
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        }
        observation_horizon = getattr(model_config, "observation_horizon", 1)
        if observation_horizon > 1:
            if data_config.observation_stride < 1:
                raise ValueError(f"DataConfig.observation_stride must be >= 1, got {data_config.observation_stride}")
            observation_delta_timestamps = [
                t * data_config.observation_stride / dataset_meta.fps for t in range(-(observation_horizon - 1), 1)
            ]
            delta_timestamps[state_key] = observation_delta_timestamps
            if next_state_key is not None and _dataset_has_key(dataset_meta, next_state_key):
                delta_timestamps[next_state_key] = observation_delta_timestamps
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
            episodes=repo_episodes if repo_episodes is not None else episodes,
        )
        _add_missing_hf_dataset_columns(dataset, defaults)
        _remove_media_features(dataset)
        return dataset

    if len(repo_ids) == 1:
        return _create_single_dataset(repo_ids[0], repo_episode_map.get(repo_ids[0]))

    return torch.utils.data.ConcatDataset(
        [_create_single_dataset(repo_id, repo_episode_map.get(repo_id)) for repo_id in repo_ids]
    )


_LOWDIM_NORM_TRANSFORM_TYPES = (
    _transforms.SubsampleActions,
    _transforms.DeltaActions,
    _transforms.PadStatesAndActions,
)


def build_lowdim_norm_transforms(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
) -> list[_transforms.DataTransformFn]:
    repack_structure = {
        "observation/state": _repack_source_key(data_config, "observation/state"),
        "actions": _repack_source_key(data_config, "actions"),
    }
    if data_config.use_next_obs:
        repack_structure["observation/next_state"] = _repack_source_key(data_config, "observation/next_state")
        repack_structure["next_actions"] = _repack_source_key(data_config, "next_actions")

    return [
        _transforms.RepackTransform(repack_structure),
        _transforms.LowDimNormInputs(model_config.action_horizon, use_next_obs=data_config.use_next_obs),
        *[
            transform
            for transform in data_config.data_transforms.inputs
            if isinstance(transform, _LOWDIM_NORM_TRANSFORM_TYPES)
        ],
    ]


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.effective_repo_ids() != ("fake",) and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        build_training_transforms(data_config, norm_stats=norm_stats),
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.effective_repo_ids() != ("fake",) and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        build_training_transforms(data_config, norm_stats=norm_stats),
        is_batched=is_batched,
    )


def build_training_transforms(
    data_config: _config.DataConfig,
    *,
    norm_stats: dict[str, _transforms.NormStats],
) -> list[_transforms.DataTransformFn]:
    return [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def build_inference_transforms(
    data_config: _config.DataConfig,
    *,
    norm_stats: dict[str, _transforms.NormStats],
) -> list[_transforms.DataTransformFn]:
    inference_model_transforms = [
        t for t in data_config.model_transforms.inputs if not getattr(t, "is_train_only", False)
    ]
    transforms = [
        *data_config.repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
    ]
    if norm_stats:
        transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.extend(inference_model_transforms)
    return transforms


def _get_data_factory_attr(data_factory: object, attr_name: str) -> object:
    if hasattr(data_factory, attr_name):
        return getattr(data_factory, attr_name)
    if hasattr(data_factory, "base_factory"):
        return _get_data_factory_attr(data_factory.base_factory, attr_name)
    raise AttributeError(f"Data factory {type(data_factory)!r} does not expose {attr_name!r}.")


def _data_factory_has_field(data_factory: object, field_name: str) -> bool:
    if dataclasses.is_dataclass(data_factory):
        field_names = {field.name for field in dataclasses.fields(data_factory)}
        if field_name in field_names:
            return True
    if hasattr(data_factory, "base_factory"):
        return _data_factory_has_field(data_factory.base_factory, field_name)
    return hasattr(data_factory, field_name)


def _replace_data_factory_fields(data_factory: object, replace_kwargs: dict[str, object]) -> object:
    if not replace_kwargs:
        return data_factory
    if not dataclasses.is_dataclass(data_factory):
        fields = ", ".join(sorted(replace_kwargs))
        raise TypeError(f"Cannot override fields [{fields}] for non-dataclass data factory {type(data_factory)!r}.")

    field_names = {field.name for field in dataclasses.fields(data_factory)}
    direct_replacements = {key: value for key, value in replace_kwargs.items() if key in field_names}
    nested_replacements = {key: value for key, value in replace_kwargs.items() if key not in field_names}
    if nested_replacements:
        if "base_factory" not in field_names:
            fields = ", ".join(sorted(nested_replacements))
            raise TypeError(f"Data factory {type(data_factory)!r} does not expose fields [{fields}].")
        direct_replacements["base_factory"] = _replace_data_factory_fields(
            data_factory.base_factory,
            nested_replacements,
        )
    return dataclasses.replace(data_factory, **direct_replacements)


def resolve_data_config_override(
    config: _config.TrainConfig,
    *,
    repo_id: str | None,
) -> _config.DataConfig:
    data_factory = config.data
    replace_kwargs: dict[str, object] = {}
    if repo_id is not None:
        assets = _get_data_factory_attr(data_factory, "assets")
        asset_id = assets.asset_id or _get_data_factory_attr(data_factory, "repo_id")
        replace_kwargs.update(
            repo_id=repo_id,
            repo_ids=None,
            assets=dataclasses.replace(assets, asset_id=asset_id),
        )
    if replace_kwargs:
        data_factory = _replace_data_factory_fields(data_factory, replace_kwargs)
    return data_factory.create(config.assets_dirs, config.model)


def load_norm_stats_for_repo(
    config: _config.TrainConfig,
    *,
    repo_id: str,
) -> tuple[dict[str, _transforms.NormStats], str]:
    assets = _get_data_factory_attr(config.data, "assets")
    data_factory = _replace_data_factory_fields(
        config.data,
        {
            "repo_id": repo_id,
            "repo_ids": None,
            "assets": dataclasses.replace(assets, asset_id=repo_id),
        },
    )
    override_data_config = data_factory.create(config.assets_dirs, config.model)
    if override_data_config.norm_stats is None:
        raise ValueError(f"Could not load norm stats for repo id: {repo_id}")
    return override_data_config.norm_stats, f"repo_id:{repo_id}"


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool | None = None,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization. If unset, uses ``not config.use_norm_stats``.
        framework: The framework to use ("jax" or "pytorch").
    """
    if skip_norm_stats is None:
        skip_norm_stats = not config.use_norm_stats

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    action_horizon = (
        config.model.async_action_horizon
        if getattr(config.model, "async_action_horizon", -1) > 0
        else config.model.action_horizon
    )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    framework: str = "jax",
    reuse_datasets: bool = False,
    episode_replay_state: EpisodeReplayState | None = None,
    episode_replay_scope: str = "default",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, reuse_datasets=reuse_datasets)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    use_configured_sampler = data_config.sampler is not None
    use_balanced_sampler = (
        data_config.balanced_data_source_values is not None
        or data_config.balance_online_success_failure
        or data_config.balance_online_termination
    )
    if use_configured_sampler and use_balanced_sampler:
        raise ValueError("DataConfig.sampler and balanced_data_source_* settings are mutually exclusive.")
    if sampler is None:
        sampler = create_training_sampler(
            dataset,
            data_config=data_config,
            batch_size=local_batch_size,
            shuffle=shuffle,
            seed=seed,
            episode_replay_state=episode_replay_state,
            episode_replay_scope=episode_replay_scope,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = _dataloader_mp_context(num_workers)

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=persistent_workers and num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map_with_path(
                        lambda path, x: self._make_jax_array(path, x),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)

    def _make_jax_array(self, path, value):
        array = np.asarray(value)
        if array.dtype.kind in {"U", "S", "O"}:
            raise TypeError(
                f"Cannot send non-numeric batch field {jax.tree_util.keystr(path)} with dtype {array.dtype} to JAX."
            )
        return jax.make_array_from_process_local_data(self._sharding, value)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def collate_indexed(items):
    indices = np.asarray([i for i, _ in items], dtype=np.int64)
    samples = [sample for _, sample in items]
    batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)
    return indices, batch


def worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _worker_init_fn(worker_id: int) -> None:
    worker_init_fn(worker_id)


def _to_numpy(x: typing.Any) -> typing.Any:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def create_inference_data_loader(
    config: _config.TrainConfig,
    *,
    data_config: _config.DataConfig | None = None,
    norm_stats: dict[str, _transforms.NormStats],
    inference_batch_size: int,
    num_workers: int = 16,
    pin_memory: bool = True,
    prefetch_factor: int | None = 8,
    persistent_workers: bool = True,
    start_index: int = 0,
    num_frames: int | None = None,
    episode_ids: list[int] | None = None,
) -> DatasetSelection:
    data_config = data_config or config.data.create(config.assets_dirs, config.model)

    action_horizon = (
        config.model.async_action_horizon
        if getattr(config.model, "async_action_horizon", -1) > 0
        else config.model.action_horizon
    )
    dataset = create_torch_dataset(data_config, action_horizon, config.model)

    raw_dataset = dataset
    while hasattr(raw_dataset, "_dataset"):
        raw_dataset = raw_dataset._dataset

    selected_episode_ids: list[int] | None = None
    if start_index > 0:
        episode_data_index = raw_dataset.episode_data_index
        num_episodes = int(raw_dataset.num_episodes)
        if start_index >= num_episodes:
            raise ValueError(f"start_index {start_index} >= num_episodes {num_episodes}")
        start_frame = int(_to_numpy(episode_data_index["from"][start_index]).item())
    else:
        start_frame = 0

    transformed_dataset = TransformedDataset(dataset, build_inference_transforms(data_config, norm_stats=norm_stats))

    mp_context = _dataloader_mp_context(num_workers)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    if episode_ids is not None:
        episode_data_index = raw_dataset.episode_data_index
        num_episodes = int(raw_dataset.num_episodes)
        selected_episode_ids = sorted(int(ep) for ep in episode_ids)
        for ep_idx in selected_episode_ids:
            if ep_idx < 0 or ep_idx >= num_episodes:
                raise ValueError(f"episode_id {ep_idx} out of range for dataset with {num_episodes} episodes")

        frame_indices: list[np.ndarray] = []
        for ep_idx in selected_episode_ids:
            ep_start = int(_to_numpy(episode_data_index["from"][ep_idx]).item())
            ep_end = int(_to_numpy(episode_data_index["to"][ep_idx]).item())
            if ep_end > ep_start:
                frame_indices.append(np.arange(ep_start, ep_end, dtype=np.int64))
        flat_frame_indices = np.concatenate(frame_indices, axis=0) if frame_indices else np.empty((0,), dtype=np.int64)
        indexed_dataset = ExplicitFrameIndexedDataset(transformed_dataset, flat_frame_indices)
        start_frame = 0
    else:
        indexed_dataset = SlicedIndexedDataset(transformed_dataset, start_frame, num_frames)

    dataloader = torch.utils.data.DataLoader(
        indexed_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_indexed,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        generator=generator,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return DatasetSelection(
        dataloader=dataloader,
        raw_dataset=typing.cast(lerobot_dataset.LeRobotDataset, raw_dataset),
        total_frames=len(indexed_dataset),
        start_frame=start_frame,
        selected_episode_ids=selected_episode_ids,
    )


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    @property
    def inner(self) -> "TorchDataLoader":
        return self._data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            if self._data_config.use_next_obs:
                next_obs_batch = {k.replace("next_", ""): v for k, v in batch.items() if k.startswith("next_")}
                next_obs_batch = _model.Observation.from_dict(next_obs_batch)
            else:
                next_obs_batch = None
            if "value" in batch:
                value_batch = batch["value"]
            else:
                value_batch = None
            if "success" in batch:
                success_batch = batch["success"]
            else:
                success_batch = None
            if "reward" in batch:
                reward_batch = batch["reward"]
            else:
                reward_batch = None
            if "next_actions" in batch and self._data_config.use_next_obs:
                next_actions_batch = batch["next_actions"]
            else:
                next_actions_batch = None
            if "control_flag" in batch:
                control_flag_batch = batch["control_flag"]
            else:
                control_flag_batch = None
            if "data_source" in batch:
                data_source_batch = batch["data_source"]
            else:
                data_source_batch = None
            if "terminate" in batch:
                terminate_batch = batch["terminate"]
            else:
                terminate_batch = None
            yield (
                _model.Observation.from_dict(batch),
                batch["actions"],
                value_batch,
                next_obs_batch,
                success_batch,
                reward_batch,
                next_actions_batch,
                control_flag_batch,
                data_source_batch,
                terminate_batch,
            )


class DualSourceDataLoader:
    """Mix a stable multiprocess loader (offline demo + history) with a live current-repo source.

    Motivation (docs/async_reload_latency_optimization.md §11): within a policy generation only the current
    deployment repo grows; the offline demo + history repos are immutable. So we keep those in the existing
    multiprocess (forkserver) TorchDataLoader that never rebuilds within a generation, and put the current
    repo in a LiveCurrentRepoDataset read on the main thread that grows in place (no dataloader rebuild, no
    worker respawn, no first-batch storm). Each yielded batch is a collated dict identical in shape to what
    TorchDataLoader yields, so DataLoaderImpl can unpack it unchanged.

    Per-batch composition follows the sampler group weights: the stable loader supplies the non-current
    groups' rows (its own sampler already balances offline_demo vs off_policy), and the live source supplies
    the current-policy group's rows. `current_batch` is the number of rows drawn from the live source each
    step; `stable_batch` = local_batch_size - current_batch comes from the stable loader.
    """

    def __init__(
        self,
        *,
        stable_loader: "TorchDataLoader",
        live_dataset: Dataset,
        local_batch_size: int,
        current_batch: int,
        sharding: jax.sharding.Sharding | None,
        seed: int,
        num_batches: int | None = None,
        episode_replay_config: _config.EpisodeReplayConfig | None = None,
        episode_replay_state: EpisodeReplayState | None = None,
        episode_replay_pool_key: str = "default:on_policy:policy=none",
    ):
        self._stable_loader = stable_loader
        self._live_dataset = live_dataset
        self._local_batch_size = local_batch_size
        self._current_batch = max(0, min(int(current_batch), local_batch_size))
        self._stable_batch = local_batch_size - self._current_batch
        self._num_batches = num_batches
        self._rng = np.random.default_rng(seed)
        self._episode_replay_config = episode_replay_config
        self._episode_replay_state = episode_replay_state or EpisodeReplayState(seed=seed)
        self._episode_replay_pool_key = episode_replay_pool_key
        self._live_candidates_lock = threading.Lock()
        self._live_candidates_signature: tuple[int, int] | None = None
        self._live_candidates_cache: dict[EpisodeKey, list[FrameRef]] | None = None
        self._live_generation = 0
        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B")
            )
        self._sharding = sharding

    @property
    def generation(self) -> int:
        return self._live_generation

    @property
    def live_dataset(self) -> Dataset:
        return self._live_dataset

    def live_current_repo(self) -> "LiveCurrentRepoDataset | None":
        """Unwrap the (possibly transform-wrapped) live source to the growable LiveCurrentRepoDataset."""
        node = self._live_dataset
        seen = 0
        while node is not None and not isinstance(node, LiveCurrentRepoDataset) and seen < 8:
            node = getattr(node, "_dataset", None)
            seen += 1
        return node if isinstance(node, LiveCurrentRepoDataset) else None

    def poll_and_grow_current(self, **kwargs) -> int:
        """Refresh the current source while keeping stable workers and live decode threads."""
        live = self.live_current_repo()
        revision = live.revision if live is not None else None
        added = live.poll_and_grow(**kwargs) if live is not None else 0
        if live is not None and live.revision != revision:
            with self._live_candidates_lock:
                self._live_candidates_signature = None
                self._live_candidates_cache = None
                self._live_generation += 1
        return added

    def set_current_batch(self, current_batch: int) -> None:
        self._current_batch = max(0, min(int(current_batch), self._local_batch_size))
        self._stable_batch = self._local_batch_size - self._current_batch

    def _sample_live_rows(self) -> list[dict]:
        n = self._current_batch
        live_len = len(self._live_dataset)
        if n <= 0 or live_len == 0:
            return []
        if self._episode_replay_config is not None:
            candidates = self._live_episode_candidates()
            idxs = self._episode_replay_state.sample(
                pool_key=self._episode_replay_pool_key,
                candidates=candidates,
                count=n,
                config=self._episode_replay_config,
            )
            return [self._live_dataset[i] for i in idxs]
        replace = n > live_len
        idxs = self._rng.choice(live_len, size=n, replace=replace)
        return [self._live_dataset[int(i)] for i in idxs]

    def _live_episode_candidates(self) -> dict[EpisodeKey, list[FrameRef]]:
        live = self.live_current_repo()
        if live is None:
            raise ValueError("Episode replay requires a LiveCurrentRepoDataset live source.")
        hf_dataset = live.hf_dataset
        signature = (id(hf_dataset), len(hf_dataset))
        with self._live_candidates_lock:
            if self._live_candidates_signature == signature and self._live_candidates_cache is not None:
                return self._live_candidates_cache
            required = {"episode_index", "frame_index"}
            missing = sorted(required - set(hf_dataset.column_names))
            if missing:
                raise ValueError(f"Episode replay live source {live.repo_id!r} is missing columns: {missing}.")
            episode_values = WeightedColumnGroupSampler._column_as_numpy(hf_dataset, "episode_index")
            frame_values = WeightedColumnGroupSampler._column_as_numpy(hf_dataset, "frame_index")
            candidates: dict[EpisodeKey, list[FrameRef]] = {}
            for sample_index, (episode_index, frame_index) in enumerate(zip(episode_values, frame_values, strict=True)):
                episode_key = EpisodeKey(live.repo_id, int(episode_index))
                candidates.setdefault(episode_key, []).append(
                    FrameRef(frame_index=int(frame_index), sample_index=sample_index)
                )
            self._live_candidates_signature = signature
            self._live_candidates_cache = candidates
            return candidates

    def _start_live_prefetch(self, threads: int = 2, depth: int = 4):
        """Decode live-row batches on background thread(s) so single-thread video decode overlaps compute.

        Each produced item re-samples from the (growing) live dataset, so appended episodes become visible
        as soon as they land — no rebuild. Uses a small pool of threads (empirically 2 is the sweet spot;
        more thrash the GIL because video decode is GIL-heavy). Returns (queue, stop_event, [threads]) or
        (None, None, []) if the live source is unused.
        """
        if self._current_batch <= 0:
            return None, None, []
        q: queue_module.Queue = queue_module.Queue(maxsize=depth)
        stop = threading.Event()

        def _worker():
            while not stop.is_set():
                generation = self._live_generation
                rows = self._sample_live_rows()
                while not stop.is_set():
                    try:
                        q.put((generation, rows), timeout=0.2)
                        break
                    except queue_module.Full:
                        continue

        pool = []
        for i in range(max(1, threads)):
            t = threading.Thread(target=_worker, name=f"dual-live-prefetch-{i}", daemon=True)
            t.start()
            pool.append(t)
        return q, stop, pool

    def _take_current_live_rows(self, live_q: queue_module.Queue) -> list[dict]:
        while True:
            generation, live_rows = live_q.get()
            if generation == self._live_generation:
                return live_rows

    def __iter__(self):
        num_items = 0
        use_stable = self._stable_batch > 0 and self._stable_loader is not None
        stable_iter = iter(self._stable_loader.torch_loader) if use_stable else None
        if self._stable_batch > 0 and self._stable_loader is None:
            # No stable loader available (e.g. only the current repo in the view yet): draw the whole
            # batch from the live source so training can proceed.
            self.set_current_batch(self._local_batch_size)
        live_q, live_stop, live_threads = self._start_live_prefetch()
        try:
            yield from self._iterate(num_items, stable_iter, live_q)
        finally:
            if live_stop is not None:
                live_stop.set()
            for t in live_threads:
                t.join(timeout=2.0)

    def _iterate(self, num_items, stable_iter, live_q):
        debug_enabled = os.environ.get("VAM_ASYNC_DATALOADER_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        debug_every = max(1, int(os.environ.get("VAM_ASYNC_DATALOADER_DEBUG_EVERY", "10")))
        while True:
            if self._num_batches is not None and num_items >= self._num_batches:
                return
            items: list[dict] = []
            stable_next_elapsed = 0.0
            stable_split_elapsed = 0.0
            live_elapsed = 0.0
            if stable_iter is not None:
                stable_t0 = time.monotonic()
                try:
                    stable_batch = next(stable_iter)
                except StopIteration:
                    stable_iter = iter(self._stable_loader.torch_loader)
                    stable_batch = next(stable_iter)
                stable_next_elapsed = time.monotonic() - stable_t0
                # stable_batch is an already-collated dict of length _stable_batch; split back to rows
                split_t0 = time.monotonic()
                items.extend(_split_collated_batch(stable_batch, self._stable_batch))
                stable_split_elapsed = time.monotonic() - split_t0
            if live_q is not None:
                live_t0 = time.monotonic()
                items.extend(self._take_current_live_rows(live_q))
                live_elapsed = time.monotonic() - live_t0
            else:
                live_t0 = time.monotonic()
                items.extend(self._sample_live_rows())
                live_elapsed = time.monotonic() - live_t0
            if not items:
                raise RuntimeError("DualSourceDataLoader produced an empty batch (both sources empty).")
            collate_t0 = time.monotonic()
            batch = _collate_fn(items)
            collate_elapsed = time.monotonic() - collate_t0
            array_t0 = time.monotonic()
            batch = jax.tree.map_with_path(lambda path, x: _dual_make_jax_array(self._sharding, path, x), batch)
            array_elapsed = time.monotonic() - array_t0
            num_items += 1
            if debug_enabled and num_items % debug_every == 0:
                live_len = len(self._live_dataset) if self._live_dataset is not None else 0
                live_queue_size = live_q.qsize() if live_q is not None else -1
                logging.info(
                    "DUAL_SOURCE_BATCH index=%d stable_batch=%d current_batch=%d live_len=%d live_queue=%d "
                    "stable_next_ms=%.0f stable_split_ms=%.0f live_get_ms=%.0f collate_ms=%.0f jax_array_ms=%.0f",
                    num_items,
                    self._stable_batch,
                    self._current_batch,
                    live_len,
                    live_queue_size,
                    stable_next_elapsed * 1e3,
                    stable_split_elapsed * 1e3,
                    live_elapsed * 1e3,
                    collate_elapsed * 1e3,
                    array_elapsed * 1e3,
                )
            yield batch


def _dual_make_jax_array(sharding, path, value):
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S", "O"}:
        raise TypeError(f"Cannot send non-numeric batch field {jax.tree_util.keystr(path)} to JAX.")
    return jax.make_array_from_process_local_data(sharding, array)


def _split_collated_batch(batch: dict, n: int) -> list[dict]:
    """Split a collated (possibly nested) dict batch into n row dicts.

    Values may be nested dicts (e.g. ``image`` is a dict of cameras, each a stacked array). We split along
    the leading batch dim n at every leaf; a leaf whose leading dim is not n is broadcast to every row.
    """

    def _index_leaf(x, i):
        a = np.asarray(x)
        return a[i] if (a.ndim >= 1 and a.shape[0] == n) else a

    return [jax.tree.map(lambda x, i=i: _index_leaf(x, i), batch) for i in range(n)]


class DualSourceDataLoaderImpl(DataLoader):
    """DataLoader-protocol wrapper that reuses DataLoaderImpl's dict->tuple unpacking for dual-source."""

    def __init__(self, data_config: _config.DataConfig, dual_loader: DualSourceDataLoader):
        self._data_config = data_config
        self._dual = dual_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def dual(self) -> DualSourceDataLoader:
        return self._dual

    @property
    def generation(self) -> int:
        return self._dual.generation

    def poll_and_grow_current(self, **kwargs) -> int:
        """Grow/refresh the live source without replacing worker processes."""
        return self._dual.poll_and_grow_current(**kwargs)

    def __iter__(self):
        # Delegate the exact dict->tuple unpacking to a throwaway DataLoaderImpl over the dual loader.
        yield from DataLoaderImpl(self._data_config, self._dual).__iter__()


def _repo_constant_policy_id(repo_id: str) -> int | None:
    """Read a repo's constant policy_id from meta/policy_metadata_map.jsonl (first row). None if unknown."""
    import json

    path = lerobot_dataset.HF_LEROBOT_HOME / repo_id / "meta" / "policy_metadata_map.jsonl"
    try:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    return int(float(json.loads(line)["policy_id"]))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return None


def create_dual_source_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    current_repo_id: str,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    current_batch: int | None = None,
    reuse_datasets: bool = True,
    episode_replay_state: EpisodeReplayState | None = None,
    episode_replay_scope: str = "default",
) -> DualSourceDataLoaderImpl:
    """Build a dual-source loader: stable multiprocess loader (all repos except current) + live current repo.

    ``current_repo_id`` is the deployment repo for the current policy generation (the one that grows). It is
    served from a main-thread LiveCurrentRepoDataset; all other repos go through the normal multiprocess
    TorchDataLoader. ``current_batch`` is how many of the ``batch_size`` rows come from the live source each
    step (defaults to the on-policy group's share, i.e. batch_size / num_groups).
    """
    if data_config.repo_specs is None:
        raise ValueError("Dual-source loader requires data_config.repo_specs (async streaming views).")
    current_specs = [s for s in data_config.repo_specs if s.repo_id == current_repo_id]
    stable_specs = [s for s in data_config.repo_specs if s.repo_id != current_repo_id]
    if not current_specs:
        raise ValueError(f"current_repo_id {current_repo_id!r} not in repo_specs.")

    local_batch_size = batch_size // jax.process_count()
    n_groups = len(data_config.sampler.groups) if data_config.sampler is not None else 3
    if current_batch is None:
        current_batch = max(1, local_batch_size // max(1, n_groups))

    replay_groups = (
        [group for group in data_config.sampler.groups if group.episode_replay is not None]
        if data_config.sampler is not None
        else []
    )
    if len(replay_groups) > 1:
        raise ValueError("Dual-source loader supports at most one episode-replay sampler group.")
    replay_group = replay_groups[0] if replay_groups else None
    policy_suffix = (
        "none"
        if data_config.sampler is None or data_config.sampler.current_policy_id is None
        else str(data_config.sampler.current_policy_id)
    )
    replay_pool_key = (
        f"{episode_replay_scope}:{replay_group.name}:policy={policy_suffix}"
        if replay_group is not None
        else f"{episode_replay_scope}:on_policy:policy={policy_suffix}"
    )
    if replay_group is not None:
        logging.info(
            "Dual-source live repo %s uses episode replay group %s (%s, frames_per_turn=%d).",
            current_repo_id,
            replay_group.name,
            replay_group.episode_replay.mode,
            replay_group.episode_replay.frames_per_turn,
        )

    # Live current source: LiveCurrentRepoDataset + the same transforms as the stable source.
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(current_repo_id)
    current_default_map = {s.repo_id: s.missing_feature_defaults for s in current_specs}
    current_defaults = current_default_map.get(current_repo_id, {})
    validate_observation_sequence_features(
        data_config,
        model_config,
        dataset_meta,
        repo_id=current_repo_id,
        missing_feature_defaults=current_defaults,
    )
    delta_timestamps = _delta_timestamps_for(data_config, model_config, action_horizon, dataset_meta)
    live_raw = LiveCurrentRepoDataset(
        current_repo_id,
        delta_timestamps=delta_timestamps,
        missing_feature_defaults=current_defaults,
        episode_ids=current_specs[0].episode_ids,
        advantage_snapshot=current_specs[0].advantage_snapshot,
    )
    live_dataset = transform_dataset(live_raw, data_config, skip_norm_stats=skip_norm_stats)

    # Stable multiprocess loader over the remaining repos, sized to the non-current per-batch count.
    stable_batch = local_batch_size - current_batch
    stable_impl = None
    if stable_batch > 0 and stable_specs:
        # Ensure the stable sampler knows current_policy_id (its on-policy group predicate references it).
        # The current repo is excluded from the stable source, so that group drops out — but _build_indices
        # still normalizes the predicate and needs current_policy_id set. Fall back to the current repo's
        # own policy_id if the data_config sampler didn't carry it.
        stable_config = dataclasses.replace(data_config, repo_specs=tuple(stable_specs))
        if stable_config.sampler is not None and stable_config.sampler.current_policy_id is None:
            current_pid = _repo_constant_policy_id(current_repo_id)
            if current_pid is not None:
                stable_config = dataclasses.replace(
                    stable_config,
                    sampler=dataclasses.replace(stable_config.sampler, current_policy_id=current_pid),
                )
        stable_impl = create_torch_data_loader(
            stable_config,
            model_config,
            action_horizon,
            batch_size=stable_batch * jax.process_count(),
            sharding=None,  # dual loader does the device placement
            skip_norm_stats=skip_norm_stats,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            framework="pytorch",  # yield collated dict rows, not sharded arrays
            reuse_datasets=reuse_datasets,
            episode_replay_state=episode_replay_state,
            episode_replay_scope=episode_replay_scope,
        )

    dual = DualSourceDataLoader(
        stable_loader=stable_impl.inner if stable_impl is not None else None,
        live_dataset=live_dataset,
        local_batch_size=local_batch_size,
        current_batch=current_batch,
        sharding=sharding,
        seed=seed,
        num_batches=num_batches,
        episode_replay_config=replay_group.episode_replay if replay_group is not None else None,
        episode_replay_state=episode_replay_state,
        episode_replay_pool_key=replay_pool_key,
    )
    return DualSourceDataLoaderImpl(data_config, dual)


def _dataset_has_key(dataset_meta: lerobot_dataset.LeRobotDatasetMetadata, key: str) -> bool:
    features = dataset_meta.info.get("features", {})
    return key in features or key in dataset_meta.video_keys or key in dataset_meta.camera_keys


def validate_observation_sequence_features(
    data_config,
    model_config,
    dataset_meta,
    *,
    repo_id: str,
    missing_feature_defaults: dict[str, typing.Any],
) -> None:
    if getattr(model_config, "observation_horizon", 1) <= 1 or data_config.observation_sequence_keys is None:
        return

    missing = []
    for key in data_config.observation_sequence_keys:
        if _dataset_has_key(dataset_meta, key):
            continue
        if key == "value_prediction":
            raise ValueError(
                f"LeRobot repo {repo_id!r} is missing required column 'value_prediction'. "
                "History value conditioning requires real per-frame predictions; "
                "missing_feature_defaults must not synthesize this column. Run "
                "preprocess_data/annotate.py value before publishing the dataset view."
            )
        if key not in missing_feature_defaults:
            missing.append(key)

    if missing:
        raise ValueError(
            f"LeRobot repo {repo_id!r} is missing required observation history columns: {sorted(missing)}."
        )


def _delta_timestamps_for(data_config, model_config, action_horizon, dataset_meta):
    """Canonical delta_timestamps computation, shared by the sync and dual-source data paths (incl. next-obs)."""
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    observation_horizon = getattr(model_config, "observation_horizon", 1)
    if observation_horizon > 1:
        if data_config.observation_sequence_keys is None:
            raise ValueError(
                "DataConfig.observation_sequence_keys must be set when model_config.observation_horizon > 1."
            )
        if data_config.observation_stride < 1:
            raise ValueError(f"DataConfig.observation_stride must be >= 1, got {data_config.observation_stride}")
        obs_delta = [
            t * data_config.observation_stride / dataset_meta.fps for t in range(-(observation_horizon - 1), 1)
        ]
        delta_timestamps.update(dict.fromkeys(data_config.observation_sequence_keys, obs_delta))
        if data_config.use_next_obs:
            delta_timestamps.update(
                {
                    key: obs_delta
                    for key in _next_observation_sequence_keys(data_config, dataset_meta)
                    if key not in delta_timestamps
                }
            )
    return delta_timestamps


def _next_observation_sequence_keys(data_config, dataset_meta) -> set[str]:
    """Raw next-observation keys that should use the same history window as current obs."""
    next_keys = set()
    for transform in data_config.repack_transforms.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue
        flat_structure = _transforms.flatten_dict(transform.structure)
        for target_key, source_key in flat_structure.items():
            if str(target_key).startswith("observation/next_") and _dataset_has_key(dataset_meta, str(source_key)):
                next_keys.add(str(source_key))
    return next_keys
