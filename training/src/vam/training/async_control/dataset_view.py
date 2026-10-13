from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from vam.training.async_control._common import _read_json
from vam.training.async_control._common import _resolve_path
import vam.training.config as _config


@dataclasses.dataclass(frozen=True)
class DatasetView:
    view_id: str
    repo_specs: tuple[_config.RepoSpec, ...]
    sampler: _config.SamplerConfig | None
    target_policy_id: int | None = None
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)


def validate_stream_configs(config: _config.TrainConfig) -> tuple[_config.AsyncTrainStreamConfig, ...]:
    publisher = config.async_training.checkpoint_publisher
    if publisher.merge is not None:
        if publisher.async_publish:
            raise ValueError("Split checkpoint merge requires checkpoint_publisher.async_publish=False.")
        if not publisher.reload_after_publish:
            raise ValueError(
                "Split checkpoint merge requires checkpoint_publisher.reload_after_publish=True so stage handoffs "
                "load the merged checkpoint before training the next policy view."
            )
        if config.async_training.split_round_steps is None or config.async_training.split_round_steps <= 0:
            raise ValueError("Split checkpoint merge requires async_training.split_round_steps to be a positive int.")
    streams = config.async_training.streams or (
        _config.AsyncTrainStreamConfig(
            name="default",
            train_type=tuple(config.train_type),
            data=config.data,
            dataset_view=config.async_training.dataset_view,
            steps_per_cycle=1,
        ),
    )
    names = [stream.name for stream in streams]
    if len(set(names)) != len(names):
        raise ValueError(f"Async training stream names must be unique, got {names}.")
    configured_train_types = set(config.train_type)
    stream_train_types = set()
    for stream in streams:
        if stream.steps_per_cycle < 1:
            raise ValueError(f"Async training stream {stream.name!r} steps_per_cycle must be >= 1.")
        if not stream.train_type:
            raise ValueError(f"Async training stream {stream.name!r} train_type must not be empty.")
        stream_declared_train_types = set(stream.train_type)
        for target_policy_id, override_train_type in stream.train_type_by_target_policy_id.items():
            if not isinstance(target_policy_id, int):
                raise ValueError(
                    f"Async training stream {stream.name!r} has non-integer target_policy_id override "
                    f"{target_policy_id!r}."
                )
            if not override_train_type:
                raise ValueError(
                    f"Async training stream {stream.name!r} target_policy_id={target_policy_id} "
                    "train_type override must not be empty."
                )
            stream_declared_train_types.update(override_train_type)
        unknown_train_types = stream_declared_train_types - configured_train_types
        if unknown_train_types:
            raise ValueError(
                f"Async training stream {stream.name!r} train_type {sorted(unknown_train_types)} is not in "
                f"TrainConfig.train_type={sorted(configured_train_types)}."
            )
        stream_train_types.update(stream_declared_train_types)
    missing_train_types = configured_train_types - stream_train_types
    if missing_train_types:
        raise ValueError(f"Async training streams do not cover train_type entries: {sorted(missing_train_types)}.")
    return streams


def _episode_ids(source: dict[str, Any]) -> tuple[int, ...] | None:
    episode_ids = source.get("episode_ids")
    if episode_ids is None:
        return None
    return tuple(int(ep) for ep in episode_ids)


def _missing_feature_defaults(source: dict[str, Any]) -> dict[str, Any]:
    defaults = source.get("missing_feature_defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("Dataset view source missing_feature_defaults must be a mapping.")
    return dict(defaults)


def _repo_spec_defaults(source: dict[str, Any]) -> dict[str, Any]:
    filters = source.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError("Dataset view repo filters must be a mapping.")
    defaults = {key: value for key, value in filters.items() if isinstance(value, int | float | bool)}
    defaults.update(_missing_feature_defaults(source))
    return defaults


def _load_dataset_view_v2(view_path: Path, raw: dict[str, Any]) -> DatasetView:
    repos = raw.get("repos") or {}
    if not isinstance(repos, dict):
        raise ValueError(f"Dataset view {view_path} field 'repos' must be a mapping.")

    target_policy_id = raw.get("target_policy_id")
    target_policy_id = int(target_policy_id) if target_policy_id is not None else None
    repo_specs = []
    for repo_id, spec in repos.items():
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset view repo spec for {repo_id!r} must be a mapping.")
        repo_specs.append(
            _config.RepoSpec(
                repo_id=str(repo_id),
                episode_ids=_episode_ids(spec),
                missing_feature_defaults=_repo_spec_defaults(spec),
                advantage_snapshot=spec.get("advantage_snapshot"),
            )
        )
    if not repo_specs:
        raise ValueError(f"Dataset view {view_path} did not contain any repos.")
    return DatasetView(
        view_id=str(raw.get("dataset_view_id") or view_path.stem),
        repo_specs=tuple(repo_specs),
        sampler=None,
        target_policy_id=target_policy_id,
        raw=raw,
    )


def _load_dataset_view_v1(view_path: Path, raw: dict[str, Any]) -> DatasetView:
    pools = raw.get("pools") or {}
    repo_to_episodes: dict[str, set[int] | None] = {}
    repo_to_defaults: dict[str, dict[str, Any]] = {}
    sampler_groups = []
    target_policy_id = raw.get("target_policy_id")
    target_policy_id = int(target_policy_id) if target_policy_id is not None else None

    for pool_name, pool in pools.items():
        sources = pool.get("sources") or []
        for source in sources:
            repo_id = source.get("repo_id")
            if not repo_id:
                raise ValueError(f"Dataset view pool {pool_name!r} has a source without repo_id.")
            episodes = _episode_ids(source)
            if repo_id not in repo_to_episodes:
                repo_to_episodes[repo_id] = None if episodes is None else set(episodes)
            elif repo_to_episodes[repo_id] is not None and episodes is not None:
                repo_to_episodes[repo_id].update(episodes)
            elif episodes is None:
                repo_to_episodes[repo_id] = None
            source_defaults = _missing_feature_defaults(source)
            if source_defaults:
                existing_defaults = repo_to_defaults.setdefault(repo_id, {})
                conflicts = {
                    key: (existing_defaults[key], value)
                    for key, value in source_defaults.items()
                    if key in existing_defaults and existing_defaults[key] != value
                }
                if conflicts:
                    raise ValueError(f"Dataset view source defaults conflict for repo {repo_id!r}: {conflicts}")
                existing_defaults.update(source_defaults)

        filters = dict(pool.get("filters") or {})
        weight = pool.get("sampling_weight")
        if weight is not None and filters:
            sampler_groups.append(_config.SamplerGroupConfig(name=str(pool_name), weight=float(weight), where=filters))

    repo_specs = tuple(
        _config.RepoSpec(
            repo_id=repo_id,
            episode_ids=None if episodes is None else tuple(sorted(episodes)),
            missing_feature_defaults=repo_to_defaults.get(repo_id, {}),
        )
        for repo_id, episodes in repo_to_episodes.items()
    )
    if not repo_specs:
        raise ValueError(f"Dataset view {view_path} did not contain any repo sources.")

    sampler = None
    if sampler_groups:
        sampler = _config.SamplerConfig(
            type="weighted_column_groups",
            groups=tuple(sampler_groups),
            current_policy_id=target_policy_id,
            drop_empty_groups=bool(raw.get("drop_empty_sampler_groups", True)),
        )

    return DatasetView(
        view_id=str(raw.get("dataset_view_id") or view_path.stem),
        repo_specs=repo_specs,
        sampler=sampler,
        target_policy_id=target_policy_id,
        raw=raw,
    )


def load_dataset_view(path: str | Path) -> DatasetView:
    view_path = _resolve_path(path)
    raw = _read_json(view_path)
    schema_version = raw.get("schema_version")
    if schema_version == "dataset_view/v1":
        return _load_dataset_view_v1(view_path, raw)
    if schema_version == "dataset_view/v2":
        return _load_dataset_view_v2(view_path, raw)
    raise ValueError(f"Unsupported dataset view schema in {view_path}: {schema_version!r}")


def apply_dataset_view(data_config: _config.DataConfig, view: DatasetView) -> _config.DataConfig:
    sampler = view.sampler if view.sampler is not None else data_config.sampler
    if view.sampler is not None and data_config.sampler is not None:
        base_groups = {group.name: group for group in data_config.sampler.groups}
        sampler = dataclasses.replace(
            view.sampler,
            groups=tuple(
                dataclasses.replace(
                    group,
                    episode_replay=(
                        base_groups[group.name].episode_replay if group.name in base_groups else group.episode_replay
                    ),
                )
                for group in view.sampler.groups
            ),
        )
    if sampler is not None and view.target_policy_id is not None:
        sampler = dataclasses.replace(sampler, current_policy_id=view.target_policy_id)
    return dataclasses.replace(
        data_config,
        repo_id=None,
        repo_ids=None,
        repo_specs=view.repo_specs,
        sampler=sampler,
    )


class DatasetViewWatcher:
    def __init__(self, config: _config.DatasetViewConfig):
        self.config = config
        self._last_path: Path | None = None
        self._last_mtime_ns: int | None = None
        self._last_view_id: str | None = None

    def retry_last_view(self) -> None:
        self._last_path = None
        self._last_mtime_ns = None
        self._last_view_id = None

    def _resolve_target_view_path(
        self,
        *,
        latest_path: Path,
        latest_view_path: Path,
        latest_view: DatasetView,
        target_policy_id: int,
    ) -> Path | None:
        latest_target = latest_view.target_policy_id
        if latest_target is None:
            logging.warning(
                "Dataset view %s has no target_policy_id; cannot pin to target_policy_id=%d.",
                latest_view.view_id,
                target_policy_id,
            )
            return latest_view_path
        if latest_target == target_policy_id:
            return latest_view_path
        if latest_target < target_policy_id:
            logging.info(
                "Dataset view latest target_policy_id=%d is behind requested target_policy_id=%d at %s.",
                latest_target,
                target_policy_id,
                latest_path,
            )
            return None

        latest_suffix = f"{latest_target:06d}"
        target_suffix = f"{target_policy_id:06d}"
        if latest_view_path.stem.endswith(latest_suffix):
            target_path = latest_view_path.with_name(
                f"{latest_view_path.stem[: -len(latest_suffix)]}{target_suffix}{latest_view_path.suffix}"
            )
            if target_path.exists():
                logging.info(
                    "Dataset view latest target_policy_id=%d is ahead of requested target_policy_id=%d; "
                    "loading pinned view %s instead of %s.",
                    latest_target,
                    target_policy_id,
                    target_path,
                    latest_view_path,
                )
                return target_path
        logging.info(
            "Dataset view latest target_policy_id=%d is ahead of requested target_policy_id=%d, "
            "but no pinned sibling view was found for %s.",
            latest_target,
            target_policy_id,
            latest_view_path,
        )
        return None

    def poll(self, step: int, *, force: bool = False, target_policy_id: int | None = None) -> DatasetView | None:
        if not self.config.latest_path or not self.config.reload_on_change:
            return None
        if not force and step % max(1, self.config.poll_interval_steps) != 0:
            return None

        latest_path = _resolve_path(self.config.latest_path)
        if not latest_path.exists():
            logging.info("Dataset view latest path is not ready yet: %s", latest_path)
            return None

        latest = _read_json(latest_path)
        view_ref = latest.get("view_path") or latest.get("dataset_view_path") or latest.get("path")
        if view_ref is None and latest.get("schema_version") == "dataset_view/v1":
            view_path = latest_path
        elif view_ref is None:
            raise ValueError(f"Dataset latest registry {latest_path} does not contain a view path.")
        else:
            view_path = _resolve_path(view_ref, base_dir=latest_path.parent)

        if target_policy_id is not None:
            latest_view = load_dataset_view(view_path)
            pinned_view_path = self._resolve_target_view_path(
                latest_path=latest_path,
                latest_view_path=view_path,
                latest_view=latest_view,
                target_policy_id=int(target_policy_id),
            )
            if pinned_view_path is None:
                return None
            view_path = pinned_view_path

        stat = view_path.stat()
        if self._last_path == view_path and self._last_mtime_ns == stat.st_mtime_ns:
            return None

        view = load_dataset_view(view_path)
        if target_policy_id is not None and view.target_policy_id != int(target_policy_id):
            raise ValueError(
                f"Requested dataset view target_policy_id={target_policy_id}, but loaded {view.view_id} "
                f"with target_policy_id={view.target_policy_id} from {view_path}."
            )
        if view.view_id == self._last_view_id and self._last_mtime_ns == stat.st_mtime_ns:
            return None

        self._last_path = view_path
        self._last_mtime_ns = stat.st_mtime_ns
        self._last_view_id = view.view_id
        logging.info("Loaded dataset view %s from %s", view.view_id, view_path)
        return view
