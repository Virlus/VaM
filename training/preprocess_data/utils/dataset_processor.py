"""General LeRobot dataset column processor.

This module consolidates the column read/compute/write/stats/info.json boilerplate
previously duplicated across ``lerobot_dataset_ops.py``, ``advantage_annotation.py``,
``value_annotation.py`` into a single engine.

Public surface:

- :class:`ColumnSpec` describes a column to write (name + LeRobot ``info.json`` feature).
- :class:`EpisodeColumns` is a read-only per-episode view handed to computation rules.
- :class:`DatasetProcessor` opens a LeRobot dataset and exposes:

  * :meth:`DatasetProcessor.remove_columns` - drop columns from every episode parquet and
    from ``meta/info.json``.
  * :meth:`DatasetProcessor.set_column` - write a single per-frame column from a
    pre-computed ``{episode_index: 1D array}`` dict (model-inference path).
  * :meth:`DatasetProcessor.apply_rules` - run :class:`ComputationRule` instances; supports
    rules that need a whole-dataset global reduction pass (e.g. GAE + online-frame
    percentile threshold).

The engine performs atomic per-episode writes (``tmp.replace``), patches
``meta/info.json`` features once at the end, and rewrites the matching entries in
``meta/episodes_stats.jsonl`` so downstream LeRobot tooling sees consistent stats.

Math for advantage / value computations lives in :mod:`preprocess_data.utils.advantage_utils`
and is reused by :mod:`preprocess_data.utils.computation_rules` - this module does not
duplicate any formulas.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from preprocess_data.utils.lerobot_ops_common import ensure_annotation_column_writable
from preprocess_data.utils.lerobot_ops_common import ensure_dataset_exists
from preprocess_data.utils.lerobot_ops_common import find_episode_parquet_files
from preprocess_data.utils.lerobot_ops_common import read_parquet_allow_duplicate_columns
from preprocess_data.utils.lerobot_ops_common import resolve_dataset_root
from preprocess_data.utils.lerobot_ops_common import write_info_json_features

if TYPE_CHECKING:
    from preprocess_data.utils.computation_rules import ComputationRule

logger = logging.getLogger("vam")


# ----------------------------- public dataclasses ----------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a column the engine will write.

    ``feature`` is the entry written to ``meta/info.json`` under ``features[name]``.
    For 1-D float labels (advantage, value_prediction, ...) use shape ``[1]`` and
    ``dtype="float32"``; the engine wraps values as ``pa.list_(pa.float32())``.
    String columns are detected from ``feature["dtype"] == "string"``.
    """

    name: str
    feature: dict[str, Any]


# ------------------------------ internal helpers -----------------------------


def _float_singleton_column(values: np.ndarray) -> pa.Array:
    """Wrap a 1-D numeric array as a parquet ``list_(float32)`` column."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return pa.array([[float(value)] for value in values], type=pa.list_(pa.float32()))


def _string_column(values: Iterable[Any]) -> pa.Array:
    return pa.array([str(v) if v is not None else "" for v in values], type=pa.string())


def _replace_or_append_column(table: pa.Table, name: str, column: pa.Array) -> pa.Table:
    """Drop *all* occurrences of ``name`` from ``table`` and append ``column`` last.

    LeRobot parquet files occasionally contain duplicate column names from prior
    re-annotation runs; selecting only the indices whose name is not ``name``
    silently de-duplicates while we're touching the file.
    """
    if name in table.column_names:
        keep = [i for i, n in enumerate(table.column_names) if n != name]
        table = table.select(keep)
    return table.append_column(name, column)


def _stats_from_array(values: np.ndarray) -> dict[str, list[Any]]:
    """LeRobot ``episodes_stats.jsonl`` entry for a 1-D or N-D column."""
    values = np.asarray(values)
    keepdims = values.ndim == 1
    return {
        "min": np.min(values, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(values, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(values, axis=0, keepdims=keepdims).tolist(),
        "std": np.std(values, axis=0, keepdims=keepdims).tolist(),
        "count": [int(values.shape[0])],
    }


def _stats_entry_float_singleton(values: np.ndarray) -> dict[str, list[Any]]:
    return _stats_from_array(np.asarray(values, dtype=np.float32).reshape(-1, 1))


def _column_to_1d_float(table: pa.Table, key: str, *, episode_index: int) -> np.ndarray:
    """Read ``key`` from ``table`` as a 1-D float64 array (singleton-list per row).

    Mirrors the helper that lived in ``advantage_annotation.py`` so the engine and the
    legacy advantage script agree on column extraction semantics.
    """
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} not found in episode {episode_index}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in table.column(key).to_pylist()]
    if not values:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([value[0] if len(value) else np.nan for value in values], dtype=np.float64)


def _column_to_2d_float(table: pa.Table, key: str, *, episode_index: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Column {key!r} not found in episode {episode_index}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in table.column(key).to_pylist()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = len(values[0])
    if any(len(value) != width for value in values):
        raise ValueError(f"Column {key!r} in episode {episode_index} has variable-length rows")
    return np.stack(values, axis=0)


def _episode_index_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _update_episodes_stats(
    dataset_root: Path,
    stats_by_episode: dict[int, dict[str, dict[str, list[Any]]]],
) -> None:
    """Patch ``meta/episodes_stats.jsonl`` with new per-column stats.

    ``stats_by_episode[episode_index][column_name]`` must already be the formatted
    ``{min, max, mean, std, count}`` dict produced by :func:`_stats_from_array`.
    """
    episode_stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not episode_stats_path.exists():
        logger.warning("meta/episodes_stats.jsonl not found at %s; skipping stats update", episode_stats_path)
        return
    entries = _read_jsonl(episode_stats_path)
    for entry in entries:
        episode_index = int(entry["episode_index"])
        if episode_index not in stats_by_episode:
            continue
        stats = entry.setdefault("stats", {})
        stats.update(stats_by_episode[episode_index])
    _write_jsonl(episode_stats_path, entries)


# ------------------------------ episode view ---------------------------------


@dataclass
class EpisodeColumns:
    """Read-only per-episode column view passed to :class:`ComputationRule`.

    Rules call :meth:`get_1d` / :meth:`get_2d` / :meth:`has` rather than touching the
    raw :class:`pyarrow.Table` directly so that the engine can swap representations
    later (e.g. lazy column reads) without breaking rule implementations.
    """

    episode_index: int
    num_rows: int
    parquet_path: Path
    _table: pa.Table

    def has(self, column: str) -> bool:
        return column in self._table.column_names

    def get_1d(self, column: str) -> np.ndarray:
        return _column_to_1d_float(self._table, column, episode_index=self.episode_index)

    def get_2d(self, column: str) -> np.ndarray:
        return _column_to_2d_float(self._table, column, episode_index=self.episode_index)

    def get_1d_optional(self, column: str, *, default: float = 0.0) -> np.ndarray:
        if not self.has(column):
            return np.full(self.num_rows, default, dtype=np.float64)
        values = self.get_1d(column)
        if len(values) != self.num_rows:
            raise ValueError(
                f"Episode {self.episode_index} column {column!r}: got {len(values)} rows, expected {self.num_rows}"
            )
        return values


# ------------------------------ result containers ---------------------------


@dataclass
class _PendingEpisodeWrite:
    parquet_path: Path
    episode_index: int
    columns: dict[str, np.ndarray] = field(default_factory=dict)  # output_name -> 1D / N-D values


# ------------------------------- processor ----------------------------------


class DatasetProcessor:
    """Open a LeRobot dataset for column-level read/modify operations.

    Construct with either an explicit ``dataset_root`` or a ``repo_id`` (resolved
    against ``HF_LEROBOT_HOME`` or ``output_path`` via
    :func:`preprocess_data.utils.lerobot_ops_common.resolve_dataset_root`).

    All methods on this class are safe to call repeatedly on the same dataset; each
    call re-discovers the parquet files under ``data/`` so freshly-converted episodes
    are picked up by the next call.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path | None = None,
        repo_id: str | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.dataset_root: Path = resolve_dataset_root(
            dataset_root=dataset_root, repo_id=repo_id, output_path=output_path
        )
        ensure_dataset_exists(self.dataset_root)

    # ---- discovery ----

    def parquet_files(self) -> list[Path]:
        return find_episode_parquet_files(self.dataset_root)

    # ---- remove ----

    def remove_columns(self, names: list[str], *, dry_run: bool = False) -> int:
        """Drop ``names`` from every episode parquet and from ``meta/info.json``.

        Returns the number of parquet files that were rewritten (a file is counted
        once per call even if it contained more than one of the requested columns).
        """
        if not names:
            return 0
        parquet_files = self.parquet_files()
        modified = 0
        for parquet_path in tqdm(parquet_files, desc=f"Removing columns {names}"):
            table = read_parquet_allow_duplicate_columns(parquet_path)
            keep_idxs = [i for i, n in enumerate(table.column_names) if n not in names]
            if len(keep_idxs) == len(table.column_names):
                continue
            if dry_run:
                modified += 1
                continue
            table = table.select(keep_idxs)
            tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
            pq.write_table(table, tmp_path)
            tmp_path.replace(parquet_path)
            modified += 1
        if not dry_run:
            write_info_json_features(self.dataset_root, remove=list(names))
        logger.info(
            "Removed columns %s from %d/%d parquet files (dry_run=%s)",
            names,
            modified,
            len(parquet_files),
            dry_run,
        )
        return modified

    # ---- set a single pre-computed column ----

    def set_column(
        self,
        spec: ColumnSpec,
        values_by_episode: Mapping[int, np.ndarray],
        *,
        overwrite: bool = False,
        update_stats: bool = True,
    ) -> int:
        """Write a single column from pre-computed per-episode values.

        Used by the model-inference annotators (``value_annotation.py`` and
        the inference branch of ``advantage_annotation.py``)
        where the values come from outside the engine.
        """
        ensure_annotation_column_writable(self.dataset_root, spec.name, overwrite=overwrite)
        parquet_files = self.parquet_files()
        stats_by_episode: dict[int, dict[str, dict[str, list[Any]]]] = {}
        modified = 0
        is_string = str(spec.feature.get("dtype", "")).lower() == "string"
        for parquet_path in tqdm(parquet_files, desc=f"Setting {spec.name}"):
            episode_index = _episode_index_from_path(parquet_path)
            if episode_index not in values_by_episode:
                continue
            values = np.asarray(values_by_episode[episode_index])
            table = read_parquet_allow_duplicate_columns(parquet_path)
            if table.num_rows != len(values):
                raise ValueError(
                    f"Episode {episode_index}: parquet has {table.num_rows} rows but got {len(values)} values"
                )
            column = _string_column(values.tolist()) if is_string else _float_singleton_column(values)
            table = _replace_or_append_column(table, spec.name, column)
            tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
            pq.write_table(table, tmp_path)
            tmp_path.replace(parquet_path)
            if update_stats and not is_string:
                stats_by_episode[episode_index] = {spec.name: _stats_entry_float_singleton(values)}
            modified += 1
        write_info_json_features(self.dataset_root, add_or_update={spec.name: dict(spec.feature)})
        if stats_by_episode:
            _update_episodes_stats(self.dataset_root, stats_by_episode)
        logger.info("Wrote column %s to %d/%d parquet files", spec.name, modified, len(parquet_files))
        return modified

    # ---- apply computation rules ----

    def apply_rules(
        self,
        rules: list[ComputationRule],
        *,
        overwrite: bool = False,
        update_stats: bool = True,
    ) -> None:
        """Run computation rules against the dataset.

        For each rule the engine:

        1. Resolves ``rule.output_columns`` against ``ensure_annotation_column_writable``
           (unless ``overwrite`` is set) so we fail fast before any writes.
        2. Pass 1 (per-episode): reads ``rule.input_columns``, calls
           ``rule.compute_episode`` to produce either the final per-frame outputs or an
           intermediate payload (the rule decides).
        3. Pass 2 (global, optional): if ``rule.needs_global_pass`` is ``True``, calls
           ``rule.reduce_global`` once with every intermediate payload, then
           ``rule.finalize_episode`` for each episode to turn payloads into final outputs.
        4. Writes the final per-frame outputs to every episode parquet (atomic
           ``tmp.replace``), patches ``meta/info.json`` features, and updates
           ``meta/episodes_stats.jsonl`` for numeric outputs.

        All rules in the list are committed in a single info.json + episode_stats
        write at the end of each rule's execution. Rules are processed sequentially -
        a later rule can read columns produced by an earlier rule in the same call.
        """
        # Local import keeps ComputationRule definition independent of this engine.
        from preprocess_data.utils.computation_rules import ComputationRule  # noqa: F401

        for rule in rules:
            self._apply_single_rule(rule, overwrite=overwrite, update_stats=update_stats)

    def _apply_single_rule(
        self,
        rule: ComputationRule,
        *,
        overwrite: bool,
        update_stats: bool,
    ) -> None:
        for spec in rule.output_columns:
            ensure_annotation_column_writable(self.dataset_root, spec.name, overwrite=overwrite)

        parquet_files = self.parquet_files()
        if not parquet_files:
            logger.warning("No parquet files under %s; rule %s no-op", self.dataset_root, type(rule).__name__)
            return

        # ---- pass 1: per-episode compute ----
        intermediates: list[tuple[Path, int, dict[str, np.ndarray], Any]] = []
        for parquet_path in tqdm(parquet_files, desc=f"{type(rule).__name__} pass1"):
            table = read_parquet_allow_duplicate_columns(parquet_path)
            episode_index = _episode_index_from_path(parquet_path)
            view = EpisodeColumns(
                episode_index=episode_index,
                num_rows=table.num_rows,
                parquet_path=parquet_path,
                _table=table,
            )
            outputs, payload = rule.compute_episode(view)
            intermediates.append((parquet_path, episode_index, outputs, payload))

        # ---- pass 2 (optional): global reduce + per-episode finalize ----
        if rule.needs_global_pass:
            payloads = [
                (episode_index, payload) for _, episode_index, _, payload in intermediates if payload is not None
            ]
            rule.reduce_global(payloads)
            finalized: list[_PendingEpisodeWrite] = []
            for parquet_path, episode_index, _outputs, payload in intermediates:
                num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
                outputs = rule.finalize_episode(episode_index=episode_index, num_rows=num_rows, payload=payload)
                pending = _PendingEpisodeWrite(parquet_path=parquet_path, episode_index=episode_index)
                pending.columns.update(outputs)
                finalized.append(pending)
        else:
            finalized = []
            for parquet_path, episode_index, outputs, _payload in intermediates:
                pending = _PendingEpisodeWrite(parquet_path=parquet_path, episode_index=episode_index)
                pending.columns.update(outputs)
                finalized.append(pending)

        # ---- pass 3: atomic writes ----
        spec_by_name = {spec.name: spec for spec in rule.output_columns}
        stats_by_episode: dict[int, dict[str, dict[str, list[Any]]]] = {}
        modified = 0
        for pending in tqdm(finalized, desc=f"{type(rule).__name__} write"):
            if not pending.columns:
                continue
            table = read_parquet_allow_duplicate_columns(pending.parquet_path)
            episode_stats: dict[str, dict[str, list[Any]]] = {}
            for name, values in pending.columns.items():
                spec = spec_by_name.get(name)
                if spec is None:
                    raise KeyError(
                        f"Rule {type(rule).__name__} produced unexpected column {name!r}; "
                        f"declared outputs: {list(spec_by_name)}"
                    )
                is_string = str(spec.feature.get("dtype", "")).lower() == "string"
                values_arr = np.asarray(values)
                if not is_string and table.num_rows != len(values_arr):
                    raise ValueError(
                        f"Episode {pending.episode_index} column {name!r}: parquet has {table.num_rows} rows but "
                        f"rule produced {len(values_arr)} values"
                    )
                column = _string_column(values_arr.tolist()) if is_string else _float_singleton_column(values_arr)
                table = _replace_or_append_column(table, name, column)
                if update_stats and not is_string:
                    episode_stats[name] = _stats_entry_float_singleton(values_arr)
            tmp_path = pending.parquet_path.with_suffix(pending.parquet_path.suffix + ".tmp")
            pq.write_table(table, tmp_path)
            tmp_path.replace(pending.parquet_path)
            if episode_stats:
                stats_by_episode[pending.episode_index] = episode_stats
            modified += 1

        write_info_json_features(
            self.dataset_root, add_or_update={spec.name: dict(spec.feature) for spec in rule.output_columns}
        )
        if stats_by_episode:
            _update_episodes_stats(self.dataset_root, stats_by_episode)
        logger.info(
            "Rule %s wrote columns %s to %d/%d parquet files",
            type(rule).__name__,
            [spec.name for spec in rule.output_columns],
            modified,
            len(parquet_files),
        )
