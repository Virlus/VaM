"""Computation rules for :class:`DatasetProcessor`.

A *computation rule* declares the three-part contract from the refactor plan:

1. **Input columns** the rule reads from each episode parquet.
2. **The operation** that turns inputs into per-frame outputs (with an optional global
   reduction pass for things like an online-frame percentile threshold).
3. **Output columns** the rule writes to each episode parquet (and to
   ``meta/info.json`` features + ``meta/episodes_stats.jsonl``).

Concrete rules live below. They delegate math to
:mod:`preprocess_data.utils.advantage_utils` so the engine and the legacy
``advantage_annotation.py`` script stay bit-for-bit equivalent for the
``--advantage-source value_prediction`` path.

To register a new rule, subclass :class:`ComputationRule` and add it to :data:`RULES`.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from preprocess_data.utils.advantage_utils import calculate_episode_gae_advantages
from preprocess_data.utils.advantage_utils import calculate_episode_horizon_advantages
from preprocess_data.utils.dataset_processor import ColumnSpec
from preprocess_data.utils.dataset_processor import EpisodeColumns

logger = logging.getLogger("vam")


# --------------------------- base class --------------------------------------


class ComputationRule(abc.ABC):
    """Pluggable column-fill rule consumed by :meth:`DatasetProcessor.apply_rules`.

    Subclasses MUST implement :attr:`input_columns`, :attr:`output_columns`, and
    :meth:`compute_episode`. Rules that need a whole-dataset reduction (percentile
    threshold, normalization, etc.) set :attr:`needs_global_pass` to ``True`` and
    implement :meth:`reduce_global` + :meth:`finalize_episode`.

    A rule MUST be stateless across :meth:`compute_episode` calls except for state
    intentionally set by :meth:`reduce_global`. The engine never re-uses rule
    instances across :meth:`DatasetProcessor.apply_rules` invocations.
    """

    needs_global_pass: bool = False

    @property
    @abc.abstractmethod
    def input_columns(self) -> list[str]:
        """Columns this rule reads from each episode parquet."""

    @property
    @abc.abstractmethod
    def output_columns(self) -> list[ColumnSpec]:
        """Columns this rule writes back to each episode parquet."""

    @abc.abstractmethod
    def compute_episode(self, ep: EpisodeColumns) -> tuple[dict[str, np.ndarray], Any]:
        """Compute per-frame outputs for a single episode.

        Returns ``(outputs, payload)`` where:

        - ``outputs`` is a ``{output_column_name: 1-D array of length num_rows}`` dict.
          For one-pass rules this is the final value written to parquet. For two-pass
          rules this MAY be empty - the final outputs are produced by
          :meth:`finalize_episode` after :meth:`reduce_global` has run.
        - ``payload`` is an arbitrary per-episode object the rule will receive back in
          :meth:`reduce_global` and :meth:`finalize_episode`. Use ``None`` for one-pass
          rules.
        """

    def reduce_global(self, payloads: Sequence[tuple[int, Any]]) -> None:  # noqa: B027
        """Receive every per-episode payload to fit any global statistics.

        No-op default; one-pass rules (``needs_global_pass=False``) never have this
        called. Two-pass rules override it to fit cross-episode statistics such as a
        percentile threshold over online frames.
        """

    def finalize_episode(self, *, episode_index: int, num_rows: int, payload: Any) -> dict[str, np.ndarray]:
        """Turn a per-episode payload into final per-frame outputs."""
        raise NotImplementedError("Two-pass rules must implement finalize_episode")


# --------------------------- advantage rules ---------------------------------


_ADVANTAGE_FEATURE = {"dtype": "float32", "shape": [1], "names": ["advantage"]}


@dataclass
class _AdvantageEpisode:
    """Per-episode bundle passed from :meth:`compute_episode` to the global pass."""

    raw_advantage: np.ndarray  # shape (num_rows,) float32
    data_source: np.ndarray  # shape (num_rows,) float32


def _binarize_advantage_for_episode(
    raw: np.ndarray,
    data_source: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mirror ``advantage_annotation.write_advantages_to_parquet`` binarization.

    Offline frames (``data_source < 0.5``) are always positive (``1.0``); online
    frames (``data_source >= 0.5``) are ``1.0`` iff ``raw >= threshold``, else ``0.0``.
    """
    raw = np.asarray(raw, dtype=np.float32)
    ds = np.asarray(data_source, dtype=np.float32)
    return np.where(ds < 0.5, 1.0, np.where(raw >= threshold, 1.0, 0.0)).astype(np.float32)


def _percentile_threshold(payloads: Sequence[tuple[int, _AdvantageEpisode]], percentile: float) -> float:
    online: list[float] = []
    non_finite: list[tuple[int, int, float, int]] = []
    for episode_index, payload in payloads:
        if payload is None:
            continue
        for frame_idx, (a, ds) in enumerate(zip(payload.raw_advantage, payload.data_source, strict=True)):
            if not np.isfinite(a) and len(non_finite) < 10:
                non_finite.append((episode_index, frame_idx, float(a), int(ds)))
            if ds >= 0.5:
                online.append(float(a))
    if non_finite:
        raise ValueError(
            "Computed non-finite advantage values; refusing to write labels. "
            f"First examples (episode, frame, advantage, data_source): {non_finite}"
        )
    if not online:
        logger.warning("No online data found, setting advantage threshold to -inf")
        return float("-inf")
    return float(np.percentile(online, percentile))


class _AdvantageRuleBase(ComputationRule):
    """Shared scaffolding for advantage rules that binarize via an online percentile.

    Both GAE and value-average advantage need:

    - ``value_prediction`` for the current step.
    - ``data_source`` to identify offline frames (always positive label) vs online
      frames (binarized).
    - ``terminate`` (optional, defaults to all-zero) to mask bootstrapping.
    - ``next_obs_value_prediction`` if present (full-episode aligned); otherwise the
      rule shifts ``value_prediction`` by one frame, mirroring the legacy script.

    The rule runs in two passes: (1) computes raw advantages per episode, (2) the
    engine calls :meth:`reduce_global` with every episode's raw advantages so we can
    pick the online-frame percentile threshold, then :meth:`finalize_episode` writes
    the binary label per frame.
    """

    needs_global_pass = True

    def __init__(self, *, percentile: float, output_column: str = "advantage") -> None:
        self.percentile = float(percentile)
        self.output_column = output_column
        self._threshold: float | None = None

    @property
    def output_columns(self) -> list[ColumnSpec]:
        feature = dict(_ADVANTAGE_FEATURE)
        feature["names"] = [self.output_column]
        return [ColumnSpec(name=self.output_column, feature=feature)]

    @property
    def input_columns(self) -> list[str]:
        return ["value_prediction", "data_source", "reward"]

    # --- per-episode raw advantage ---

    @abc.abstractmethod
    def _raw_episode_advantage(self, ep: EpisodeColumns) -> np.ndarray:
        """Return the (num_rows,) raw advantage array for one episode."""

    @staticmethod
    def _next_obs_values(ep: EpisodeColumns, pred_values: np.ndarray) -> np.ndarray:
        """Pick the next-observation value series the legacy script uses."""
        if ep.has("next_obs_value_prediction"):
            return ep.get_1d("next_obs_value_prediction")
        # Fall back to a one-step shift of value_prediction, with the last step pinned
        # to itself so terminate handling masks the bootstrap.
        if len(pred_values) == 0:
            return pred_values
        shifted = np.empty_like(pred_values)
        shifted[:-1] = pred_values[1:]
        shifted[-1] = pred_values[-1]
        return shifted

    def compute_episode(self, ep: EpisodeColumns) -> tuple[dict[str, np.ndarray], _AdvantageEpisode]:
        data_source = ep.get_1d("data_source")
        raw = self._raw_episode_advantage(ep)
        return {}, _AdvantageEpisode(
            raw_advantage=np.asarray(raw, dtype=np.float32),
            data_source=np.asarray(data_source, dtype=np.float32),
        )

    def reduce_global(self, payloads: Sequence[tuple[int, _AdvantageEpisode]]) -> None:
        self._threshold = _percentile_threshold(payloads, self.percentile)
        logger.info("Advantage threshold (p%s online frames): %s", self.percentile, self._threshold)

    def finalize_episode(
        self, *, episode_index: int, num_rows: int, payload: _AdvantageEpisode
    ) -> dict[str, np.ndarray]:
        if self._threshold is None:
            raise RuntimeError("reduce_global must run before finalize_episode")
        if payload is None or len(payload.raw_advantage) != num_rows:
            raise ValueError(
                f"Episode {episode_index}: payload length {0 if payload is None else len(payload.raw_advantage)} "
                f"does not match parquet row count {num_rows}"
            )
        labels = _binarize_advantage_for_episode(payload.raw_advantage, payload.data_source, self._threshold)
        return {self.output_column: labels}


class GaeAdvantageRule(_AdvantageRuleBase):
    """Binary advantage label from GAE on an existing ``value_prediction`` column.

    Mirrors ``advantage_annotation.py --advantage-source value_prediction
    --advantage-method gae``. The math comes from
    :func:`preprocess_data.utils.advantage_utils.calculate_episode_gae_advantages`.
    """

    def __init__(
        self,
        *,
        percentile: float = 30.0,
        advantage_obs_horizon: int = 1,
        gamma: float = 1.0,
        gae_lambda: float = 0.9,
        action_horizon: int = 1,
        action_step_frames: int = 1,
        output_column: str = "advantage",
    ) -> None:
        super().__init__(percentile=percentile, output_column=output_column)
        self.advantage_obs_horizon = int(advantage_obs_horizon)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.action_horizon = int(action_horizon)
        self.action_step_frames = int(action_step_frames)

    @property
    def input_columns(self) -> list[str]:
        return ["value_prediction", "data_source", "reward"]

    def _raw_episode_advantage(self, ep: EpisodeColumns) -> np.ndarray:
        pred = ep.get_1d("value_prediction")
        data_source = ep.get_1d("data_source")
        # reward may be a per-frame scalar or an action-chunk vector; sum over the
        # chunk axis to match the legacy script's reward_sum semantics.
        if ep.has("reward"):
            reward_2d = ep.get_2d("reward")
            reward_sums = reward_2d.sum(axis=1) if reward_2d.ndim == 2 else reward_2d
        else:
            reward_sums = np.zeros_like(pred)
        terminates = ep.get_1d_optional("terminate", default=0.0)
        next_obs_values = self._next_obs_values(ep, pred)
        return calculate_episode_gae_advantages(
            all_pred_values=pred,
            all_next_obs_values=next_obs_values,
            all_data_sources=data_source,
            all_reward_sums=reward_sums,
            all_terminates=terminates,
            ep_starts=[0],
            ep_ends=[len(pred)],
            advantage_obs_horizon=self.advantage_obs_horizon,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            action_horizon=self.action_horizon,
            action_step_frames=self.action_step_frames,
        )


class HorizonAdvantageRule(_AdvantageRuleBase):
    """Binary advantage label from the value-average horizon estimator.

    Mirrors ``advantage_annotation.py --advantage-source value_prediction
    --advantage-method value_avg``.
    """

    def __init__(
        self,
        *,
        percentile: float = 30.0,
        advantage_obs_horizon: int = 1,
        output_column: str = "advantage",
    ) -> None:
        super().__init__(percentile=percentile, output_column=output_column)
        self.advantage_obs_horizon = int(advantage_obs_horizon)

    @property
    def input_columns(self) -> list[str]:
        return ["value_prediction", "data_source"]

    def _raw_episode_advantage(self, ep: EpisodeColumns) -> np.ndarray:
        pred = ep.get_1d("value_prediction")
        data_source = ep.get_1d("data_source")
        terminates = ep.get_1d_optional("terminate", default=0.0)
        next_obs_values = self._next_obs_values(ep, pred)
        return calculate_episode_horizon_advantages(
            all_pred_values=pred,
            all_next_obs_values=next_obs_values,
            all_data_sources=data_source,
            all_terminates=terminates,
            ep_starts=[0],
            ep_ends=[len(pred)],
            advantage_obs_horizon=self.advantage_obs_horizon,
        )


# --------------------------- progress / copy rules ----------------------------


class CopyColumnRule(ComputationRule):
    """Copy one column to another, optionally wrapping scalars as singleton lists.

    Port of :func:`run_copy_column_operation`. The target feature spec is provided
    explicitly so the engine writes a consistent ``info.json`` entry; if you want
    to inherit the source feature use
    :meth:`DatasetProcessor.parquet_files` + a custom call site instead.
    """

    needs_global_pass = False

    def __init__(
        self,
        *,
        source_column: str,
        target_column: str,
        target_feature: dict[str, Any] | None = None,
    ) -> None:
        if not source_column or not target_column:
            raise ValueError("CopyColumnRule requires non-empty source_column and target_column")
        if source_column == target_column:
            raise ValueError("source_column and target_column must differ")
        self.source_column = source_column
        self.target_column = target_column
        self.target_feature = target_feature or {
            "dtype": "float32",
            "shape": [1],
            "names": [target_column],
        }

    @property
    def input_columns(self) -> list[str]:
        return [self.source_column]

    @property
    def output_columns(self) -> list[ColumnSpec]:
        return [ColumnSpec(name=self.target_column, feature=dict(self.target_feature))]

    def compute_episode(self, ep: EpisodeColumns) -> tuple[dict[str, np.ndarray], Any]:
        values = ep.get_1d(self.source_column)
        return {self.target_column: values.astype(np.float32, copy=False)}, None


class AugmentRewardValueSuccessRule(ComputationRule):
    """Fill ``success / reward / value / data_source / control_flag`` for an episode.

    Port of :func:`run_augment_reward_value_success_operation`. Useful for back-filling
    offline-only demos that lack online-style metadata columns.

    Modes for ``value``:
      - ``"fractional_progress"`` (default): ``arange(N) / N``, matches the legacy op.
      - ``"linear_progress"`` / ``"linear_0_1"``: ``linspace(0, 1, N, endpoint=value_endpoint)``.
      - ``"terminal_reward_to_go"``: ``full(N, reward_terminal_value)``.
    """

    needs_global_pass = False

    def __init__(
        self,
        *,
        success_value: float = 1.0,
        reward_terminal_value: float = 1.0,
        reward_default_value: float = 0.0,
        value_mode: str = "fractional_progress",
        value_endpoint: bool = False,
        data_source_value: float = 0.0,
        control_flag_value: float = -1.0,
        success_column: str = "success",
        reward_column: str = "reward",
        value_column: str = "value",
        data_source_column: str = "data_source",
        control_flag_column: str = "control_flag",
    ) -> None:
        self.success_value = float(success_value)
        self.reward_terminal_value = float(reward_terminal_value)
        self.reward_default_value = float(reward_default_value)
        self.value_mode = str(value_mode)
        self.value_endpoint = bool(value_endpoint)
        self.data_source_value = float(data_source_value)
        self.control_flag_value = float(control_flag_value)
        self.success_column = success_column
        self.reward_column = reward_column
        self.value_column = value_column
        self.data_source_column = data_source_column
        self.control_flag_column = control_flag_column

    @property
    def input_columns(self) -> list[str]:
        return []

    @property
    def output_columns(self) -> list[ColumnSpec]:
        def _spec(name: str, sensor_name: str | None = None) -> ColumnSpec:
            names = [sensor_name or name]
            return ColumnSpec(name=name, feature={"dtype": "float32", "shape": [1], "names": names})

        return [
            _spec(self.success_column),
            _spec(self.reward_column),
            _spec(self.value_column),
            _spec(self.data_source_column, "source"),
            _spec(self.control_flag_column, "flag"),
        ]

    def compute_episode(self, ep: EpisodeColumns) -> tuple[dict[str, np.ndarray], Any]:
        n = ep.num_rows
        if n == 0:
            return {}, None
        success = np.full(n, self.success_value, dtype=np.float32)
        reward = np.full(n, self.reward_default_value, dtype=np.float32)
        reward[-1] = self.reward_terminal_value
        if self.value_mode == "fractional_progress":
            value = np.arange(0, n, dtype=np.float32) / n
        elif self.value_mode in {"linear_progress", "linear_0_1"}:
            value = np.linspace(0.0, 1.0, n, dtype=np.float32, endpoint=self.value_endpoint)
        elif self.value_mode == "terminal_reward_to_go":
            value = np.full(n, self.reward_terminal_value, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported value_mode: {self.value_mode}")
        data_source = np.full(n, self.data_source_value, dtype=np.float32)
        control_flag = np.full(n, self.control_flag_value, dtype=np.float32)
        return {
            self.success_column: success,
            self.reward_column: reward,
            self.value_column: value,
            self.data_source_column: data_source,
            self.control_flag_column: control_flag,
        }, None


# --------------------------- registry ----------------------------------------


RULES: dict[str, type[ComputationRule]] = {
    "gae_advantage": GaeAdvantageRule,
    "horizon_advantage": HorizonAdvantageRule,
    "copy_column": CopyColumnRule,
    "augment_reward_value_success": AugmentRewardValueSuccessRule,
}


def build_rule(spec: dict[str, Any]) -> ComputationRule:
    """Build a :class:`ComputationRule` from a YAML-style mapping.

    Expected shape::

        {"type": "gae_advantage", "params": {"percentile": 70.0, ...}}

    Unknown ``type`` values raise :class:`KeyError`. ``params`` defaults to ``{}``.
    """
    if "type" not in spec:
        raise ValueError(f"rule spec missing 'type': {spec}")
    rule_type = str(spec["type"])
    if rule_type not in RULES:
        raise KeyError(f"Unknown rule type {rule_type!r}. Available: {sorted(RULES)}")
    params = spec.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError(f"rule 'params' must be a mapping, got {type(params).__name__}")
    return RULES[rule_type](**params)
