import json
from unittest.mock import patch

import pytest

from consumer.policy_connector import PolicyConnector
from utils.online_training import (
    load_policy_rollout_state,
    policy_rollout_state_path_from_config,
    write_policy_rollout_state,
)


def _connector_with_state(state: dict) -> PolicyConnector:
    connector = object.__new__(PolicyConnector)
    connector.summary_shm = None
    connector.online_training_enabled = True
    connector.max_current_policy_rollouts = 20
    connector._persisted_policy_rollout_state = state
    return connector


def test_policy_rollout_state_round_trip_records_checkpoint_and_count(tmp_path) -> None:
    state_path = policy_rollout_state_path_from_config({}, tmp_path)

    write_policy_rollout_state(
        state_path,
        {
            "policy_id": 9,
            "checkpoint_id": "merged_000009_step_00003600",
            "checkpoint_index": 3600,
            "train_step": 3600,
            "checkpoint_dir": str(tmp_path / "merged_000009_step_00003600"),
            "current_policy_episode_cnt": 10,
            "max_current_policy_rollouts": 20,
        },
    )

    raw_state = json.loads(state_path.read_text())
    loaded_state = load_policy_rollout_state(state_path)

    assert state_path == tmp_path / "policy_rollout_state.json"
    assert raw_state["schema_version"] == "policy_rollout_state/v1"
    assert raw_state["policy_id"] == 9
    assert raw_state["train_step"] == 3600
    assert raw_state["current_policy_episode_cnt"] == 10
    assert raw_state["max_current_policy_rollouts"] == 20
    assert loaded_state["checkpoint_id"] == "merged_000009_step_00003600"
    assert loaded_state["current_policy_episode_cnt"] == 10


def test_restart_restores_count_for_same_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "merged_000009_step_00003600"
    connector = _connector_with_state(
        {
            "checkpoint_id": "merged_000009_step_00003600",
            "checkpoint_dir": str(checkpoint_dir),
            "current_policy_episode_cnt": 10,
        }
    )

    restored_count = connector._restored_policy_rollout_count(
        {
            "checkpoint_id": "merged_000009_step_00003600",
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    assert restored_count == 10


def test_new_checkpoint_starts_with_zero_rollouts(tmp_path) -> None:
    checkpoint_dir = tmp_path / "current"
    connector = _connector_with_state(
        {
            "checkpoint_id": "merged_000009_step_00003600",
            "train_step": 3600,
            "checkpoint_dir": str(checkpoint_dir),
            "current_policy_episode_cnt": 10,
        }
    )

    restored_count = connector._restored_policy_rollout_count(
        {
            "checkpoint_id": "merged_000010_step_00004000",
            "train_step": 4000,
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    assert restored_count == 0


def test_legacy_current_policy_count_is_migrated_when_state_is_missing() -> None:
    connector = _connector_with_state({})

    restored_count = connector._restored_policy_rollout_count(
        {
            "checkpoint_id": "merged_000009_step_00003600",
            "current_policy_episode_cnt": 10,
        }
    )

    assert restored_count == 10


def test_success_and_failure_persist_count_but_discard_does_not(tmp_path) -> None:
    state_path = tmp_path / "policy_rollout_state.json"
    checkpoint_dir = tmp_path / "merged_000009_step_00003600"
    connector = _connector_with_state({})
    connector.policy_rollout_state_path = state_path
    connector.current_policy_info = {
        "policy_id": 9,
        "checkpoint_id": "merged_000009_step_00003600",
        "checkpoint_index": 3600,
        "train_step": 3600,
        "checkpoint_dir": str(checkpoint_dir),
        "current_policy_episode_cnt": 10,
    }
    connector._online_policy_log = lambda *args, **kwargs: None

    connector._increment_current_policy_episode_count("success")
    connector._increment_current_policy_episode_count("failure")
    connector._increment_current_policy_episode_count("discard")

    state = load_policy_rollout_state(state_path)
    assert state["current_policy_episode_cnt"] == 12
    assert connector._current_policy_episode_count() == 12


def test_current_policy_registry_excludes_rollout_count() -> None:
    registry_info = PolicyConnector._policy_registry_info(
        {
            "checkpoint_id": "merged_000009_step_00003600",
            "current_policy_episode_cnt": 10,
        }
    )

    assert registry_info == {"checkpoint_id": "merged_000009_step_00003600"}


def test_rollout_limit_wait_disables_inference_until_new_checkpoint() -> None:
    connector = _connector_with_state({})
    connector.current_policy_info = {
        "checkpoint_id": "merged_000009_step_00003600",
        "current_policy_episode_cnt": 20,
    }
    connector.policy_rollout_limit_wait_interval = 0
    connector.inference_enabled = True
    connector.running = True
    connector._online_policy_log = lambda *args, **kwargs: None

    scan_inference_states = []

    def maybe_switch() -> bool:
        scan_inference_states.append(connector.inference_enabled)
        return len(scan_inference_states) == 2

    connector._maybe_switch_policy_at_episode_boundary = maybe_switch
    connector._signal_policy_runner_reset = lambda: None

    with patch("consumer.policy_connector.time.sleep", return_value=None):
        connector._maybe_switch_or_wait_for_policy_budget()

    assert scan_inference_states == [True, False]
    assert connector.inference_enabled is True


def test_run_checks_rollout_budget_before_starting_saver_or_connecting() -> None:
    connector = _connector_with_state({})
    events = []
    connector._maybe_switch_or_wait_for_policy_budget = lambda: events.append("budget_gate")
    connector._should_start_infer_rollout = lambda: events.append("saver_check") or False
    connector.connect = lambda: events.append("connect") or False

    connector.run()

    assert events == ["budget_gate", "saver_check", "connect"]
    assert connector.running is False


@pytest.mark.parametrize("result,expected_count", [("success", 1), ("failure", 1), ("discard", 0)])
def test_lerobot_transition_preserves_stop_before_halt_order(result, expected_count) -> None:
    """Use only mocks: restoring the recorder must not change robot sequencing."""
    connector = object.__new__(PolicyConnector)
    connector.summary_shm = None
    connector._episode_in_transition = False
    connector._episode_count = 0
    connector.episode_home_wait_time = 0
    connector.episode_auto_restart = True
    connector.infer_rollout_config = {"format": "lerobot"}
    connector._read_robot_home_index = lambda: 0
    events = []
    names = (
        "_write_episode_result",
        "_signal_stop_recording",
        "_increment_current_policy_episode_count",
        "_signal_and_wait_for_robot_halt",
        "_signal_policy_runner_reset",
        "_signal_teleoperator_reset",
        "_signal_robot_home",
        "_maybe_switch_or_wait_for_policy_budget",
        "_signal_start_recording",
    )
    for name in names:
        setattr(connector, name, lambda *args, name=name, **kwargs: events.append(name))
    with patch("consumer.policy_connector.time.sleep", return_value=None):
        connector._handle_episode_transition(result)
    assert events == list(names)
    assert connector._episode_count == expected_count
    assert connector.inference_enabled
