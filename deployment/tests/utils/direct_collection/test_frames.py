"""Reward, missing-source and intervention boundaries before writing any videos."""

import numpy as np
import pytest

from utils.lerobot_direct import OnlineFrames, validate_config
from utils.lerobot_frames import align


def build(config, policy, samples, result="success"):
    data, times, start, end = samples
    return OnlineFrames(data, times, config, policy, result, start, end)


@pytest.mark.parametrize("result", ["success", "failure"])
def test_chunks_reward_and_real_values(config, policy, samples, result):
    frames = build(config, policy, samples, result)
    assert frames.prefixes == [
        {"end_index": 20, "terminal_reason": "policy_to_human"},
        {"end_index": 60, "terminal_reason": "episode_end"},
    ]
    np.testing.assert_array_equal(frames.actions[20:40], 2)
    np.testing.assert_array_equal(frames.actions[:20], 1)
    rows = list(frames.frames("action", frames.end))
    assert len(rows) == 19
    for position, row in enumerate(rows):
        index = row["source_action_index"].item()
        assert row["value_prediction"].item() == pytest.approx(0.1 + 0.01 * index, abs=0.011)
        expected = (
            np.dot(row["reward"], 0.999 ** np.arange(10))
            + 0.999**10
            * (1 - row["terminate"].item())
            * rows[min(position + 3, len(rows) - 1)]["value_prediction"].item()
            - row["value_prediction"].item()
        )
        assert row["advantage_raw"].item() == pytest.approx(expected)
        assert row["success"].item() == (result == "success")
        assert "advantage" not in row
    assert rows[-1]["terminate"].item() == 1
    assert rows[-1]["reward"].sum() == (result == "success")
    assert rows[-1]["reward"][1:].sum() == 0
    np.testing.assert_array_equal(rows[-1]["actions"], 1)
    prefix = list(frames.frames("value", 20))
    assert all(row["reward"].sum() == 0 for row in prefix)
    assert all(row["success"].item() == (result == "success") for row in prefix)
    assert prefix[-1]["control_flag"].item() == 1
    assert prefix[-1]["actions"][0, 0] == 2
    # Legacy next observations are built from the full trajectory before splitting.
    assert any(
        audit["sources"]["next_state"]["timestamp_ns"] > frames.action_times[20] for audit in frames.audit["value"]
    )
    full = list(frames.frames("value", 60, 1))
    assert all(row["source_action_index"].item() % 10 == 0 for row in full[:-1])
    assert full[-1]["source_action_index"].item() == 60


def test_excluded_gap_is_removed_without_reward_leak(config, policy, samples):
    samples[0]["control_flag"][10:20] = 0.5
    frames = build(config, policy, samples)
    assert len(frames.actions) == 51
    assert frames.intervals == [[10_100_000_000, 10_200_000_000]]
    assert set(frames.flags) == {0, 1}
    rows = list(frames.frames("action", frames.end))
    assert rows[-1]["reward"].sum() == 1
    samples[0]["control_flag"][-1] = 0.5
    tail = build(config, policy, samples)
    assert list(tail.frames("action", tail.end))[-1]["reward"].sum() == 1


@pytest.mark.parametrize("problem", ["value", "clock", "flag", "identity"])
def test_invalid_sources_rejected(config, policy, samples, problem):
    data, times, _, _ = samples
    if problem == "value":
        data["value_prediction"][:] = np.nan
    elif problem == "clock":
        times["state"][2] = times["state"][1]
    elif problem == "flag":
        data["control_flag"][0] = 0.7
    else:
        policy["policy_id"] = 2**24 + 1
    with pytest.raises(ValueError):
        frames = build(config, policy, samples)
        list(frames.frames("action", frames.end))


def test_nearest_and_config(config):
    assert align(np.array([10.0, 20.0]), 15.0, 10.0) == 0
    assert align(np.array([10.0, 20.0]), 50.0, 10.0) == -1
    assert align(np.array([]), 15.0, 10.0) == -1
    config["advantage"]["mode"] = "fixed"
    with pytest.raises(ValueError, match="cloud_p70"):
        validate_config(config)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, "3"])
def test_invalid_action_stride(config, stride):
    """Reject non-positive/non-integer strides before recording starts."""
    config["action_stride"] = stride
    with pytest.raises(ValueError, match="action_stride"):
        validate_config(config)


@pytest.mark.parametrize("stride", [3, 1000])
def test_action_stride_keeps_terminal_and_contiguous_audit(config, policy, samples, stride):
    """Sparse source indices and an off-grid terminal must not use row striding."""
    dense = build(config, policy, samples)
    assert dense.cfg["action_stride"] == 1  # Old recordings keep their recipe.
    # Make the last observation off-grid and remove a valid intermediate row.
    while max(dense.observations) % 3 == 0:
        dense.observations.pop(max(dense.observations))
    dense.observations.pop(10, None)
    expected = list(dense.frames("action", dense.end))
    config["action_stride"] = stride
    sampled = build(config, policy, samples)
    sampled.observations = dense.observations.copy()
    rows = list(sampled.frames("action", sampled.end))
    terminal = expected[-1]["source_action_index"].item()
    assert terminal % 3 != 0
    expected = [
        r
        for r in expected
        if r["source_action_index"].item() % stride == 0 or r["source_action_index"].item() == terminal
    ]
    assert len(rows) == len(expected)
    for actual, reference in zip(rows, expected, strict=True):
        for key in actual:
            np.testing.assert_array_equal(actual[key], reference[key])
    assert rows[-1]["terminate"].item() == 1
    assert rows[-1]["reward"].sum() == 1
    assert [row["frame_index"] for row in sampled.audit["action"]] == list(range(len(rows)))
