import numpy as np

from utils.human_rtc import build_human_rtc_prefix


def test_human_rtc_prefix_starts_at_latest_command_and_stops_smoothly() -> None:
    previous = np.array([0.0, 1.0])
    latest = np.array([0.01, 0.98])

    prefix = build_human_rtc_prefix(
        latest_action=latest,
        previous_action=previous,
        sample_dt_s=0.01,
        chunk_length=40,
        control_fps=100.0,
        velocity_decay_steps=20,
        max_velocity=2.0,
    )

    np.testing.assert_allclose(prefix[0], latest)
    np.testing.assert_allclose(prefix[-1], prefix[-2])
    assert np.all(np.diff(prefix[:, 0]) >= 0.0)
    assert np.all(np.diff(prefix[:, 1]) <= 0.0)


def test_human_rtc_prefix_clips_velocity() -> None:
    prefix = build_human_rtc_prefix(
        latest_action=np.array([1.0]),
        previous_action=np.array([0.0]),
        sample_dt_s=0.01,
        chunk_length=3,
        control_fps=100.0,
        velocity_decay_steps=2,
        max_velocity=0.5,
    )

    assert prefix[1, 0] - prefix[0, 0] <= 0.005


def test_human_rtc_prefix_holds_when_velocity_is_zero() -> None:
    action = np.array([0.2, -0.4])
    prefix = build_human_rtc_prefix(
        latest_action=action,
        previous_action=action,
        sample_dt_s=0.01,
        chunk_length=10,
        control_fps=100.0,
        velocity_decay_steps=5,
        max_velocity=2.0,
    )

    np.testing.assert_allclose(prefix, np.repeat(action[None, :], 10, axis=0))
