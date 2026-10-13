"""The retained VAM inference and joint executor paths work without hardware."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from consumer import policy_connector
from manager.base_policy_runner import BasePolicyRunner
from manager.utils import JointActionManager


def test_joint_executor_initialization_keeps_reset_and_rtc(monkeypatch, tmp_path):
    monkeypatch.setattr("manager.base_policy_runner.signal.signal", lambda *args: None)
    runner = BasePolicyRunner(
        config={"policy": {"params": {"enable_rtc": True}}},
        signal_files={"policy_runner_reset_command": str(tmp_path / "reset")},
        robot_control_shms=[{"shm": "I2RTRobot_1_control"}],
        chunk_manager={
            "type": "joint_action_manager",
            "search_tolerance": 20,
            "discount_factor": 0.998,
            "direction_consistency_range": 10,
        },
    )
    assert isinstance(runner.chunk_manager, JointActionManager)
    assert runner.enable_rtc
    assert runner.device_first_actions == {"I2RTRobot_1": None}
    assert runner.policy_shm is None
    assert not runner.running


@pytest.mark.parametrize("value", [0.6, np.array([0.6]), np.array([0.6, 0.7])])
def test_value_worker_only_publishes_observation_scalar(monkeypatch, value):
    output = Mock(created=True)
    connector = SimpleNamespace(
        value_prediction_query_interval=0.1,
        value_prediction_running=True,
        inference_enabled=True,
        enable_value_prediction=True,
        _assemble_observation_dict=lambda: ({"state": np.zeros(14)}, 12345),
        _prepare_policy_input=lambda *args, **kwargs: {},
        action_configs=[{"action_dim": 14}],
        chunk_length=100,
        _policy_lock=nullcontext(),
        value_shm=output,
    )

    def predict(*args):
        connector.value_prediction_running = False
        return {"value": value}

    connector.policy = SimpleNamespace(predict_value=predict)
    monkeypatch.setattr(policy_connector, "precise_loop_timing", lambda _: lambda: None)
    policy_connector.PolicyConnector._value_prediction_worker(connector)
    if np.asarray(value).size == 1:
        output.write.assert_called_once()
        np.testing.assert_array_equal(output.write.call_args.args[0], [0.6])
        assert output.write.call_args.args[1] == 12345
    else:
        output.write.assert_not_called()
