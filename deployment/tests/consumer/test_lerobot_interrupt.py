"""The inference owner unwinds before SHM cleanup, then drains its saver."""

from unittest.mock import Mock

import pytest

from consumer.policy_connector import PolicyConnector


def test_repeated_interrupt_does_not_release_shm_or_interrupt_drain():
    connector = object.__new__(PolicyConnector)
    connector.infer_rollout_config = {"format": "lerobot"}
    connector.disconnect = Mock()
    connector.stop = Mock()
    with pytest.raises(KeyboardInterrupt):
        connector._signal_handler(2, None)
    assert not connector.running
    assert not connector.inference_enabled
    connector._signal_handler(2, None)
    connector.stop.assert_not_called()
    connector.disconnect.assert_not_called()


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_early_inference_failure_still_drains_saver(failure):
    connector = object.__new__(PolicyConnector)
    connector.infer_rollout_config = {"format": "lerobot"}
    connector.infer_rollout_process = Mock()
    connector._run_policy = Mock(side_effect=failure)
    connector._stop_infer_rollout = Mock()
    connector.disconnect = Mock()
    if failure is RuntimeError:
        with pytest.raises(RuntimeError):
            connector.run()
    else:
        connector.run()
    connector._stop_infer_rollout.assert_called_once()
    connector.disconnect.assert_called_once()
