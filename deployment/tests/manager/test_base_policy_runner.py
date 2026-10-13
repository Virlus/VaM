"""Regression tests for shutdown without robot or visualization hardware."""

import signal
import unittest
from unittest.mock import Mock

from manager.base_policy_runner import BasePolicyRunner
from utils.shm_utils import DataChannel


def make_channel(name):
    buffer = memoryview(bytearray(8))
    shared_memory = Mock()
    shared_memory.name = name
    shared_memory.buf = buffer
    shared_memory.close.side_effect = buffer.release
    return DataChannel(shared_memory), shared_memory


class PolicyRunnerShutdownTests(unittest.TestCase):
    def setUp(self):
        self.runner = BasePolicyRunner.__new__(BasePolicyRunner)
        self.runner.running = True
        self.robot_channel, self.robot_memory = make_channel("robot_control")
        self.policy_channel, self.policy_memory = make_channel("policy")
        self.runner.robot_states = [{"shm": self.robot_channel}]
        self.runner.policy_shm = self.policy_channel

    def assert_released(self):
        self.assertFalse(self.runner.running)
        self.assertIsNone(self.runner.policy_shm)
        self.assertIsNone(self.runner.robot_states[0]["shm"])
        self.robot_memory.close.assert_called_once()
        self.policy_memory.close.assert_called_once()

    def test_repeated_stop_releases_each_resource_once(self):
        """A second stop must not inspect the released channel's length."""
        self.runner.stop()
        with self.assertRaises(ValueError):
            len(self.policy_channel)
        self.runner.stop()
        self.assert_released()

    def test_signal_then_start_finally_can_both_stop(self):
        """Reproduce SIGINT cleanup followed by the main loop's finally."""
        self.runner.connect = Mock(return_value=True)
        self.runner._update_execution = Mock(side_effect=lambda: self.runner._signal_handler(signal.SIGINT, None))
        self.assertTrue(self.runner.start())
        self.runner._update_execution.assert_called_once()
        self.assert_released()

    def test_stop_accepts_an_already_closed_channel(self):
        """A channel closed elsewhere must not trigger a truth-value check."""
        self.policy_channel.close()
        self.policy_memory.close.reset_mock()
        self.runner.stop()
        self.assert_released()

    def test_close_failure_does_not_skip_other_resources(self):
        """Robot close errors must not prevent policy cleanup."""
        self.robot_memory.close.side_effect = BufferError("exported pointers")
        self.runner.stop()
        self.runner.stop()
        self.assert_released()
