import sys
import threading
import struct
from collections import deque
from pathlib import Path

import can
import pytest


I2RT_ROOT = Path(__file__).resolve().parents[4] / "packages" / "i2rt"
sys.path.insert(0, str(I2RT_ROOT))

from i2rt.motor_drivers.can_interface import CanInterface  # noqa: E402
from i2rt.motor_drivers.dm_driver import (  # noqa: E402
    DMChainCanInterface,
    PassiveEncoderReader,
)


class FakeReceiveMode:
    def __init__(self, response_id):
        self.response_id = response_id

    def get_receive_id(self, _motor_id):
        return self.response_id


class FakeBus:
    channel_info = "fake-can"

    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.sent = []
        self.shutdown_count = 0

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout):
        del timeout
        if self.responses:
            return self.responses.popleft()
        return None

    def shutdown(self):
        self.shutdown_count += 1


def make_can_interface(bus, response_id=0x11):
    interface = object.__new__(CanInterface)
    interface.bus = bus
    interface.name = "test-can"
    interface.receive_mode = FakeReceiveMode(response_id)
    interface.use_buffered_reader = False
    interface._closed = False
    return interface


@pytest.mark.xfail(strict=True, reason="Known in pinned I2RT d13d1a1; consolidation preserves driver behavior")
def test_send_waits_for_expected_arbitration_id():
    bus = FakeBus(
        [
            can.Message(arbitration_id=0x77, data=[0] * 8),
            can.Message(arbitration_id=0x11, data=[1] * 8),
        ]
    )
    interface = make_can_interface(bus)

    response = interface._send_message_get_response(1, 1, [0] * 8, max_retry=1)

    assert response.arbitration_id == 0x11
    assert len(bus.sent) == 1


@pytest.mark.xfail(strict=True, reason="Known in pinned I2RT d13d1a1; consolidation preserves driver behavior")
def test_can_close_is_idempotent():
    bus = FakeBus()
    interface = make_can_interface(bus)

    interface.close()
    interface.close()

    assert bus.shutdown_count == 1


@pytest.mark.xfail(strict=True, reason="Known in pinned I2RT d13d1a1; consolidation preserves driver behavior")
def test_dm_chain_close_stops_thread_and_closes_bus_once():
    bus = FakeBus()
    motor_interface = make_can_interface(bus)
    chain = object.__new__(DMChainCanInterface)
    chain.running = True
    chain.channel = "fake-can"
    chain._closed = False
    chain._close_lock = threading.Lock()
    chain._control_thread = None
    chain.motor_interface = motor_interface

    chain.close()
    chain.close()

    assert not chain.running
    assert bus.shutdown_count == 1


@pytest.mark.xfail(strict=True, reason="Known in pinned I2RT d13d1a1; consolidation preserves driver behavior")
def test_passive_encoder_wraps_one_turn_to_zero_before_normalizing():
    raw_position = 4098  # Slightly above one full turn in encoder ticks.
    response = can.Message(
        arbitration_id=0x50F,
        data=struct.pack("!B h h B", 1, raw_position, 0, 0),
    )

    class FakeEncoderInterface:
        def _send_message_get_response(self, *args, **kwargs):
            del args, kwargs
            return response

    state = PassiveEncoderReader(FakeEncoderInterface()).read_encoder(0x50E)

    assert state.position < 0.01
    assert 1.0 - state.position > 0.99
