from __future__ import annotations

import struct

import pytest

from vtla_grip.config import GripperConfig
from vtla_grip.hardware.dh5 import (
    DH5Gripper,
    DH5ProtocolError,
    build_read_request,
    build_write_multiple_request,
    build_write_single_request,
    modbus_crc,
    parse_response,
)


def with_crc(payload: bytes) -> bytes:
    return payload + struct.pack("<H", modbus_crc(payload))


def test_builds_code_confirmed_dh5_requests() -> None:
    assert build_read_request(1, 0x0202, 1) == with_crc(bytes.fromhex("01 03 02 02 00 01"))
    assert build_write_single_request(1, 0x0100, 1) == with_crc(bytes.fromhex("01 06 01 00 00 01"))
    assert build_write_multiple_request(1, 0x0103, [1000]) == with_crc(
        bytes.fromhex("01 10 01 03 00 01 02 03 E8")
    )


def test_parses_register_feedback_and_rejects_bad_crc() -> None:
    response = with_crc(bytes.fromhex("01 03 02 03 E8"))
    assert parse_response(response, 1, 0x03) == [1000]

    with pytest.raises(DH5ProtocolError, match="CRC"):
        parse_response(response[:-1] + bytes([response[-1] ^ 0xFF]), 1, 0x03)


def test_rejects_modbus_exception() -> None:
    response = with_crc(bytes.fromhex("01 83 02"))
    with pytest.raises(DH5ProtocolError, match="exception code: 2"):
        parse_response(response, 1, 0x03)


class FakeSerial:
    def __init__(self, *, port: str, **kwargs: object) -> None:
        self.port = port
        self.kwargs = kwargs
        self.is_open = True
        self.writes: list[bytes] = []

    def reset_input_buffer(self) -> None:
        pass

    def write(self, value: bytes) -> int:
        self.writes.append(value)
        return len(value)

    def read(self, length: int) -> bytes:
        if self.port == "COM_BAD":
            return b""
        return with_crc(bytes.fromhex("01 03 02 00 00"))[:length]

    def close(self) -> None:
        self.is_open = False


def test_connect_requires_a_valid_read_only_dh5_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeSerial] = []

    def serial_factory(**kwargs: object) -> FakeSerial:
        instance = FakeSerial(**kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr("vtla_grip.hardware.dh5.serial.Serial", serial_factory)
    gripper = DH5Gripper(GripperConfig(port="COM_BAD"))

    with pytest.raises(DH5ProtocolError, match="valid DH5 Modbus response"):
        gripper.connect()

    assert gripper.connected is False
    assert created[0].is_open is False
    assert created[0].kwargs["write_timeout"] == 0.5
    assert created[0].writes == [build_read_request(1, 0x021F, 1)]


def test_changing_port_closes_old_connection_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeSerial] = []

    def serial_factory(**kwargs: object) -> FakeSerial:
        instance = FakeSerial(**kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr("vtla_grip.hardware.dh5.serial.Serial", serial_factory)
    gripper = DH5Gripper(GripperConfig(port="COM50"))

    assert gripper.connect()["port"] == "COM50"
    assert gripper.connect(GripperConfig(port="COM51"))["port"] == "COM51"

    assert len(created) == 2
    assert created[0].is_open is False
    assert created[1].is_open is True
