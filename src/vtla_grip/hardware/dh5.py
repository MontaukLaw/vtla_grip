from __future__ import annotations

import struct
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import serial
from serial.tools import list_ports

from ..config import GripperConfig


class DH5ProtocolError(RuntimeError):
    pass


def modbus_crc(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def build_read_request(slave_id: int, address: int, count: int) -> bytes:
    payload = struct.pack(">BBHH", slave_id, 0x03, address, count)
    return payload + struct.pack("<H", modbus_crc(payload))


def build_write_single_request(slave_id: int, address: int, value: int) -> bytes:
    payload = struct.pack(">BBHH", slave_id, 0x06, address, value)
    return payload + struct.pack("<H", modbus_crc(payload))


def build_write_multiple_request(slave_id: int, address: int, values: list[int]) -> bytes:
    encoded = b"".join(struct.pack(">H", value) for value in values)
    payload = struct.pack(">BBHHB", slave_id, 0x10, address, len(values), len(encoded)) + encoded
    return payload + struct.pack("<H", modbus_crc(payload))


def parse_response(response: bytes, slave_id: int, function_code: int) -> list[int]:
    if len(response) < 5:
        raise DH5ProtocolError(f"DH5 response is incomplete: {response.hex()}")
    if modbus_crc(response[:-2]) != struct.unpack("<H", response[-2:])[0]:
        raise DH5ProtocolError("DH5 response CRC check failed")
    if response[0] != slave_id:
        raise DH5ProtocolError(f"unexpected DH5 slave id: {response[0]}")
    if response[1] & 0x80:
        raise DH5ProtocolError(f"DH5 Modbus exception code: {response[2]}")
    if response[1] != function_code:
        raise DH5ProtocolError(f"unexpected DH5 function code: {response[1]}")
    if function_code == 0x03:
        byte_count = response[2]
        data = response[3 : 3 + byte_count]
        if len(data) != byte_count or byte_count % 2:
            raise DH5ProtocolError("invalid DH5 register payload length")
        return [struct.unpack(">H", data[index : index + 2])[0] for index in range(0, len(data), 2)]
    return []


@dataclass(frozen=True)
class DH5Feedback:
    position: int | None = None
    speed: int | None = None
    current: int | None = None
    fault: int | None = None


class DH5Gripper:
    """Thread-safe DH5 Modbus RTU adapter based on the proven arm_gripping protocol."""

    def __init__(self, config: GripperConfig) -> None:
        self.config = config
        self._active_config: GripperConfig | None = None
        self._serial: serial.Serial | None = None
        self._lock = threading.RLock()
        self._last_error: str | None = None

    @property
    def connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    def connect(self, config: GripperConfig | None = None) -> dict[str, Any]:
        with self._lock:
            requested = config or self.config
            if self.connected and self._active_config == requested:
                return self.status()
            if self._serial is not None:
                self._serial.close()
                self._serial = None
                self._active_config = None
            self.config = requested
            try:
                self._serial = serial.Serial(
                    port=self.config.port,
                    baudrate=self.config.baud_rate,
                    stopbits=self.config.stop_bits,
                    parity=self.config.parity,
                    timeout=0.2,
                    write_timeout=0.5,
                    inter_byte_timeout=0.02,
                )
                self._active_config = self.config
                # Opening a Windows COM port is not proof that it is the DH5.
                # A read-only Modbus probe rejects Bluetooth/other serial ports.
                self._read(0x021F, 1)
                self._last_error = None
            except (OSError, serial.SerialException, DH5ProtocolError) as exc:
                if self._serial is not None:
                    self._serial.close()
                self._serial = None
                self._active_config = None
                self._last_error = f"{self.config.port}: {exc}"
                raise DH5ProtocolError(
                    f"{self.config.port} opened but did not return a valid DH5 Modbus response: {exc}"
                ) from exc
            return self.status()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            if self._serial is not None:
                self._serial.close()
                self._serial = None
                self._active_config = None
            return self.status()

    def status(self, read_feedback: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "connected": self.connected,
            "port": self._active_config.port if self._active_config else self.config.port,
            "baud_rate": self.config.baud_rate,
            "modbus_id": self.config.modbus_id,
            "last_error": self._last_error,
        }
        if read_feedback and self.connected:
            result["feedback"] = asdict(self.feedback())
        return result

    def initialize(self, mode: int = 0x01, timeout_s: float = 6.0) -> dict[str, Any]:
        if mode not in {0x01, 0xA5}:
            raise ValueError("DH5 initialization mode must be 0x01 or 0xA5")
        with self._lock:
            self._write_single(0x0100, mode)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                values = self._read(0x0100, 1)
                if values and values[0] == 0:
                    return {"initialized": True, **self.status()}
                time.sleep(0.3)
        raise DH5ProtocolError("DH5 initialization timed out")

    def set_position(self, value: int) -> dict[str, Any]:
        self._validate_range("position", value, 0, 1000)
        with self._lock:
            self._write_multiple(0x0103, [value])
        return {"position_command": value}

    def set_speed(self, value: int) -> dict[str, Any]:
        self._validate_range("speed", value, 1, 100)
        with self._lock:
            self._write_multiple(0x0104, [value])
        return {"speed_command": value}

    def set_force(self, value: int) -> dict[str, Any]:
        self._validate_range("force", value, 20, 100)
        with self._lock:
            self._write_multiple(0x0101, [value])
        return {"force_command": value}

    def feedback(self) -> DH5Feedback:
        with self._lock:
            return DH5Feedback(
                position=self._read(0x0202, 1)[0],
                speed=self._read(0x0203, 1)[0],
                current=self._read(0x0204, 1)[0],
                fault=self._read(0x021F, 1)[0],
            )

    def _exchange(self, request: bytes, response_length: int, function_code: int) -> list[int]:
        if not self.connected or self._serial is None:
            raise DH5ProtocolError("DH5 gripper is not connected")
        try:
            self._serial.reset_input_buffer()
            self._serial.write(request)
            response = self._serial.read(response_length)
        except (OSError, serial.SerialException) as exc:
            self._last_error = str(exc)
            raise DH5ProtocolError(f"DH5 serial transaction failed: {exc}") from exc
        return parse_response(response, self.config.modbus_id, function_code)

    def _read(self, address: int, count: int) -> list[int]:
        request = build_read_request(self.config.modbus_id, address, count)
        return self._exchange(request, 5 + 2 * count, 0x03)

    def _write_single(self, address: int, value: int) -> None:
        request = build_write_single_request(self.config.modbus_id, address, value)
        self._exchange(request, 8, 0x06)

    def _write_multiple(self, address: int, values: list[int]) -> None:
        request = build_write_multiple_request(self.config.modbus_id, address, values)
        self._exchange(request, 8, 0x10)

    @staticmethod
    def _validate_range(name: str, value: int, minimum: int, maximum: int) -> None:
        if not minimum <= int(value) <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")


def list_serial_ports() -> list[dict[str, Any]]:
    return [
        {
            "device": port.device,
            "description": port.description,
            "manufacturer": port.manufacturer,
            "vid": port.vid,
            "pid": port.pid,
        }
        for port in list_ports.comports()
    ]
