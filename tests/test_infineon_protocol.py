from __future__ import annotations

import struct

from vtla_grip.hardware.infineon import InfineonFrameParser


def data_frame(frame_type: int, values: list[int]) -> bytes:
    payload = struct.pack("<" + "H" * len(values), *values)
    return b"DATA" + bytes([1, frame_type, 32, 0]) + struct.pack("<H", len(payload)) + payload


def test_parses_signal_frame_incrementally() -> None:
    parser = InfineonFrameParser()
    frame = data_frame(1, list(range(32)))
    assert parser.feed(frame[:13]) == []
    assert parser.feed(frame[13:]) == [[float(value) for value in range(32)]]


def test_debug_frame_exposes_first_raw_value_per_channel_block() -> None:
    parser = InfineonFrameParser()
    values = list(range(160))
    frames = parser.feed(data_frame(2, values))
    assert frames == [[float(value) for value in range(32)]]


def test_resynchronizes_after_noise_and_bad_header() -> None:
    parser = InfineonFrameParser()
    bad = b"DATA" + bytes([9, 1, 32, 0]) + struct.pack("<H", 64)
    valid = data_frame(1, [7] * 32)
    assert parser.feed(b"noise" + bad + valid) == [[7.0] * 32]
    assert parser.invalid_headers == 1
