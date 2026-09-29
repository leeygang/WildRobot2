"""Minimal Hiwonder/HTD-45H serial bus interface used by SysID tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence


HEADER = bytes((0x55, 0x55))
SERVO_BROADCAST_ID = 0xFE

CMD_MOVE_TIME_WRITE = 1
CMD_MOVE_STOP = 12
CMD_ID_READ = 14
CMD_ANGLE_LIMIT_READ = 21
CMD_VIN_LIMIT_READ = 23
CMD_TEMP_MAX_LIMIT_READ = 25
CMD_TEMP_READ = 26
CMD_VIN_READ = 27
CMD_POS_READ = 28
CMD_OR_MOTOR_MODE_READ = 30
CMD_LOAD_OR_UNLOAD_WRITE = 31
CMD_LOAD_OR_UNLOAD_READ = 32


def checksum(packet_without_checksum: Iterable[int]) -> int:
    """Return the inverted byte sum used by the Hiwonder protocol."""
    data = list(packet_without_checksum)
    return (~sum(data[2:])) & 0xFF


def build_packet(
    servo_id: int, command: int, params: Sequence[int] | None = None
) -> bytes:
    values = [int(value) & 0xFF for value in (params or ())]
    packet = bytearray(HEADER)
    packet.extend((int(servo_id) & 0xFF, 3 + len(values), int(command) & 0xFF))
    packet.extend(values)
    packet.append(checksum(packet))
    return bytes(packet)


@dataclass(frozen=True)
class ServoPacket:
    servo_id: int
    command: int
    params: tuple[int, ...]
    raw: bytes


def parse_packets(data: bytes) -> list[ServoPacket]:
    packets: list[ServoPacket] = []
    for start in range(max(0, len(data) - 1)):
        if data[start : start + 2] != HEADER or start + 4 > len(data):
            continue
        length = int(data[start + 3])
        end = start + 4 + length - 1
        if length < 3 or end > len(data):
            continue
        raw = data[start:end]
        if int(raw[-1]) != checksum(raw[:-1]):
            continue
        packets.append(
            ServoPacket(
                servo_id=int(raw[2]),
                command=int(raw[4]),
                params=tuple(int(value) for value in raw[5:-1]),
                raw=raw,
            )
        )
    return packets


class PacketTransport(Protocol):
    def write(self, packet: bytes) -> None: ...

    def read_available(self, *, deadline_s: float, quiet_s: float) -> bytes: ...

    def reset_input_buffer(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SerialTransportConfig:
    port: str
    baudrate: int = 115200
    byte_timeout_s: float = 0.001
    write_timeout_s: float = 0.020


class SerialTransport:
    """Lazy pyserial transport for a USB Hiwonder debug board."""

    def __init__(self, config: SerialTransportConfig):
        self.config = config
        self._serial = None

    def open(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required; run `uv sync --extra sysid`"
            ) from exc
        self._serial = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.config.byte_timeout_s,
            write_timeout=self.config.write_timeout_s,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def write(self, packet: bytes) -> None:
        self.open()
        self._serial.write(packet)
        self._serial.flush()

    def read_available(self, *, deadline_s: float, quiet_s: float) -> bytes:
        self.open()
        data = bytearray()
        quiet_deadline: float | None = None
        while time.monotonic() < deadline_s:
            waiting = int(getattr(self._serial, "in_waiting", 0) or 0)
            chunk = self._serial.read(waiting or 1)
            if chunk:
                data.extend(chunk)
                quiet_deadline = time.monotonic() + quiet_s
            elif quiet_deadline is not None and time.monotonic() >= quiet_deadline:
                break
        return bytes(data)

    def reset_input_buffer(self) -> None:
        self.open()
        self._serial.reset_input_buffer()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


class Htd45hBus:
    """Typed commands required by the single-servo characterization fixture."""

    def __init__(
        self,
        transport: PacketTransport,
        *,
        response_timeout_s: float = 0.008,
        quiet_s: float = 0.0005,
    ):
        self.transport = transport
        self.response_timeout_s = float(response_timeout_s)
        self.quiet_s = float(quiet_s)

    def close(self) -> None:
        self.transport.close()

    def _write(self, servo_id: int, command: int, params: Sequence[int] = ()) -> None:
        if not 0 <= int(servo_id) <= 253:
            raise ValueError(f"servo id out of range: {servo_id}")
        self.transport.write(build_packet(servo_id, command, params))

    def _request(self, servo_id: int, command: int) -> ServoPacket | None:
        self.transport.reset_input_buffer()
        self._write(servo_id, command)
        raw = self.transport.read_available(
            deadline_s=time.monotonic() + self.response_timeout_s,
            quiet_s=self.quiet_s,
        )
        return next(
            (
                packet
                for packet in parse_packets(raw)
                if packet.servo_id == int(servo_id) and packet.command == command
            ),
            None,
        )

    def move(self, servo_id: int, position: int, duration_ms: int) -> None:
        if not 0 <= int(position) <= 1000:
            raise ValueError(f"servo position out of range: {position}")
        duration = max(0, min(30000, int(duration_ms)))
        self._write(
            servo_id,
            CMD_MOVE_TIME_WRITE,
            (
                int(position) & 0xFF,
                (int(position) >> 8) & 0xFF,
                duration & 0xFF,
                (duration >> 8) & 0xFF,
            ),
        )

    def move_stop(self, servo_id: int) -> None:
        self._write(servo_id, CMD_MOVE_STOP)

    def set_loaded(self, servo_id: int, loaded: bool) -> None:
        self._write(servo_id, CMD_LOAD_OR_UNLOAD_WRITE, (1 if loaded else 0,))

    def read_loaded(self, servo_id: int) -> bool | None:
        packet = self._request(servo_id, CMD_LOAD_OR_UNLOAD_READ)
        return bool(packet.params[0]) if packet and packet.params else None

    def _read_u16(self, servo_id: int, command: int) -> int | None:
        packet = self._request(servo_id, command)
        if packet is None or len(packet.params) < 2:
            return None
        return int(packet.params[0]) | (int(packet.params[1]) << 8)

    def read_position(self, servo_id: int) -> int | None:
        return self._read_u16(servo_id, CMD_POS_READ)

    def read_voltage_v(self, servo_id: int) -> float | None:
        value = self._read_u16(servo_id, CMD_VIN_READ)
        return None if value is None else value / 1000.0

    def read_temperature_c(self, servo_id: int) -> int | None:
        packet = self._request(servo_id, CMD_TEMP_READ)
        return int(packet.params[0]) if packet and packet.params else None

    def read_id(self, servo_id: int) -> int | None:
        packet = self._request(servo_id, CMD_ID_READ)
        return int(packet.params[0]) if packet and packet.params else None

    def read_angle_limits(self, servo_id: int) -> tuple[int, int] | None:
        packet = self._request(servo_id, CMD_ANGLE_LIMIT_READ)
        if packet is None or len(packet.params) < 4:
            return None
        return (
            int(packet.params[0]) | (int(packet.params[1]) << 8),
            int(packet.params[2]) | (int(packet.params[3]) << 8),
        )

    def read_voltage_limits_v(self, servo_id: int) -> tuple[float, float] | None:
        packet = self._request(servo_id, CMD_VIN_LIMIT_READ)
        if packet is None or len(packet.params) < 4:
            return None
        lower = int(packet.params[0]) | (int(packet.params[1]) << 8)
        upper = int(packet.params[2]) | (int(packet.params[3]) << 8)
        return lower / 1000.0, upper / 1000.0

    def read_temperature_limit_c(self, servo_id: int) -> int | None:
        packet = self._request(servo_id, CMD_TEMP_MAX_LIMIT_READ)
        return int(packet.params[0]) if packet and packet.params else None

    def read_motor_mode(self, servo_id: int) -> tuple[int, int] | None:
        packet = self._request(servo_id, CMD_OR_MOTOR_MODE_READ)
        if packet is None or len(packet.params) < 4:
            return None
        raw_speed = int(packet.params[2]) | (int(packet.params[3]) << 8)
        speed = raw_speed - 0x10000 if raw_speed & 0x8000 else raw_speed
        return int(packet.params[0]), speed
