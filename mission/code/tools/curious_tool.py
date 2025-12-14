"""Tool that wiggles the follower servo when the agent is unsure or encounters strange errors."""

from __future__ import annotations

import logging
import time

import serial
from smolagents import Tool

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 1000000
TARGET_ID = 8
WIGGLE_AMOUNT = 200

logger = logging.getLogger(__name__)


def calculate_checksum(data: list[int]) -> int:
    return (~sum(data)) & 0xFF


def send_packet(ser: serial.Serial, servo_id: int, instruction: int, params: list[int] | None = None) -> None:
    params = params or []
    length = len(params) + 2
    packet = [0xFF, 0xFF, servo_id, length, instruction] + params
    chk = calculate_checksum(packet[2:])
    logger.debug("Sending packet %s + checksum %02X", packet, chk)
    ser.reset_input_buffer()
    ser.write(bytearray(packet + [chk]))


def read_response(ser: serial.Serial, expected_bytes: int = 6) -> bytes:
    time.sleep(0.005)
    response = ser.read(ser.in_waiting or expected_bytes)
    return response


def write_position(ser: serial.Serial, servo_id: int, position: int) -> None:
    position = max(0, min(4095, position))
    pos_l = position & 0xFF
    pos_h = (position >> 8) & 0xFF
    send_packet(ser, servo_id, 0x03, [0x2A, pos_l, pos_h])


def scan_bus(ser: serial.Serial) -> list[int]:
    found: list[int] = []
    for check_id in range(21):
        send_packet(ser, check_id, 0x01)
        resp = read_response(ser)
        if len(resp) >= 6:
            found.append(check_id)
    return found


def set_to_mid(ser: serial.Serial, servo_id: int) -> None:
    write_position(ser, servo_id, 2048)
    time.sleep(1.0)


def perform_wiggle(ser: serial.Serial, servo_id: int, center_pos: int, amount: int) -> None:
    left_target = center_pos - amount
    right_target = center_pos + amount
    write_position(ser, servo_id, left_target)
    time.sleep(0.5)
    write_position(ser, servo_id, right_target)
    time.sleep(0.3)


class CuriousTool(Tool):
    name = "curious"
    output_type = "string"
    description = (
        "Run the wiggle sequence on the follower servo. Use this when the agent encounters confusing behavior and needs to reset or verify the arm is responsive."
    )
    inputs = {}
    outputs = {
        "details": {
            "type": "string",
            "description": "Result summary from the wiggle routine or an error message.",
        },
    }

    def forward(self) -> dict[str, str]:
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05) as ser:
                motors = scan_bus(ser)
                if TARGET_ID not in motors:
                    msg = f"Servo ID {TARGET_ID} not found on bus (seen {motors})."
                    logger.warning(msg)
                    return {"details": msg}

                set_to_mid(ser, TARGET_ID)
                for loop_idx in range(3):
                    perform_wiggle(ser, TARGET_ID, 2048, WIGGLE_AMOUNT)
                set_to_mid(ser, TARGET_ID)
            return {"details": "Curious wiggle completed; servo centered afterward."}
        except serial.SerialException as exc:
            logger.exception("Serial error during curious wiggle")
            return {"details": f"Serial error while wiggle: {exc}"}
        except Exception as exc:
            logger.exception("Unexpected error during curious wiggle")
            return {"details": f"Unexpected error while wiggle: {exc}"}
