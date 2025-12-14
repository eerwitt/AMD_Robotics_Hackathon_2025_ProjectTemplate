#!/usr/bin/env python3
import serial
import time

# ================= CONFIGURATION =================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 1000000
TARGET_ID = 8
WIGGLE_AMOUNT = 200
# =================================================


def calculate_checksum(data):
    return (~sum(data)) & 0xFF


def send_packet(ser, servo_id, instruction, params=None):
    params = params or []
    length = len(params) + 2
    packet = [0xFF, 0xFF, servo_id, length, instruction] + params
    chk = calculate_checksum(packet[2:])
    ser.reset_input_buffer()
    ser.write(bytearray(packet + [chk]))


def read_response(ser, expected_bytes=6):
    time.sleep(0.005)
    response = ser.read(ser.in_waiting or expected_bytes)
    return response


def write_position(ser, servo_id, position):
    position = max(0, min(4095, position))
    pos_l = position & 0xFF
    pos_h = (position >> 8) & 0xFF
    send_packet(ser, servo_id, 0x03, [0x2A, pos_l, pos_h])


def scan_bus(ser):
    print("\n--- Scanning Bus (IDs 0-20) ---")
    found_motors = []
    for check_id in range(21):
        send_packet(ser, check_id, 0x01)
        resp = read_response(ser)
        if len(resp) >= 6:
            print(f"✅ Found Motor at ID: {check_id}")
            found_motors.append(check_id)
        else:
            print(".", end="", flush=True)
    print("\nScan Complete.")
    return found_motors


def set_to_mid(ser, servo_id):
    print(f"\n[Action] Centering ID {servo_id} to Midpoint (2048)...")
    write_position(ser, servo_id, 2048)
    time.sleep(1.0)
    print("-> Centered.")


def perform_wiggle(ser, servo_id, center_pos, amount):
    left_target = center_pos - amount
    right_target = center_pos + amount

    print(f"  <-- Left to {left_target} (0.5s hold)")
    write_position(ser, servo_id, left_target)
    time.sleep(0.5)

    print(f"  --> Right to {right_target} (0.3s hold)")
    write_position(ser, servo_id, right_target)
    time.sleep(0.3)


def main():
    try:
        print(f"Opening {SERIAL_PORT} at {BAUD_RATE}...")
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05) as ser:
            motors = scan_bus(ser)
            if TARGET_ID not in motors:
                print(f"\n❌ Error: Servo ID {TARGET_ID} not found.")
                return

            print(f"\n--- Controlling Servo ID {TARGET_ID} ---")
            set_to_mid(ser, TARGET_ID)

            print(f"\n[Action] Starting Wiggle Routine (+/- {WIGGLE_AMOUNT} steps)...")
            for i in range(3):
                print(f"Loop {i+1}:")
                perform_wiggle(ser, TARGET_ID, 2048, WIGGLE_AMOUNT)

            set_to_mid(ser, TARGET_ID)
            print("Done.")

    except serial.SerialException:
        print(f"\n❌ PERMISSION ERROR: Could not open {SERIAL_PORT}.")
        print(f"Try: sudo chmod 666 {SERIAL_PORT}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
