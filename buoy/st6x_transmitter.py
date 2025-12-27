#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import datetime
import signal
import sys

from st6100_send_msg import st6100_send_msg
from st6000_send_msg import st6000_send_msg


# ===== Global counter =====
msg_counter = 1
running = True


def nowts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def signal_handler(sig, frame):
    global running
    print(f"\n[TX] {nowts()} Stop signal received, exiting...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def select_send_func(model: str):
    """
    Select send function based on model
    """
    model = model.lower()

    if model == "st6100":
        print("[TX] Using ST6100 send function")
        return st6100_send_msg

    if model == "st6000":
        print("[TX] Using ST6000 send function")
        return st6000_send_msg

    print(f"[TX][ERR] Unknown model: {model}")
    print("Usage: python3 st_transmitter.py st6100 | st6000")
    sys.exit(1)


def main():
    global msg_counter

    # ===== parse argument =====
    if len(sys.argv) < 2:
        print("Usage: python3 st_transmitter.py st6100 | st6000")
        sys.exit(1)

    model = sys.argv[1]
    send_func = select_send_func(model)

    print(f"[TX] {nowts()} {model.upper()} Transmitter started")

    while running:
        try:
            msg_id = msg_counter
            msg_counter += 1

            if msg_counter > 10:
                msg_counter = 1

            msg = ""   # payload

            print(f"[TX] {nowts()} sending msg_id={msg_id}: {msg}")

            send_func(
                msg_id=msg_id,
                msg=msg,
                port="/dev/ttyUSB0",
                baudrate=9600,
                retries=1,
                wait_time=45,
                stale_secs=300
            )

        except Exception as e:
            print(f"[Transmitter Error] {nowts()} {e}")

        time.sleep(3)

    print(f"[TX] {nowts()} Transmitter stopped")
    sys.exit(0)


if __name__ == "__main__":
    main()
