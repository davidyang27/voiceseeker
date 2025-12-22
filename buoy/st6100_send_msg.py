import serial
import time
import re
import datetime
import os
import csv


def write_full_msg_to_csv(full_msg, csv_path="st6100_tx_log.csv"):
    """
    Save full_msg to CSV file
    full_msg format:
    lat,lon,utc,temp,msg
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = full_msg.split(",", 4)
    if len(parts) != 5:
        print("[CSV] Invalid full_msg format, skip saving")
        return

    lat, lon, utc, temp, msg = parts

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "latitude", "longitude", "utc", "temp", "message"])

        writer.writerow([now, lat, lon, utc, temp, msg])

    print(f"[CSV] Saved to {csv_path}")


def get_gps_info(ser, retries, wait_time, stale_secs):
    """
    Try multiple times to get GPS GGA information from the ST6100.
    :param ser: An opened serial connection
    :param retries: Maximum retry attempts
    :param wait_time: Maximum seconds to wait for GPS response (corresponds to waitSecs)
    :param stale_secs: Expiration seconds (staleSecs)
    :return: (latitude_raw, lat_dir, longitude_raw, lon_dir) or None
    """
    for attempt in range(1, retries + 1):
        print(f"[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Attempt {attempt}...")
        result = read_gps(ser, max_wait=wait_time, stale_secs=stale_secs)
        if result:
            print(f"[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Success: {result}")
            return result
        else:
            print(f"[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Attempt {attempt} failed")
            time.sleep(1)

    print(f"[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Exceeded maximum attempts, still no GPGGA found")
    return None


def read_gps(ser, max_wait, stale_secs):
    """
    Continuously read GPS GGA data from the serial port until $GPGGA is found
    or the maximum waiting time is exceeded.
    :param ser: serial.Serial object
    :param max_wait: Maximum waiting seconds (waitSecs)
    :param stale_secs: Expiration seconds (staleSecs)
    :return: (latitude_raw, lat_dir, longitude_raw, lon_dir) or None
    """
    try:
        command = f'AT%GPS={stale_secs},{max_wait},"GGA"\r'
        print(f"[AT] {datetime.datetime.now().strftime('%H:%M:%S')} Sending GPS command: {command.strip()}")
        ser.write(command.encode('ascii'))
        time.sleep(0.5)  # Wait for GPS module to start responding

        start_time = time.time()
        buffer = ""

        while time.time() - start_time < max_wait:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode(errors="ignore")
                buffer += data

                lines = buffer.split("\n")
                buffer = lines[-1]  # Keep the last (possibly incomplete) line

                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue

                    # print("GPS RAW:", line)

                    if "ERROR" in line:
                        print("f[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Received ERROR")
                        return None

                    if "$GPGGA" in line:
                        if line.startswith("%GPS:"):
                            line = line.replace("%GPS:", "").strip()

                        parts = line.split(",")
                        if len(parts) < 6:
                            print("f[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Incomplete message:", line)
                            continue

                        latitude_raw = parts[2]
                        lat_dir = parts[3]
                        longitude_raw = parts[4]
                        lon_dir = parts[5]

                        if latitude_raw and longitude_raw:
                            # print(f"[GPS] Success: {latitude_raw}, {lat_dir}, {longitude_raw}, {lon_dir}")
                            return latitude_raw, lat_dir, longitude_raw, lon_dir
            else:
                time.sleep(0.1)  # Wait a bit if no data

        print(f"[GPS] {datetime.datetime.now().strftime('%H:%M:%S')} Timeout: No GPGGA message found")
        return None

    except Exception as e:
        print("f[GPS Parsing Error] {datetime.datetime.now().strftime('%H:%M:%S')} ", e)
        return None

def get_utc_info(ser, timeout=3):
    """
    Get UTC time via AT%UTC
    Return string: YYYY/MM/DD,HH:MM:SS or None
    """
    try:
        ser.write(b"AT%UTC\r")
        time.sleep(0.5)

        start = time.time()
        buffer = ""

        while time.time() - start < timeout:
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting).decode(errors="ignore")
                lines = buffer.replace("\r", "\n").split("\n")

                for line in lines:
                    line = line.strip()
                    if line.startswith("%UTC:"):
                        return line.replace("%UTC:", "").strip()
            time.sleep(0.1)

        print("[UTC] Timeout")
        return None

    except Exception as e:
        print("[UTC] Error:", e)
        return None

def get_temp_info(ser, timeout=3):
    """
    Get temperature via ATS85?
    Remove leading zeros from result
    Return string or None
    """
    try:
        ser.write(b"ATS85?\r")
        time.sleep(0.5)

        start = time.time()
        buffer = ""

        while time.time() - start < timeout:
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting).decode(errors="ignore")
                lines = buffer.replace("\r", "\n").split("\n")

                for line in lines:
                    line = line.strip()
                    if line.startswith("S85:"):
                        temp_raw = line.replace("S85:", "").strip()
                        temp_clean = temp_raw.lstrip("0") or "0"
                        return temp_clean
            time.sleep(0.1)

        print("[TEMP] Timeout")
        return None

    except Exception as e:
        print("[TEMP] Error:", e)
        return None

def st6100_send_msg(msg_id: int, msg: str, port: str = "/dev/ttyUSB0",
                     baudrate: int = 9600, retries: int = 1,
                     wait_time: int = 45, stale_secs: int = 2):
    """
    Send an AT command to the ST6100 satellite modem.
    It first obtains GPS coordinates, then sends the message.
    stale_secs: Expiration seconds (1–600)
    wait_time: GPS fix waiting seconds (1–600)
    """
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # Get GPS info first
            gps_info = get_gps_info(ser, retries=retries, wait_time=wait_time, stale_secs=stale_secs)
            # print("GPS Info:", gps_info)

            # Get UTC & Temperature
            utc_info = get_utc_info(ser) or "N"
            temp_info = get_temp_info(ser) or "N"

            if not gps_info:
                # print("[GPS] Failed to get GPS info")
                full_msg = f"N,N,{msg}"
            else:
                # Prefix GPS coordinates to the message
                full_msg = f"{gps_info[0]},{gps_info[2]},{msg}"

            # Save to CSV
            write_full_msg_to_csv(full_msg)
            
            # Calculate message length
            msg_len = len(full_msg.encode("utf-8"))+2

            # Fixed parameters
            service_class = 2
            lifetime = 10
            data_format = 1
            sin = 128
            min_code = 1

            # --- Check and clear msg_id before sending ---
            momc_cmd = f"AT%MOMC={msg_id}\r"
            momd_cmd = f"AT%MOMD={msg_id}\r"
            
            print(f"[AT] {datetime.datetime.now().strftime('%H:%M:%S')} Sending: {momc_cmd.strip()}")
            ser.write(momc_cmd.encode("utf-8"))
            time.sleep(1.5)
            response = ser.read_all().decode(errors="ignore").strip()
            if response:
                print(f"[AT] Response: {response}")

            print(f"[AT] {datetime.datetime.now().strftime('%H:%M:%S')} Sending: {momd_cmd.strip()}")
            ser.write(momd_cmd.encode("utf-8"))
            time.sleep(1.5)
            response = ser.read_all().decode(errors="ignore").strip()
            if response:
                print(f"[AT] Response: {response}")

            # Compose AT command
            at_command = f'AT%MOMT={msg_id},{service_class},{lifetime},{msg_len},{data_format},{sin},{min_code},"{full_msg}"\r'

            print(f"[AT] {datetime.datetime.now().strftime('%H:%M:%S')} Sending command: {at_command.strip()}")
            
            # Send AT%MOMT
            ser.write(at_command.encode("utf-8"))
            time.sleep(1.5)

            # Read response
            response = ser.read_all().decode(errors="ignore")
            for line in response.replace("\r", "\n").split("\n"):
                line = line.strip()
                if line:
                    print(f"[AT] {datetime.datetime.now().strftime('%H:%M:%S')} {line}")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None
