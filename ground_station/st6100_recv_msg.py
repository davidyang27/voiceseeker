# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:19:03 2025

@author: David
"""
#沒有記憶體去循環接收資料(Skybee已修復重複收到的問題)

import socket
import threading
import time
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ==========================================================
#                     基本設定區
# ==========================================================
HOST = "www.skybees.net"
PORT = 9779
POLL_INTERVAL_SEC = 10.0
RECV_TIMEOUT = 5.0
CONNECT_TIMEOUT = 10.0
BASE_BACKOFF = 1.0
MAX_BACKOFF = 60.0

DIR_RAW = r"C:\Users\adm\Desktop\oceanechoing\1.code\python\satelite"
DIR_WHISTLE = r"C:\Users\adm\Desktop\oceanechoing\1.code\C#\WpfApp1\dolphindetecter"
DIR_NOISE = r"C:\Users\adm\Desktop\oceanechoing\1.code\C#\WpfApp1\ambientnoise"
for d in [DIR_RAW, DIR_WHISTLE, DIR_NOISE]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
@dataclass
class Credential:
    user: str
    passwd: str
    type_rec: int   # 1:Text 2:ASC-HEX 3:base64

DEVICE_ID_LIST = [
    Credential("echoing_ocean_01", "echoing_ocean_01", 1),
    Credential("nsysu1", "nsysu1", 1),
]

print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# ==========================================================
#                    工具函式
# ==========================================================
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def get_file_path(user: str, suffix: str):
    """依照檔名後綴自動選擇儲存路徑"""
    safe_user = user.replace("/", "_")
    if "raw" in suffix:
        base_dir = DIR_RAW
    elif "whistle" in suffix:
        base_dir = DIR_WHISTLE
    elif "ambientnoise" in suffix:
        base_dir = DIR_NOISE
    else:
        base_dir = DIR_RAW
    return os.path.join(base_dir, f"{safe_user}_{suffix}")

# ==========================================================
#                    資料解析部分
# ==========================================================
def gga_to_decimal(coord: float) -> float:
    """GGA 格式轉十進位"""
    try:
        deg = int(coord / 100)
        minute = coord - deg * 100
        return deg + minute / 60
    except Exception:
        return coord

def split_recv_lines(data_bytes: bytes):
    s = data_bytes.decode('utf-8', errors='ignore')
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    return [ln.strip() for ln in s.split('\n') if ln.strip() != '']

def parse_recv_line(line: str):
    """解析 OK,<server_time>,<lat>,<lon>,<message>"""
    parsed = {
        "status": None, "server_time": None,
        "lat": None, "lon": None, "message": None, "raw": line
    }
    if line == "NULL":
        parsed["status"] = "NULL"; return parsed
    if line.startswith("ER"):
        parsed["status"] = "ERROR"; parsed["message"] = line; return parsed
    if line.startswith("OK"):
        parts = line.split(",", 4)
        if len(parts) >= 5:
            _, server_time, lat_str, lon_str, message = parts
            try:
                lat_raw = float(lat_str.strip())
                lon_raw = float(lon_str.strip())
                lat_dec = gga_to_decimal(lat_raw)
                lon_dec = gga_to_decimal(lon_raw)
            except Exception:
                lat_dec, lon_dec = None, None
            parsed.update({
                "status": "OK",
                "server_time": server_time.strip(),
                "lat": lat_dec,
                "lon": lon_dec,
                "message": message.strip()
            })
        else:
            parsed["status"] = "OK"; parsed["message"] = line
        return parsed
    parsed["status"] = "UNKNOWN"; parsed["message"] = line
    return parsed

def parse_satellite_message(parsed: dict) -> Optional[dict]:
    """解析 message 內容"""
    msg_fields = parsed["message"].split(",")
    try:
        category_code = int(msg_fields[0])  # 0: whistle, 1: noise
        msg_id = int(msg_fields[1])
    except Exception:
        return None
    record = {
        "category": "whistle" if category_code == 0 else "ambientnoise",
        "msg_id": msg_id,
        "server_time": parsed["server_time"],
        "lat": parsed["lat"],
        "lon": parsed["lon"],
        "raw_message": parsed["message"],
        "received_at": datetime.utcnow().isoformat() + "Z"
    }
    if category_code == 0:
        if len(msg_fields) >= 6:
            record.update({
                "count": int(msg_fields[2]),
                "f_start": float(msg_fields[3]),
                "f_end": float(msg_fields[4]),
                "duration_ms": float(msg_fields[5])
            })
    else:
        bands = [float(x) for x in msg_fields[2:] if x]
        record.update({"bands": bands})
    return record

# ==========================================================
#                    檔案寫入部分
# ==========================================================
def append_ndjson(filepath: str, data: dict):
    ensure_dir(filepath)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def append_json_array(filepath: str, data: dict):
    """將每筆資料 append 至 JSON 陣列檔（非重寫）"""
    ensure_dir(filepath)
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
    else:
        with open(filepath, "r+", encoding="utf-8") as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    content.append(data)
                else:
                    content = [content, data]
            except json.JSONDecodeError:
                content = [data]
            f.seek(0)
            json.dump(content, f, ensure_ascii=False, indent=2)
            f.truncate()

def save_record(device_id: Credential, record: dict):
    """直接 append 資料到對應 JSON"""
    user = device_id.user
    category = record["category"]

    # 1️⃣ 寫入原始封包記錄
    raw_file = get_file_path(user, "raw.ndjson")
    append_ndjson(raw_file, record)

    # 2️⃣ 根據類別寫入主檔
    if category == "whistle":
        target_file = get_file_path(user, "whistle.json")
    else:
        target_file = get_file_path(user, "ambientnoise.json")

    append_json_array(target_file, record)
    safe_print(f"[{user}] saved {category} msg_id={record['msg_id']}")

# ==========================================================
#                    接收主程式
# ==========================================================
def recv_msg_task(device_id: Credential, stop_event: threading.Event):
    backoff = BASE_BACKOFF
    while not stop_event.is_set():
        client = None
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(CONNECT_TIMEOUT)
            client.connect((HOST, PORT))
            client.settimeout(RECV_TIMEOUT)
            safe_print(f"[{device_id.user}] Connected to {HOST}:{PORT}")

            backoff = BASE_BACKOFF
            while not stop_event.is_set():
                cmd = f"AT+GRMG={device_id.user},{device_id.passwd},{device_id.type_rec}\r"
                client.sendall(cmd.encode("utf-8"))

                try:
                    data = client.recv(4096)
                except socket.timeout:
                    safe_print(f"[{device_id.user}] recv timeout.")
                    data = b""
                except socket.error as e:
                    safe_print(f"[{device_id.user}] Recv error: {e}")
                    raise

                if not data:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                lines = split_recv_lines(data)
                for line in lines:
                    parsed = parse_recv_line(line)
                    if parsed["status"] == "OK":
                        record = parse_satellite_message(parsed)
                        if record:
                            save_record(device_id, record)
                        else:
                            safe_print(f"[{device_id.user}] parse error: {line}")
                    else:
                        safe_print(f"[{device_id.user}] skip: {parsed['status']}")
                time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            safe_print(f"[{device_id.user}] Error: {e}. Backoff {backoff}s.")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass
    safe_print(f"[{device_id.user}] Worker stopping.")

# ==========================================================
#                    主執行區
# ==========================================================
def main():
    stop_event = threading.Event()
    threads = []
    for device_id in DEVICE_ID_LIST:
        t = threading.Thread(target=recv_msg_task, args=(device_id, stop_event), daemon=True)
        t.start()
        threads.append(t)

    safe_print("All workers started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        safe_print("Stopping workers...")
        stop_event.set()
        for t in threads:
            t.join(timeout=5)
    safe_print("Exited.")

if __name__ == "__main__":
    main()
