#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading, queue, collections, datetime
import sounddevice as sd
import numpy as np
from scipy.signal import spectrogram, butter, filtfilt
import noisereduce as nr
from scipy.ndimage import sobel
import cv2
import openvino as ov
import sys

# ====== 匯入衛星傳輸模組 ======
sys.path.append("/home/david/Desktop/skybee")
from st6100_send_msg import st6100_send_msg


# ================== 參數設定 ==================
fs = 48000
frame_duration = 2
frame_samples = int(fs * frame_duration)
device_index = 1
DRAW = False  # 海上運行建議關閉
WINDOW_NAME = "YOLO Detection on Spectrogram"

# --- 影像座標空間篩選條件 (針對 640x640 影像) ---
MIN_Y2 = 350       # 框的下緣 (y_max) 必須大於此值 (靠近影像下方)
MAX_Y1 = 500       # 框的上緣 (y_min) 必須小於此值 (避免框太靠下方邊緣)
MIN_HEIGHT = 50    # 框的高度(像素)必須大於此數值

# 資料佇列
audio_queue = queue.Queue()
result_queue = queue.Queue()  # 辨識結果
tx_queue = queue.Queue()      # 傳送佇列
fft_buffer = []               

# 5 分鐘滑動視窗
results_window = collections.deque(maxlen=150)
threshold = 10  

# YOLO 模型設定
model_dir = "/home/david/Desktop/code/weights/best_openvino_model"
model_xml_path = f"{model_dir}/best.xml"
model_bin_path = f"{model_dir}/best.bin"
input_size = (640, 640)
class_names = ["Whistle"]
conf_threshold = 0.30
nms_iou_threshold = 0.50

# 頻率與物理轉換設定
FMIN, FMAX = 1000.0, 30000.0
IMG_W = IMG_H = 640
PIXEL_TO_HZ = (FMAX - FMIN) / IMG_H
TOTAL_MS = frame_duration * 1000.0

# Octave band 定義
OCT_CENTER = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
OCT_LOW = OCT_CENTER / np.sqrt(2)
OCT_HIGH = OCT_CENTER * np.sqrt(2)
NOISE_INTERVAL = 600  


# ================== 通用函式 ==================
def nowts():
    return datetime.datetime.now().strftime('%H:%M:%S')

def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='high', analog=False)
    return filtfilt(b, a, data)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[REC] {nowts()} 警告: {status}")
    audio_queue.put(indata[:, 0].copy())

def run_nms_xyxy(boxes_xyxy, scores, conf_th, iou_th):
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)
    rects = [[int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]
             for x1, y1, x2, y2 in boxes_xyxy]
    idxs = cv2.dnn.NMSBoxes(rects, scores, conf_th, iou_th)
    if isinstance(idxs, np.ndarray):
        return idxs.flatten()
    try:
        return np.array(idxs).flatten()
    except Exception:
        return np.array([], dtype=int)

def box_to_features(xc, yc, w, h):
    duration_ms = (w / IMG_W) * TOTAL_MS
    f_center_pixel = IMG_H - yc
    f_center = FMIN + f_center_pixel * PIXEL_TO_HZ
    f_start = max(FMIN, min(FMAX, f_center - (h / 2.0) * PIXEL_TO_HZ))
    f_end   = max(FMIN, min(FMAX, f_center + (h / 2.0) * PIXEL_TO_HZ))
    return f_start, f_end, duration_ms


# ================== Whistle 辨識分析 ==================
def analysis_thread(compiled_model, input_name, output_layer, infer_request):
    while True:
        try:
            audio_data = audio_queue.get()
            filtered = highpass_filter(audio_data, 2000, fs)
            clean = nr.reduce_noise(y=filtered, sr=fs, stationary=True)

            fft_buffer.append(clean)
            if len(fft_buffer) > int(NOISE_INTERVAL / frame_duration):
                fft_buffer.pop(0)

            # 頻譜圖生成與 Sobel 增強
            f, t, Sxx = spectrogram(clean, fs=fs, nperseg=int(0.01*fs), noverlap=int(fs*0.005))
            sobel_edges = np.hypot(sobel(Sxx, axis=0), sobel(Sxx, axis=1))
            sobel_edges_dB = np.log10(sobel_edges + 1e-10)
            sobel_norm = (sobel_edges_dB - np.min(sobel_edges_dB)) / (np.max(sobel_edges_dB) - np.min(sobel_edges_dB))
            sobel_img = (sobel_norm ** 2 * 255).astype(np.uint8)
            spectrogram_img = cv2.applyColorMap(255 - sobel_img, cv2.COLORMAP_BONE)

            freq_min_idx = np.searchsorted(f, 1000)
            freq_max_idx = np.searchsorted(f, 30000)
            spectrogram_img = spectrogram_img[freq_min_idx:freq_max_idx, :]
            spectrogram_img = cv2.flip(spectrogram_img, 0)
            spectrogram_img = cv2.resize(spectrogram_img, (IMG_W, IMG_H))

            # --- YOLO 推論 ---
            inp = spectrogram_img.astype(np.float32) / 255.0
            inp = inp.transpose(2, 0, 1)
            inp = np.expand_dims(inp, axis=0)
            infer_request.infer(inputs={input_name: inp})
            outputs = infer_request.get_output_tensor(output_layer.index).data
            preds = outputs.transpose(0, 2, 1)[0]
            preds = preds[preds[:, 4] >= conf_threshold]

            display_img = spectrogram_img.copy() if DRAW else None

            if preds.shape[0] == 0:
                if DRAW:
                    cv2.imshow(WINDOW_NAME, display_img)
                    cv2.waitKey(1)
                result_queue.put({"count": 0, "f_start": None, "f_end": None, "duration_ms": None})
                continue

            # --- 步驟 1: 物理條件篩選 (準備給 NMS 的候選名單) ---
            candidate_boxes = []
            candidate_scores = []
            
            for xc, yc, w, h, sc in preds:
                x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
                box_h = y2 - y1
                
                # 符合物理篩選條件才加入候選
                if (y2 >= MIN_Y2) and (y1 <= MAX_Y1) and (box_h >= MIN_HEIGHT):
                    candidate_boxes.append([x1, y1, x2, y2])
                    candidate_scores.append(float(sc))

            # --- 步驟 2: 執行 NMS (解決重疊框問題) ---
            kept_idx = run_nms_xyxy(candidate_boxes, candidate_scores, conf_threshold, nms_iou_threshold)
            final_count = len(kept_idx)

            if final_count > 0:
                # --- 步驟 3: 只針對被 NMS 保留下來的框進行處理 ---
                for i in kept_idx:
                    bx1, by1, bx2, by2 = candidate_boxes[i]
                    score = candidate_scores[i]

                    # 只有在 DRAW=True 且是被留下的框才畫出來
                    if DRAW:
                        cv2.rectangle(display_img, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)
                        label = f"{class_names[0]}: {score:.2f}"
                        cv2.putText(display_img, label, (int(bx1), int(by1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 挑選分數最高的一個計算特徵回傳給衛星任務
                best_i = max(kept_idx, key=lambda i: candidate_scores[i])
                bx1, by1, bx2, by2 = candidate_boxes[best_i]
                bxc, byc, bw, bh = (bx1+bx2)/2, (by1+by2)/2, (bx2-bx1), (by2-by1)
                f_start, f_end, dur = box_to_features(bxc, byc, bw, bh)
                
                result_queue.put({"count": final_count, "f_start": f_start, "f_end": f_end, "duration_ms": dur})
            else:
                result_queue.put({"count": 0, "f_start": None, "f_end": None, "duration_ms": None})

            if DRAW:
                cv2.imshow(WINDOW_NAME, display_img)
                cv2.waitKey(1)

        except Exception as e:
            print(f"[分析錯誤] {nowts()} {e}")


# ================== Results Handler Thread ==================
def results_handler_thread():

    last_valid_det = None
    printed_after_reset = False  # 達標清零後的第一次印出控制
    first_print_done = False     # 程式啟動後第一次印出控制

    while True:
        try:
            det = result_queue.get()
            c = det.get("count", 0) if det else 0
            results_window.append(c)

            # 記錄最後有效的偵測
            if det and det.get("f_start") is not None:
                last_valid_det = det

            total_detected = int(sum(results_window))

            # === 決定是否印出 ===
            should_print = False

            # 第一次一定印
            if not first_print_done:
                should_print = True
                first_print_done = True

            # segment 有偵測
            elif c > 0:
                should_print = True
                printed_after_reset = True

            # 如果剛達標清零後、下一次又有偵測才印
            elif not printed_after_reset and total_detected > 0:
                should_print = True
                printed_after_reset = True

            if should_print:
                print(f"[AGG] {nowts()} 5-min window total: {total_detected} (this segment={c})")

            # === 偵測達標 ===
            if total_detected > threshold and last_valid_det:
                tx_queue.put({
                    "category": 0,
                    "payload": {
                        "count": total_detected,
                        "f_start": int(last_valid_det["f_start"]),
                        "f_end": int(last_valid_det["f_end"]),
                        "duration_ms": int(last_valid_det["duration_ms"])
                    }
                })
                print(f"[TRIGGER] {nowts()} threshold exceeded → queue TX")

                # 清零滑動視窗與狀態
                results_window.clear()
                results_window.extend([0] * results_window.maxlen)
                last_valid_det = None
                printed_after_reset = False  # 達標清零後重新等待下一次非 0 印出

        except Exception as e:
            print(f"[Results Handler Error] {nowts()} {e}")



# ================== Noise Analyzer Thread ==================
def noise_analyzer_thread():
    while True:
        try:
            time.sleep(NOISE_INTERVAL)
            if len(fft_buffer) == 0:
                continue

            data = np.concatenate(fft_buffer)
            N = len(data)
            freqs = np.fft.rfftfreq(N, 1/fs)
            fft_vals = np.abs(np.fft.rfft(data)) ** 2
            psd = fft_vals / N

            bands_energy = []
            for f_low, f_high in zip(OCT_LOW, OCT_HIGH):
                mask = (freqs >= f_low) & (freqs < f_high)
                band_power = np.sum(psd[mask])
                band_db = 10 * np.log10(band_power + 1e-12)
                # 1. 轉成標準 float (移除 np.float64)
                # 2. 格式化為一位小數，節省傳輸字元
                bands_energy.append(float(f"{band_db:.1f}"))

            tx_queue.put({"category": 1, "payload": {"bands": bands_energy}})
            print(f"[NOISE] {nowts()} 發送噪音資料 bands={bands_energy}\n")
        except Exception as e:
            print(f"[Noise Analyzer Error] {nowts()} {e}")


# ================== Transmitter Thread ==================
msg_counter = 1

def transmitter_thread():
    global msg_counter
    while True:
        try:
            item = tx_queue.get()
            if not item:
                continue
            category = item.get("category")
            msg_id = msg_counter
            msg_counter += 1
            if msg_counter > 10:
                msg_counter = 1

            if category == 0:
                p = item["payload"]
                msg = f"0,{msg_id},{p['count']},{p['f_start']},{p['f_end']},{p['duration_ms']}"
            elif category == 1:
                bands = item["payload"]["bands"]
                # 確保所有數值都經過 float() 轉換，避免出現 np.float64 字樣
                msg = f"1,{msg_id}," + ",".join(f"{v:.1f}" for v in bands)
            else:
                continue

            print(f"[TX] {nowts()} 發送資料 (cat={category}) msg_id={msg_id}: {msg}")
            st6100_send_msg(msg_id=msg_id, msg=msg, port="/dev/ttyUSB0",
                            baudrate=9600, retries=1, wait_time=45, stale_secs=300)
        except Exception as e:
            print(f"[Transmitter Error] {nowts()} {e}")


# ================== Main ==================
def main():
    print(f"[MAIN] {nowts()} 啟動錄音與辨識系統")
    core = ov.Core()
    model = core.read_model(model=model_xml_path, weights=model_bin_path)
    compiled_model = core.compile_model(model, "AUTO")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    input_name = list(input_layer.get_names())[0]
    infer_request = compiled_model.create_infer_request()

    threading.Thread(target=analysis_thread,
                     args=(compiled_model, input_name, output_layer, infer_request),
                     daemon=True).start()
    threading.Thread(target=results_handler_thread, daemon=True).start()
    threading.Thread(target=noise_analyzer_thread, daemon=True).start()
    threading.Thread(target=transmitter_thread, daemon=True).start()

    with sd.InputStream(samplerate=fs, channels=1, dtype='float32',
                        callback=audio_callback, blocksize=frame_samples,
                        device=device_index):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"[MAIN] {nowts()} 停止程式")

if __name__ == "__main__":
    main()
