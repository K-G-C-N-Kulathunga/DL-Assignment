
import os
import sys
import time
import glob
import json
import argparse
from collections import deque

import numpy as np
import cv2
import tensorflow as tf

CLASS_NAMES_DEFAULT = ["Closed_Eyes","Open_Eyes","Yawn","No_Yawn"]

def latest_model(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_labels(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            lab = json.load(f)
        keys = sorted(lab.keys(), key=lambda k: int(k))
        return [lab[k] for k in keys]
    return CLASS_NAMES_DEFAULT

def load_face_cascade(xml_path):
    if xml_path and os.path.exists(xml_path):
        return cv2.CascadeClassifier(xml_path)
    return None

def get_model_input_size(model):
    shape = model.inputs[0].shape
    if len(shape) == 4:
        return int(shape[1]), int(shape[2])
    elif len(shape) == 5:
        return int(shape[2]), int(shape[3])
    else:
        return 64, 64

def preprocess_frame(bgr, target_size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    x = (resized / 255.0).astype("float32")
    return x

def predict_frame_model(model, frame_rgb_norm):
    x = np.expand_dims(frame_rgb_norm, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs

def predict_sequence_model(model, frames_rgb_norm):
    x = np.expand_dims(frames_rgb_norm, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs

def draw_overlay(frame_bgr, text, color=(0,255,0)):
    cv2.putText(frame_bgr, text, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def beep_if_possible(alarm_wav=None):
    try:
        import platform
        system = platform.system().lower()
        if alarm_wav and os.path.exists(alarm_wav):
            try:
                import simpleaudio as sa
                wave_obj = sa.WaveObject.from_wave_file(alarm_wav)
                wave_obj.play()
                return
            except Exception:
                pass
        if system == "windows":
            import winsound
            winsound.Beep(1000, 300)
        else:
            sys.stdout.write('\a'); sys.stdout.flush()
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Demo (OpenCV + TF)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--labels", type=str, default="models/labels.json")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto","frame","sequence"])
    parser.add_argument("--haarcascade", type=str, default="assets/haarcascades/haarcascade_frontalface_default.xml")
    parser.add_argument("--show_face", action="store_true")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min_consec", type=int, default=8)
    parser.add_argument("--alarm", type=str, default="assets/alarm.wav")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        prefs = [
            "models/transfer_*_final_*.keras",
            "models/efficientnet_final_*.keras",
            "models/vit_final_*.keras",
            "models/baseline_cnn_final_*.keras",
            "models/cnn_lstm_final_*.keras"
        ]
        for pat in prefs:
            p = latest_model(pat)
            if p:
                model_path = p
                break

    if not model_path or not os.path.exists(model_path):
        print("[ERROR] No model found. Use --model to specify a .keras file.")
        sys.exit(1)

    print("[INFO] Loading model:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    labels = load_labels(args.labels)

    if args.mode == "auto":
        input_rank = len(model.inputs[0].shape)
        mode = "sequence" if input_rank == 5 else "frame"
    else:
        mode = args.mode
    print("[INFO] Inference mode:", mode)

    H, W = get_model_input_size(model)
    target_size = (W, H)

    face_cascade = load_face_cascade(args.haarcascade)

    src = 0 if (args.source.isdigit() and len(args.source) == 1) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Could not open video source:", args.source)
        sys.exit(1)

    from collections import deque
    prob_window = deque(maxlen=args.window)
    seq_window = deque(maxlen=args.window)
    consec_drowsy = 0

    try:
        idx_closed = labels.index("Closed_Eyes")
        idx_yawn = labels.index("Yawn")
    except ValueError:
        idx_closed, idx_yawn = 0, 2

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        view = frame
        roi = frame

        if args.show_face and face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
            if len(faces) > 0:
                x,y,w,h = faces[0]
                roi = frame[max(0,y):y+h, max(0,x):x+w]
                cv2.rectangle(view, (x,y), (x+w,y+h), (0,255,0), 2)

        x = preprocess_frame(roi, target_size)

        if mode == "frame":
            idx, conf, probs = predict_frame_model(model, x)
            prob_window.append(probs)
            avg_probs = np.mean(np.vstack(prob_window), axis=0) if len(prob_window)>0 else probs
            pred_idx = int(np.argmax(avg_probs))
            pred = labels[pred_idx]
            pred_conf = float(avg_probs[pred_idx])

            is_drowsy = (avg_probs[idx_closed] >= args.threshold) or (avg_probs[idx_yawn] >= args.threshold)
            consec_drowsy = consec_drowsy + 1 if is_drowsy else 0

            color = (0,0,255) if consec_drowsy >= args.min_consec else (0,255,0)
            cv2.putText(view, f"{pred} ({pred_conf:.2f})", (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            if consec_drowsy >= args.min_consec:
                cv2.putText(view, "DROWSY! WAKE UP!", (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                beep_if_possible(args.alarm)

        else:
            seq_window.append(x)
            if len(seq_window) == args.window:
                frames_rgb_norm = np.stack(list(seq_window), axis=0)
                idx, conf, probs = predict_sequence_model(model, frames_rgb_norm)
                pred = labels[idx]
                color = (0,0,255) if (idx == idx_closed or idx == idx_yawn) else (0,255,0)
                cv2.putText(view, f"{pred} ({conf:.2f})", (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                if idx in (idx_closed, idx_yawn) and conf >= args.threshold:
                    cv2.putText(view, "DROWSY! WAKE UP!", (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    beep_if_possible(args.alarm)

        cv2.imshow("Driver Drowsiness Demo", view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
