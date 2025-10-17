# demo/app.py
import os
import sys
import glob
import json
import argparse
import shutil
import subprocess
from collections import deque

import numpy as np
import cv2
import tensorflow as tf

CLASS_NAMES_DEFAULT = ["Closed_Eyes", "Open_Eyes", "Yawn", "No_Yawn"]


# -----------------------
# Utilities
# -----------------------
def latest_model(patterns):
    """Return newest file that matches any of the given glob patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pat in patterns:
        files += glob.glob(pat)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_labels(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            lab = json.load(f)
        # accept {"0":"Closed_Eyes", ...} or {0:"Closed_Eyes", ...}
        keys = sorted(lab.keys(), key=lambda k: int(k))
        return [lab[k] for k in keys]
    return CLASS_NAMES_DEFAULT


def load_face_cascade(xml_path):
    if xml_path and os.path.exists(xml_path):
        return cv2.CascadeClassifier(xml_path)
    return None


def get_model_input_size(model):
    """Return (H, W) for frame or sequence models."""
    shape = model.inputs[0].shape
    if len(shape) == 4:      # (None, H, W, C)
        return int(shape[1]), int(shape[2])
    if len(shape) == 5:      # (None, T, H, W, C)
        return int(shape[2]), int(shape[3])
    return 64, 64


def load_model_with_custom_objects(model_path: str):
    """
    Fix Keras 3 deserialization when a saved model contains Lambda(preprocess_input).
    We map the correct preprocess function via custom_objects and disable safe_mode.
    """
    name = model_path.lower()
    custom = {}
    if "mobilenetv2" in name:
        custom["preprocess_input"] = tf.keras.applications.mobilenet_v2.preprocess_input
    elif "resnet50" in name:
        custom["preprocess_input"] = tf.keras.applications.resnet50.preprocess_input
    elif "efficientnet" in name:
        custom["preprocess_input"] = tf.keras.applications.efficientnet.preprocess_input
    # baseline/vit/cnn-lstm: usually none needed

    try:
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom, safe_mode=False)
    except Exception:
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom)


def pick_preprocess_fn(model_path, model, arg_value):
    """
    Decide which preprocess_input to apply at runtime.
    Returns (preprocess_fn, description_string).
    """
    if arg_value != "auto":
        key = arg_value.lower()
    else:
        name = (getattr(model, "name", "") + " " + (model_path or "")).lower()
        if "mobilenetv2" in name:
            key = "mobilenetv2"
        elif "resnet50" in name:
            key = "resnet50"
        elif "efficientnet" in name or "efficientnetb" in name:
            key = "efficientnet"
        elif "vit" in name or "transformer" in name:
            key = "vit"
        else:
            key = "none"  # baseline cnn, cnn-lstm

    if key == "mobilenetv2":
        return (tf.keras.applications.mobilenet_v2.preprocess_input, "MobileNetV2 [-1,1]")
    if key == "resnet50":
        return (tf.keras.applications.resnet50.preprocess_input, "ResNet50 (zero-centered/BGR)")
    if key == "efficientnet":
        return (tf.keras.applications.efficientnet.preprocess_input, "EfficientNet scale")
    # ViT/Baseline/CNN-LSTM trained on [0,1]
    return (lambda x: x, "none")


def preprocess_frame(bgr, target_size, preprocess_fn):
    """BGR -> RGB, resize, scale to [0,1], then apply chosen preprocess."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    x = (resized / 255.0).astype("float32")
    x = preprocess_fn(x)  # crucial for transfer models
    return x


def predict_frame_model(model, frame_rgb_ready):
    x = np.expand_dims(frame_rgb_ready, axis=0)  # (1,H,W,C)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs


def predict_sequence_model(model, frames_rgb_ready):
    x = np.expand_dims(frames_rgb_ready, axis=0)  # (1,T,H,W,C)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs


def draw_text(frame_bgr, text, y=30, color=(0, 255, 0)):
    cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def beep_if_possible(alarm_wav=None):
    """
    Cross-platform attempts to play a sound; falls back to terminal bell.
    No simpleaudio required.
    """
    try:
        system = sys.platform.lower()
        # Windows: winsound
        if system.startswith("win"):
            try:
                import winsound
                if alarm_wav and os.path.exists(alarm_wav):
                    winsound.PlaySound(alarm_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.Beep(1000, 300)
                return
            except Exception:
                pass
        # macOS: afplay
        if system == "darwin" and alarm_wav and os.path.exists(alarm_wav) and shutil.which("afplay"):
            subprocess.Popen(["afplay", alarm_wav])
            return
        # Linux: paplay/aplay/play (sox)
        if system.startswith("linux") and alarm_wav and os.path.exists(alarm_wav):
            for player in ["paplay", "aplay", "play"]:
                if shutil.which(player):
                    subprocess.Popen([player, alarm_wav])
                    return
        # Fallback: terminal bell
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Demo (OpenCV + TensorFlow)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .keras model. If omitted, auto-select newest in ./models or ./notebooks/models")
    parser.add_argument("--labels", type=str, default="models/labels.json", help="Path to labels.json")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g., 0) or video path")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "frame", "sequence"], help="Inference mode")
    parser.add_argument("--preprocess", type=str, default="auto",
                        choices=["auto", "none", "mobilenetv2", "resnet50", "efficientnet", "vit"],
                        help="Runtime preprocess. 'auto' infers from model path/name.")
    parser.add_argument("--haarcascade", type=str, default="assets/haarcascades/haarcascade_frontalface_default.xml",
                        help="Optional Haar cascade path for face crop")
    parser.add_argument("--show_face", action="store_true", help="Crop detected face if available")
    parser.add_argument("--window", type=int, default=8, help="Smoothing window (frame mode) or sequence length (sequence mode)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Prob threshold to trigger drowsy alert")
    parser.add_argument("--min_consec", type=int, default=3, help="Consecutive frames above threshold to alert (frame mode)")
    parser.add_argument("--alarm", type=str, default="assets/alarm.wav", help="Optional alarm WAV file")
    args = parser.parse_args()

    # Auto-pick model if not provided (search both folders)
    model_path = args.model
    if model_path is None:
        patterns = [
            "models/transfer_*_final_*.keras",
            "models/efficientnet_final_*.keras",
            "models/vit_final_*.keras",
            "models/baseline_cnn_final_*.keras",
            "models/cnn_lstm_final_*.keras",
            "notebooks/models/transfer_*_final_*.keras",
            "notebooks/models/efficientnet_final_*.keras",
            "notebooks/models/vit_final_*.keras",
            "notebooks/models/baseline_cnn_final_*.keras",
            "notebooks/models/cnn_lstm_final_*.keras",
        ]
        model_path = latest_model(patterns)

    if not model_path or not os.path.exists(model_path):
        print("[ERROR] No model found. Use --model to point to a .keras file.")
        sys.exit(1)

    print("[INFO] Loading model:", model_path)
    model = load_model_with_custom_objects(model_path)

    labels = load_labels(args.labels)
    print("[INFO] Labels:", labels)

    # Decide mode
    if args.mode == "auto":
        input_rank = len(model.inputs[0].shape)
        mode = "sequence" if input_rank == 5 else "frame"
    else:
        mode = args.mode
    print("[INFO] Inference mode:", mode)

    # Runtime preprocess
    preprocess_fn, pp_info = pick_preprocess_fn(model_path, model, args.preprocess)
    print("[INFO] Using preprocess:", pp_info)

    # Target input size
    H, W = get_model_input_size(model)
    target_size = (W, H)
    print("[INFO] Model expects:", (H, W))

    # Face detector (optional)
    face_cascade = load_face_cascade(args.haarcascade)

    # Video source
    src = 0 if (args.source.isdigit() and len(args.source) in (1, 2)) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Could not open video source:", args.source)
        sys.exit(1)

    prob_window = deque(maxlen=args.window)
    seq_window = deque(maxlen=args.window)
    consec_drowsy = 0

    # drowsy class indices
    try:
        idx_closed = labels.index("Closed_Eyes")
    except ValueError:
        idx_closed = 0
    try:
        idx_yawn = labels.index("Yawn")
    except ValueError:
        idx_yawn = 2

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        view = frame
        roi = frame

        # Optional face crop
        if args.show_face and face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = frame[max(0, y): y + h, max(0, x): x + w]
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Preprocess for the selected backbone
        x = preprocess_frame(roi, target_size, preprocess_fn)

        if mode == "frame":
            idx, conf, probs = predict_frame_model(model, x)
            prob_window.append(probs)
            avg_probs = np.mean(np.vstack(prob_window), axis=0) if len(prob_window) > 0 else probs

            # Check both eyes and yawn probabilities
            eyes_closed_prob = avg_probs[idx_closed]
            yawn_prob = avg_probs[idx_yawn]
            
            # Determine the dominant state
            if eyes_closed_prob >= args.threshold:
                pred = "Closed_Eyes"
                pred_conf = eyes_closed_prob
            elif yawn_prob >= args.threshold:
                pred = "Yawn"
                pred_conf = yawn_prob
            else:
                pred_idx = int(np.argmax(avg_probs))
                pred = labels[pred_idx]
                pred_conf = float(avg_probs[pred_idx])

            is_drowsy = (eyes_closed_prob >= args.threshold) or (yawn_prob >= args.threshold)
            consec_drowsy = consec_drowsy + 1 if is_drowsy else 0

            color = (0, 0, 255) if consec_drowsy >= args.min_consec else (0, 255, 0)
            draw_text(view, f"{pred} ({pred_conf:.2f})", y=30, color=color)

            if consec_drowsy >= args.min_consec:
                draw_text(view, "DROWSY! WAKE UP!", y=60, color=(0, 0, 255))
                beep_if_possible(args.alarm)

        else:  # sequence
            seq_window.append(x)
            if len(seq_window) == args.window:
                frames_rgb_ready = np.stack(list(seq_window), axis=0)  # (T,H,W,C)
                idx, conf, probs = predict_sequence_model(model, frames_rgb_ready)
                pred = labels[idx]
                color = (0, 0, 255) if (idx in (idx_closed, idx_yawn) and conf >= args.threshold) else (0, 255, 0)
                draw_text(view, f"{pred} ({conf:.2f})", y=30, color=color)
                if idx in (idx_closed, idx_yawn) and conf >= args.threshold:
                    draw_text(view, "DROWSY! WAKE UP!", y=60, color=(0, 0, 255))
                    beep_if_possible(args.alarm)

        cv2.imshow("Driver Drowsiness Demo", view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
