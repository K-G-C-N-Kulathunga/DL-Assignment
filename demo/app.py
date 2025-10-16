# demo/app.py
import os
import sys
import glob
import json
import argparse
from collections import deque

import numpy as np
import cv2
import tensorflow as tf

CLASS_NAMES_DEFAULT = ["Closed_Eyes", "Open_Eyes", "Yawn", "No_Yawn"]

# ----------------- Register preprocessors so Lambda(preprocess_input) deserializes -----------------
@tf.keras.utils.register_keras_serializable(name="preprocess_input")
def _mobilenet_preprocess(x):
    # For MobileNetV2: scales to [-1, 1]
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)

@tf.keras.utils.register_keras_serializable(name="efficientnet_preprocess_input")
def _efficientnet_preprocess(x):
    # For EfficientNet: also scales appropriately (TF/Keras impl)
    return tf.keras.applications.efficientnet.preprocess_input(x)

CUSTOM_OBJECTS = {
    "preprocess_input": _mobilenet_preprocess,
    "efficientnet_preprocess_input": _efficientnet_preprocess,
}

# ------------------------------------- Utils -------------------------------------
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

def model_has_internal_preproc(model):
    """Detect Lambda(preprocess_input) or Rescaling(1/127.5, offset=-1.0) inside the graph."""
    try:
        for layer in model.layers:
            name = layer.__class__.__name__
            if name == "Lambda":
                cfg = layer.get_config()
                fn_name = None
                fn = cfg.get("function")
                if isinstance(fn, dict) and "config" in fn:
                    fn_name = fn["config"]
                elif isinstance(fn, str):
                    fn_name = fn
                if fn_name and "preprocess_input" in str(fn_name):
                    return True
            if name == "Rescaling":
                cfg = layer.get_config()
                scale = cfg.get("scale")
                offset = cfg.get("offset")
                if scale is not None and offset is not None:
                    if abs(scale - (1.0 / 127.5)) < 1e-5 and abs(offset - (-1.0)) < 1e-5:
                        return True
    except Exception:
        pass
    return False

def preprocess_frame(bgr, target_size, expect_raw_0_255):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    if expect_raw_0_255:
        # Model will do preprocess_input inside
        x = resized.astype("float32")
    else:
        # External normalization (0..1)
        x = (resized / 255.0).astype("float32")
    return x

def predict_frame_model(model, frame_rgb):
    x = np.expand_dims(frame_rgb, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs

def predict_sequence_model(model, frames_rgb):
    x = np.expand_dims(frames_rgb, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs

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

# -------------------------------------- Main --------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Demo (OpenCV + TF)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--labels", type=str, default="models/labels.json")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "frame", "sequence"])
    parser.add_argument("--haarcascade", type=str, default="assets/haarcascades/haarcascade_frontalface_default.xml")
    parser.add_argument("--show_face", action="store_true")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min_consec", type=int, default=8)
    parser.add_argument("--alarm", type=str, default="assets/alarm.wav")

    # Debug & control flags
    parser.add_argument("--debug", action="store_true", help="Print per-frame stats and show ROI window")
    parser.add_argument("--force_internal_preproc", action="store_true",
                        help="Force: model does its own preprocess_input/Rescaling")
    parser.add_argument("--force_external_preproc", action="store_true",
                        help="Force: app normalizes x/255 (model has no internal preproc)")

    args = parser.parse_args()

    # Pick a model if not provided
    model_path = args.model
    if model_path is None:
        prefs = [
            "notebooks/models/transfer_*_final_*.keras",
            "notebooks/models/vit_final_*.keras",
            "notebooks/models/vit_final_*.keras",
            "notebooks/models/baseline_cnn_final_*.keras",
            "notebooks/models/cnn_lstm_final_*.keras",
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
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects= CUSTOM_OBJECTS,
        safe_mode=False,  # relax strictness due to Lambda/custom functions
    )

    labels = load_labels(args.labels)

    # Decide mode based on input rank
    if args.mode == "auto":
        input_rank = len(model.inputs[0].shape)
        mode = "sequence" if input_rank == 5 else "frame"
    else:
        mode = args.mode
    print("[INFO] Inference mode:", mode)

    # Input size
    H, W = get_model_input_size(model)
    target_size = (W, H)

    # Detect/override preprocessing
    has_internal = model_has_internal_preproc(model)
    if args.force_internal_preproc:
        has_internal = True
    if args.force_external_preproc:
        has_internal = False

    print(
        "[INFO] Preprocessing:",
        "INTERNAL (Lambda/Rescaling in model)" if has_internal else "EXTERNAL (x/255 in app)"
    )

    face_cascade = load_face_cascade(args.haarcascade)

    # Initialize camera
    src = 0 if (args.source.isdigit() and len(args.source) == 1) else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if os.name == "nt" and str(src).isdigit() else 0)
    if not cap.isOpened():
        print("[ERROR] Could not open video source:", args.source)
        sys.exit(1)

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
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = frame[max(0, y): y + h, max(0, x): x + w]
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x = preprocess_frame(roi, target_size, expect_raw_0_255=has_internal)

        # Debug visuals and stats
        if args.debug:
            mn, mx = float(np.min(x)), float(np.max(x))
            if np.isnan(x).any() or np.isinf(x).any():
                print("[WARN] NaN/Inf in input tensor!")
            if args.show_face:
                vis = cv2.cvtColor(cv2.resize(roi, target_size), cv2.COLOR_BGR2RGB)
                cv2.imshow("ROI (fed to model size)", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] input range min={mn:.4f} max={mx:.4f}")

        if mode == "frame":
            idx, conf, probs = predict_frame_model(model, x)
            prob_window.append(probs)
            avg_probs = np.mean(np.vstack(prob_window), axis=0) if len(prob_window) > 0 else probs
            pred_idx = int(np.argmax(avg_probs))
            pred = labels[pred_idx]
            pred_conf = float(avg_probs[pred_idx])

            if args.debug:
                print("[DEBUG] probs:", np.round(avg_probs, 3), "| argmax=", pred_idx, "| label=", pred)

            is_drowsy = (avg_probs[idx_closed] >= args.threshold) or (avg_probs[idx_yawn] >= args.threshold)
            consec_drowsy = consec_drowsy + 1 if is_drowsy else 0

            color = (0, 0, 255) if consec_drowsy >= args.min_consec else (0, 255, 0)
            cv2.putText(view, f"{pred} ({pred_conf:.2f})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            if consec_drowsy >= args.min_consec:
                cv2.putText(view, "DROWSY! WAKE UP!", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                beep_if_possible(args.alarm)

        else:  # sequence
            seq_window.append(x)
            if len(seq_window) == args.window:
                frames_rgb = np.stack(list(seq_window), axis=0)
                idx, conf, probs = predict_sequence_model(model, frames_rgb)
                pred = labels[idx]
                if args.debug:
                    print("[DEBUG] probs:", np.round(probs, 3), "| argmax=", idx, "| label=", pred)
                color = (0, 0, 255) if (idx == idx_closed or idx == idx_yawn) else (0, 255, 0)
                cv2.putText(view, f"{pred} ({conf:.2f})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                if idx in (idx_closed, idx_yawn) and conf >= args.threshold:
                    cv2.putText(view, "DROWSY! WAKE UP!", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    beep_if_possible(args.alarm)

        cv2.imshow("Driver Drowsiness Demo", view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment for exact numeric reproducibility (disables oneDNN optimizations):
    # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()
