# tools/check_model.py
import os, json, argparse, glob, numpy as np, cv2, tensorflow as tf
from collections import Counter

# ---- Register both preprocess fns so Lambda(preprocess_input) can deserialize ----
@tf.keras.utils.register_keras_serializable(name="preprocess_input")
def _mobilenet_preprocess(x):  # MobileNetV2
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)

@tf.keras.utils.register_keras_serializable(name="efficientnet_preprocess_input")
def _efficientnet_preprocess(x):  # EfficientNet
    return tf.keras.applications.efficientnet.preprocess_input(x)

CUSTOM_OBJECTS = {
    "preprocess_input": _mobilenet_preprocess,
    "efficientnet_preprocess_input": _efficientnet_preprocess,
}

CLASS_NAMES_DEFAULT = ["Closed_Eyes","Open_Eyes","Yawn","No_Yawn"]

def load_labels(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            lab = json.load(f)
        keys = sorted(lab.keys(), key=lambda k: int(k))
        return [lab[k] for k in keys]
    return CLASS_NAMES_DEFAULT

def model_has_internal_preproc(model):
    for layer in model.layers:
        name = layer.__class__.__name__
        if name == "Lambda":
            cfg = layer.get_config()
            fn = cfg.get("function")
            fn_name = fn.get("config") if isinstance(fn, dict) else fn
            if fn_name and "preprocess_input" in str(fn_name):
                return True
        if name == "Rescaling":
            cfg = layer.get_config()
            if abs(cfg.get("scale", 0) - (1/127.5)) < 1e-5 and abs(cfg.get("offset", 0) + 1.0) < 1e-5:
                return True
    return False

def get_model_input_size(model):
    shape = model.inputs[0].shape
    if len(shape) == 4:
        return int(shape[1]), int(shape[2])
    elif len(shape) == 5:
        return int(shape[2]), int(shape[3])
    else:
        return 64, 64

def preprocess_img(bgr, target_size, expect_raw_0_255):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    return resized.astype("float32") if expect_raw_0_255 else (resized/255.0).astype("float32")

def load_model(model_path):
    print("[INFO] Loading:", model_path)
    return tf.keras.models.load_model(
        model_path, compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False
    )

def cmd_probe(args):
    model = load_model(args.model)
    H, W = get_model_input_size(model)
    has_internal = model_has_internal_preproc(model)
    print("\n=== PROBE ===")
    print("Input shape:", model.inputs[0].shape)
    print("Output shape:", model.outputs[0].shape)
    print("Internal preprocessing:", has_internal)
    print("First 10 layers:")
    for i, layer in enumerate(model.layers[:10]):
        print(f" {i:02d}. {layer.name}  ->  {layer.__class__.__name__}")
    print("\nTip: If Internal preprocessing=True, feed 0..255 floats; else feed 0..1.")

def cmd_predict(args):
    model = load_model(args.model)
    labels = load_labels(args.labels)
    H, W = get_model_input_size(model); size = (W, H)
    has_internal = model_has_internal_preproc(model)

    img = cv2.imread(args.image)
    if img is None:
        print("[ERROR] Cannot read image:", args.image); return
    x = preprocess_img(img, size, expect_raw_0_255=has_internal)
    if np.isnan(x).any() or np.isinf(x).any():
        print("[WARN] NaN/Inf in input tensor!")

    probs = model.predict(np.expand_dims(x, 0), verbose=0)[0]
    idx = int(np.argmax(probs)); conf = float(probs[idx]); pred = labels[idx] if idx < len(labels) else f"id_{idx}"
    print("\n=== PREDICT ===")
    print("Image:", args.image)
    print("Preprocess mode:", "INTERNAL" if has_internal else "EXTERNAL x/255")
    print("Input range:", float(x.min()), "->", float(x.max()))
    print("Softmax:", np.round(probs, 3))
    print(f"Pred: {pred} ({conf:.3f})  idx={idx}")

def iter_images(folder):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.isdir(p):
            for q in glob.glob(os.path.join(p, "*")):
                if q.lower().endswith(exts): yield q
        else:
            if p.lower().endswith(exts): yield p

def infer_label_from_path(p, class_names):
    # expects folder name to be the class (e.g., root/Yawn/xxx.png)
    parts = os.path.normpath(p).split(os.sep)
    for c in reversed(parts[:-1]):
        if c in class_names:
            return class_names.index(c)
    return None

def cmd_eval(args):
    model = load_model(args.model)
    labels = load_labels(args.labels)
    H, W = get_model_input_size(model); size = (W, H)
    has_internal = model_has_internal_preproc(model)

    y_true, y_pred = [], []
    counts = Counter()
    imgs = list(iter_images(args.folder))
    if not imgs:
        print("[ERROR] No images found in:", args.folder); return

    for p in imgs:
        bgr = cv2.imread(p); 
        if bgr is None: continue
        x = preprocess_img(bgr, size, expect_raw_0_255=has_internal)
        probs = model.predict(np.expand_dims(x, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        gt = infer_label_from_path(p, labels)
        if gt is None: 
            # skip unlabeled paths
            continue
        y_true.append(gt); y_pred.append(idx)
        counts[gt] += 1

    if not y_true:
        print("[WARN] Couldn’t infer any ground-truth labels from folder names.")
        return

    # Confusion matrix
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1

    print("\n=== EVAL ===")
    print("Images evaluated:", len(y_true))
    print("Preprocess mode:", "INTERNAL" if has_internal else "EXTERNAL x/255")
    print("Confusion Matrix (rows=true, cols=pred):")
    hdr = "      " + " ".join([f"{i:>6d}" for i in range(n)])
    print(hdr)
    for i in range(n):
        print(f"{i:>3d} | " + " ".join([f"{cm[i,j]:>6d}" for j in range(n)]) + f"   {labels[i]}")

    acc = (np.array(y_true) == np.array(y_pred)).mean()
    print(f"\nAccuracy: {acc:.4f}")

    # per-class recall/precision
    print("\nPer-class metrics:")
    for i in range(n):
        tp = cm[i,i]
        fn = cm[i,:].sum() - tp
        fp = cm[:,i].sum() - tp
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        print(f"  [{i}] {labels[i]:<12}  prec={prec:.3f}  rec={rec:.3f}  support={cm[i,:].sum()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("probe")
    p1.add_argument("--model", required=True)

    p2 = sub.add_parser("predict")
    p2.add_argument("--model", required=True)
    p2.add_argument("--labels", default="models/labels.json")
    p2.add_argument("--image", required=True)

    p3 = sub.add_parser("eval")
    p3.add_argument("--model", required=True)
    p3.add_argument("--labels", default="models/labels.json")
    p3.add_argument("--folder", required=True, help="root folder with subfolders named by class")

    args = ap.parse_args()
    if args.cmd == "probe":   cmd_probe(args)
    elif args.cmd == "predict": cmd_predict(args)
    elif args.cmd == "eval":    cmd_eval(args)


