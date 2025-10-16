
# Demo (Optional Bonus) — Real-time / Video Detection

Run a live webcam or video file demo that uses your best trained model to detect drowsiness.

## Quick Start

```bash
pip install opencv-python tensorflow numpy simpleaudio
python demo/app.py --source 0 --show_face
```

- Press `q` to quit.
- Use `--source path/to/video.mp4` to run on a video file.

## Options

- `--model` : Path to a `.keras` model. If omitted, the script auto-picks the latest model in `models/`.
- `--labels`: Path to `models/labels.json` (optional, defaults provided).
- `--mode`  : `auto` (default), `frame`, or `sequence`.
- `--window`: Smoothing window (frame mode) or sequence length (sequence mode). Default `8`.
- `--threshold`: Probability threshold to consider drowsy (`Closed_Eyes` or `Yawn`). Default `0.6`.
- `--min_consec`: Consecutive frames above threshold to trigger an alarm (frame mode). Default `8`.
- `--show_face`: Enables Haar cascade face crop for better focus on the driver face.
- `--haarcascade`: Path to Haar cascade (e.g., `assets/haarcascades/haarcascade_frontalface_default.xml`).

> Tip: To use face detection, download the Haar cascade from OpenCV and place it at:
> `assets/haarcascades/haarcascade_frontalface_default.xml`

## Examples

```bash
# Use latest model automatically, webcam 0:
python demo/app.py --source 0 --show_face

# Explicit transfer model:
python demo/app.py --model models/transfer_mobilenetv2_final_*.keras --source 0

# CNN-LSTM with 12-frame sequences on a video file:
python demo/app.py --model models/cnn_lstm_final_*.keras --mode sequence --window 12 --source path/to/video.mp4
```
C:\Users\kgcha\Documents\GitHub\DL-Assignment\notebooks\models