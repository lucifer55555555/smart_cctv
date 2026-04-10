# 🎯 Smart-CCTV Campus Safety System

> AI-powered surveillance that detects weapons and violence in real-time — built for campus security teams.

Smart-CCTV fuses two deep learning models into a single threat-assessment pipeline. It watches a live camera feed, identifies weapons and fighting, assigns a risk level, and automatically saves an incident clip + log entry the moment something goes wrong.

---

## How It Works

Before the system can run, **both models must exist** in the `models/` folder. One is trained by you using the provided script; the other must be sourced or trained separately.

```
STEP 0 — Train / Obtain Models
        │
        ├── weapon_yolo.pt       ← train using train_roboflow.py (instructions below)
        └── violence_cnn_lstm.pt ← TorchScript CNN+LSTM model (source/train separately)
                │
                ▼
STEP 1 — Camera Feed  (webcam or RTSP stream)
                │
                ▼
STEP 2 — Detection  (two models run in parallel)
        │
        ├── WeaponDetector  [SYNC, every frame]
        │     YOLOv8 → bounding boxes → multi-frame confirmation filter
        │
        └── FightDetector   [ASYNC, background thread]
              CNN+LSTM → 30-frame sequence → smoothed violence probability
                │
                ▼
STEP 3 — RiskEngine  (combines both detector outputs)
              NO_RISK / MEDIUM_RISK / HIGH_RISK / CRITICAL
                │
                ▼
STEP 4 — Incident Handling  (only on threat)
        │
        ├── VideoBuffer  → saves MP4 clip (5s before + 5s after the trigger)
        └── IncidentLogger → appends row to incidents/logs/incidents.csv
                │
                ▼
STEP 5 — Web Dashboard  (Flask, runs at localhost:5000)
              Live feed · Incident log · Analytics dashboard
```

---

## The Two Models

| | `weapon_yolo.pt` | `violence_cnn_lstm.pt` |
|---|---|---|
| **What it detects** | Guns, knives, blades, rifles, etc. | Fighting / violent activity |
| **Architecture** | YOLOv8 (Ultralytics) | CNN + LSTM (TorchScript) |
| **Loaded with** | `YOLO(path)` | `torch.jit.load(path)` |
| **Input** | Single BGR frame | Sequence of 30 `(3×224×224)` tensors |
| **Output** | Bounding boxes + labels + confidence | Single float — violence probability |
| **Runs on** | Main thread (every frame) | Background daemon thread |
| **Trained via** | `train_roboflow.py` ✅ | Must be sourced / trained separately |
| **In repo?** | ✅ `models/weapon_yolo.pt` | ❌ Not included — must be added |

> If either model file is missing, that detector silently enters a safe no-detection mode. The system will not crash.

---

## Risk Levels

The `RiskEngine` combines both detectors into one threat level per frame:

| Situation | Level |
|---|---|
| Nothing detected | `NO_RISK` |
| Fight probability above threshold | `MEDIUM_RISK` |
| Weapon detected, no fight | `HIGH_RISK` |
| Weapon + fight, **or** fight lasting ≥ 4 seconds | `CRITICAL` |

---

## Installation

**Prerequisites:** Python 3.9+, and a CUDA GPU is recommended (CPU fallback is automatic).

```bash
git clone https://github.com/your-org/smart_cctv.git
cd smart_cctv
pip install -r requirements.txt
```

---

## Training the Weapon Detection Model

The YOLO weapon model is trained on a dataset from [Roboflow Universe](https://universe.roboflow.com). A ready-made script is included.

**1. Get a Roboflow API key** from your account settings at roboflow.com.

**2. Find a weapon-detection dataset** on Roboflow Universe (search "weapon detection"). Note the workspace name, project name, and version number from its URL.

**3. Edit `train_roboflow.py`** with your details:

```python
ROBOFLOW_API_KEY = "your_key_here"
WORKSPACE_NAME   = "your-workspace"
PROJECT_NAME     = "your-project-name"
VERSION_NUMBER   = 1
```

**4. Run the training script:**

```bash
pip install roboflow
python train_roboflow.py
```

Training uses `yolov8n.pt` (YOLOv8 Nano) — fastest for CCTV use. Switch to `yolov8s.pt` in the script for better accuracy at the cost of speed. Training runs for 50 epochs at 640×640 resolution.

**5. Copy the output weights into the models folder:**

```bash
cp runs/detect/train/weights/best.pt models/weapon_yolo.pt
```

---

## Obtaining the Violence Detection Model

The `violence_cnn_lstm.pt` model is a **TorchScript CNN+LSTM** trained on a violence/fight video dataset. It is not included in this repository and must be sourced or trained separately.

Once you have it, place it here:

```
models/violence_cnn_lstm.pt
```

The model must accept input of shape `(1, 30, 3, 224, 224)` and output a single logit (before sigmoid).

---

## Configuration

All settings live in `config.py`:

```python
"camera": {
    "source": 0,                          # 0 = webcam, or an RTSP URL string
    "location_name": "AIML corridor 1st floor",
    "fps": 20,
},
"detection": {
    "weapon_conf_threshold": 0.60,        # YOLO minimum confidence
    "fight_prob_threshold": 0.49,         # Violence model minimum probability
    "min_fight_duration_sec": 4.0,        # Seconds of fighting before escalating to CRITICAL
    "sequence_length": 30,                # Frames per CNN+LSTM inference
},
"incidents": {
    "pre_event_sec": 5,                   # Seconds of pre-incident footage to capture
    "post_event_sec": 5,
},
```

For an RTSP camera, replace `source: 0` with the full stream URL:
```python
"source": "rtsp://user:pass@192.168.1.100:554/stream1"
```

---

## Running

**Web Dashboard** (recommended) — starts everything, accessible in the browser:

```bash
python web_app.py
```

Open `http://localhost:5000` — live feed, incident log, and analytics are all there.

**Headless mode** — detection + logging only, shows an OpenCV window locally:

```bash
python main.py
```

Press **`q`** to stop.

---

## Project Structure

```
smart_cctv/
├── main.py                     # Headless runner
├── web_app.py                  # Flask web app
├── config.py                   # All configuration
├── train_roboflow.py           # Script to train the weapon YOLO model
│
├── app/
│   ├── pipeline.py             # Core orchestration
│   ├── camera_stream.py        # Camera / RTSP capture (threaded)
│   ├── detectors/
│   │   ├── weapon_detector.py  # YOLOv8 + multi-frame confirmation
│   │   └── fight_detector.py   # CNN+LSTM + smoothing
│   ├── risk/
│   │   └── risk_engine.py      # Threat level logic
│   └── logging_utils/
│       ├── incident_logger.py  # CSV log + MP4 clip writer
│       └── video_buffer.py     # Rolling ring-buffer
│
├── models/
│   ├── weapon_yolo.pt          # ✅ Included (train via train_roboflow.py)
│   └── violence_cnn_lstm.pt    # ❌ Must be added manually
│
└── incidents/
    ├── clips/                  # Saved incident .mp4 files
    └── logs/incidents.csv      # Master incident log
```

---

## Requirements

```
flask>=2.0.0
numpy>=1.21.0
opencv-python>=4.5.3
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
```

> Install `roboflow` additionally only if you are running `train_roboflow.py`.
