# 🛡️ Smart CCTV – AI-Powered Campus Safety System

> Real-time surveillance system that detects **weapons and violent activity using YOLO models**, assigns risk levels, and automatically logs incidents.

---

## 🎯 Overview

Smart CCTV is an intelligent video surveillance system designed for **campus safety and security monitoring**.

It processes live camera feeds, detects threats using **two YOLO-based models**, evaluates risk, and automatically records incidents with video evidence.

---

## ⚙️ System Pipeline

```
STEP 0 — Load Models
        │
        ├── weapon_yolo.pt     (YOLOv8 – weapon detection)
        └── violence_yolo.pt   (YOLOv8 – fight detection)
                │
                ▼
STEP 1 — Camera Feed
        Webcam / RTSP Stream
                │
                ▼
STEP 2 — Detection (Parallel)
        │
        ├── Weapon Detector   → bounding boxes (guns, knives)
        └── Violence Detector → detects fight activity
                │
                ▼
STEP 3 — Risk Engine
        Combines detections into:
        NO_RISK / MEDIUM / HIGH / CRITICAL
                │
                ▼
STEP 4 — Incident Handling
        │
        ├── VideoBuffer → saves clip (pre + post event)
        └── IncidentLogger → logs to CSV
                │
                ▼
STEP 5 — Web Dashboard (Flask)
        Live Feed · Logs · Monitoring UI
```

---

## 🧠 Models Used

| Model              | Purpose                               | Type   |
| ------------------ | ------------------------------------- | ------ |
| `weapon_yolo.pt`   | Detect weapons (guns, knives, blades) | YOLOv8 |
| `violence_yolo.pt` | Detect fights / violent activity      | YOLOv8 |

> Both models run on **each frame**, enabling fast real-time inference.

---

## 🚀 Features

* 🎥 Live video streaming via Flask
* 🔫 Weapon detection using YOLOv8
* 🥊 Violence detection using YOLOv8
* ⚠️ Risk-level classification engine
* 🎬 Automatic incident recording (video clips)
* 🗂️ CSV-based incident logging
* 🔁 Continuous real-time processing pipeline

---

## 🏗️ Project Structure

```
smart_cctv/
├── web_app.py                  # Flask web interface
├── main.py                     # CLI runner
├── config.py                   # Configurations
├── train_roboflow.py           # Weapon training script
│
├── app/
│   ├── pipeline.py             # Core pipeline
│   ├── camera_stream.py        # Camera handling
│   │
│   ├── detectors/
│   │   ├── weapon_detector.py
│   │   └── fight_detector.py
│   │
│   ├── risk/
│   │   └── risk_engine.py
│   │
│   └── logging_utils/
│       ├── incident_logger.py
│       └── video_buffer.py
│
├── models/
│   ├── weapon_yolo.pt
│   └── violence_yolo.pt
│
└── incidents/
    ├── clips/
    └── logs/incidents.csv
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/smart_cctv.git
cd smart_cctv
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Both detections use **YOLOv8**, ensuring a unified and efficient pipeline.

---

### 🔫 Weapon Detection Training (Roboflow)

#### 1. Configure `train_roboflow.py`

```python
ROBOFLOW_API_KEY = "your_key"
WORKSPACE_NAME   = "workspace"
PROJECT_NAME     = "weapon-dataset"
VERSION_NUMBER   = 1
```

#### 2. Train

```bash
pip install roboflow
python train_roboflow.py
```

#### 3. Save Model

```bash
cp runs/detect/train/weights/best.pt models/weapon_yolo.pt
```

---

### 🥊 Violence Detection Training (YOLO)

#### Dataset Structure

```
dataset/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

#### Train Model

```bash
yolo detect train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640
```

#### Save Model

```bash
cp runs/detect/train/weights/best.pt models/violence_yolo.pt
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
"camera": {
    "source": 0,
    "fps": 20
},

"detection": {
    "weapon_conf_threshold": 0.6,
    "violence_conf_threshold": 0.5
},

"models": {
    "weapon_model": "models/weapon_yolo.pt",
    "violence_model": "models/violence_yolo.pt"
}
```

---

## ▶️ Running the System

### Web Dashboard (Recommended)

```bash
python web_app.py
```

Open:

```
http://127.0.0.1:5000
```

---

### CLI Mode

```bash
python main.py
```

---

## ⚠️ Risk Levels

| Condition        | Risk     |
| ---------------- | -------- |
| Nothing detected | NO_RISK  |
| Fight detected   | MEDIUM   |
| Weapon detected  | HIGH     |
| Weapon + Fight   | CRITICAL |

---

## 📊 Output

* 🎬 Video clips saved in:

```
incidents/clips/
```

* 📄 Logs saved in:

```
incidents/logs/incidents.csv
```

---

## 📦 Requirements

```
flask
numpy
opencv-python
torch
torchvision
ultralytics
```

---

## 🔮 Future Improvements

* Multi-camera support
* Email/SMS alerts
* Face recognition
* Cloud deployment
* Mobile app dashboard

---

## 👨‍💻 Author

Krish Agrawal

---

## 📄 License

For educational and research purposes.
