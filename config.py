import os
import torch
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG = {
    "camera": {
        # 0 for default laptop webcam, or RTSP URL for a real campus CCTV
        "source": 0,
        "location_name": "AIML corridor 1st floor",
        "frame_width": 640,
        "frame_height": 480,


        "fps": 20,
    },
    "detection": {
        "weapon_conf_threshold": 0.50,
        "fight_prob_threshold": 0.80, # Even higher to stop false triggers from over-sensitive YOLO model
        "min_fight_duration_sec": 4.0,
    },
    "models": {
        "weapon_yolo_weights": os.path.join(BASE_DIR, "models", "weapon_yolo.pt"),
        "violence_model_weights": os.path.join(BASE_DIR, "models", "violence_yolo.pt"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "incidents": {
        "base_dir": os.path.join(BASE_DIR, "incidents"),
        "clips_dir": os.path.join(BASE_DIR, "incidents", "clips"),
        "logs_dir": os.path.join(BASE_DIR, "incidents", "logs"),
        "log_csv": os.path.join(BASE_DIR, "incidents", "logs", "incidents.csv"),
        "pre_event_sec": 5,
        "post_event_sec": 5,
    },
}


def ensure_directories() -> None:
    """Make sure incident folders exist for campus recordings."""
    inc = CONFIG["incidents"]
    os.makedirs(inc["clips_dir"], exist_ok=True)
    os.makedirs(inc["logs_dir"], exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)


def timestamp_str() -> str:
    """Human-readable timestamp for campus incident logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

