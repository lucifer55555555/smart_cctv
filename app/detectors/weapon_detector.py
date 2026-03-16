from typing import List, Dict, Any

import numpy as np
import os

from config import CONFIG

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    _ULTRALYTICS_AVAILABLE = False


class WeaponDetector:
    """
    YOLOv8-based weapon detector for college campus cameras.

    If YOLO or weights are not available, this falls back to a safe dummy mode
    that returns no detections so the corridor monitoring pipeline still runs.
    """

    def __init__(self) -> None:
        self.conf_threshold = CONFIG["detection"]["weapon_conf_threshold"]
        self.model_path = CONFIG["models"]["weapon_yolo_weights"]
        self.device = CONFIG["models"]["device"]
        self.model = None

        if not _ULTRALYTICS_AVAILABLE:
            print("[WeaponDetector] ultralytics not installed; running in dummy mode.")
            return

        if not os.path.exists(self.model_path):
            print(f"[WeaponDetector] Weapon YOLO weights not found: {self.model_path} (SAFE mode)")
            self.model = None
            return

        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"[WeaponDetector] Loaded YOLO weights from {self.model_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[WeaponDetector] Could not load YOLO model: {e}")
            self.model = None

    def detect_weapons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 on a BGR frame from a campus camera.

        Returns list of detections:
        [
          {\"bbox\": (x1, y1, x2, y2), \"label\": \"gun\", \"confidence\": 0.85},
          ...
        ]
        """
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WeaponDetector] Inference failed: {e}")
            return []

        detections: List[Dict[str, Any]] = []

        for r in results:
            boxes = r.boxes
            names = r.names  # class_id -> label

            for box in boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label,
                        "confidence": conf,
                    }
                )

        # Filter to gun/weapon-like labels (depends on your trained classes).
        weapon_detections = [
            d
            for d in detections
            if any(k in d["label"].lower() for k in ["gun", "weapon", "pistol", "rifle"])
        ]
        return weapon_detections

