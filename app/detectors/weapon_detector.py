from collections import deque
from typing import Deque, List, Dict, Any

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

    Includes multi-frame confirmation to prevent false positives during
    fights (e.g., fists/limbs misclassified as weapons).
    """

    # A weapon must be detected in at least this many of the last N frames
    CONFIRM_WINDOW = 5       # look at last 5 detection cycles
    CONFIRM_MIN_HITS = 3     # need 3/5 hits to confirm a weapon is real

    # Minimum bounding box area (pixels) — tiny boxes are likely noise
    MIN_BBOX_AREA = 1500

    def __init__(self) -> None:
        self.conf_threshold = CONFIG["detection"]["weapon_conf_threshold"]
        self.model_path = CONFIG["models"]["weapon_yolo_weights"]
        self.device = CONFIG["models"]["device"]
        self.model = None

        # Rolling history of raw detections for confirmation logic
        self._detection_history: Deque[List[Dict[str, Any]]] = deque(maxlen=self.CONFIRM_WINDOW)

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

    def _raw_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO inference and return raw detections (before confirmation).
        """
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
                imgsz=320,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WeaponDetector] Inference failed: {e}")
            return []

        detections: List[Dict[str, Any]] = []

        for r in results:
            boxes = r.boxes
            names = r.names

            for box in boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Filter by weapon-like labels
                if not any(k in label.lower() for k in ["gun", "weapon", "pistol", "rifle", "knife", "dagger", "sharp", "handgun"]):
                    continue

                # Filter by minimum bounding box area
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.MIN_BBOX_AREA:
                    continue

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label,
                        "confidence": conf,
                    }
                )

        return detections

    def detect_weapons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 on a BGR frame with multi-frame confirmation.

        A weapon is only reported if it has been detected in at least
        CONFIRM_MIN_HITS of the last CONFIRM_WINDOW frames.
        This eliminates single-frame false positives from fights.
        """
        raw_dets = self._raw_detect(frame)
        self._detection_history.append(raw_dets)

        if len(self._detection_history) < self.CONFIRM_MIN_HITS:
            # Not enough history yet — be conservative, don't report
            return []

        # Count how many recent frames had ANY weapon detection
        frames_with_weapons = sum(
            1 for dets in self._detection_history if len(dets) > 0
        )

        if frames_with_weapons >= self.CONFIRM_MIN_HITS:
            # Confirmed: weapons detected consistently. Return current detections.
            if raw_dets:
                for d in raw_dets:
                    print(f"[WeaponDetector CONFIRMED] {d['label']} ({d['confidence']:.2f}) over {frames_with_weapons}/{len(self._detection_history)} frames")
            return raw_dets
        else:
            # Not enough consistency — likely false positive from fight motion
            if raw_dets:
                print(f"[WeaponDetector REJECTED] {len(raw_dets)} detection(s) — only {frames_with_weapons}/{len(self._detection_history)} frames consistent")
            return []
