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

    # A weapon must be detected in at least this many of last N frames
    CONFIRM_WINDOW = 8         # Increased for better stability
    CONFIRM_MIN_HITS = 2       # Base hits required
 
    # Minimum bounding box area (pixels) — tiny boxes in low-conf are likely noise
    MIN_BBOX_AREA = 1500       # Raised again to filter out smaller false-positive limb shapes

    def __init__(self) -> None:
        self.conf_threshold = CONFIG["detection"]["weapon_conf_threshold"]
        self.model_path = CONFIG["models"]["weapon_yolo_weights"]
        self.device = CONFIG["models"]["device"]
        self.model = None

        # Rolling history of raw detections for confirmation logic
        self._detection_history: Deque[List[Dict[str, Any]]] = deque(maxlen=self.CONFIRM_WINDOW)
        # Store last seen non-empty detections to bridge frames while confirmed
        self._last_valid_dets: List[Dict[str, Any]] = []

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

    def _raw_detect(self, frame: np.ndarray, fight_prob: float = 0.0) -> List[Dict[str, Any]]:
        """
        Run YOLO inference and return raw detections (before confirmation).
        """
        if self.model is None:
            return []

        # Use the standard model threshold (0.60 as recommended)
        current_conf = self.conf_threshold

        try:
            results = self.model.predict(
                source=frame,
                conf=current_conf,
                device=self.device,
                verbose=False,
                imgsz=640,    # Higher resolution for better small-object detection
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
                bbox_area = (x2 - x1) * (y2 - y1)

                # DEBUG: Log ALL raw detections so we can see what the model outputs
                print(f"[WeaponDetector RAW] cls={cls_id} label='{label}' conf={conf:.3f} area={bbox_area:.0f}")

                # Filter by weapon-like labels
                weapon_keywords = ["gun", "weapon", "pistol", "rifle", "knife",
                                   "dagger", "sharp", "handgun", "blade",
                                   "machete", "sword", "firearm", "revolver"]
                if not any(k in label.lower() for k in weapon_keywords):
                    continue

                # Filter by minimum bounding box area
                if bbox_area < self.MIN_BBOX_AREA:
                    print(f"[WeaponDetector] Skipped {label} — bbox too small ({bbox_area:.0f} < {self.MIN_BBOX_AREA})")
                    continue

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label,
                        "confidence": conf,
                    }
                )

        return detections

    def detect_weapons(self, frame: np.ndarray, fight_prob: float = 0.0) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 on a BGR frame with multi-frame confirmation.

        A weapon is only reported if it has been detected in at least
        CONFIRM_MIN_HITS of the last CONFIRM_WINDOW frames.
        This eliminates single-frame false positives from fights.
        """
        raw_dets = self._raw_detect(frame, fight_prob=fight_prob)
        self._detection_history.append(raw_dets)

        # Label-Specific Stability Rules:
        # - Knife: 2 hits (fast/responsive for your primary use-case)
        # - Others (Gun/Pistol): 4 hits (stronger filter for fight-limb false positives)
        
        def get_required_hits(label: str) -> int:
            if "knife" in label.lower() or "blade" in label.lower():
                return 2
            return 4

        # Process each current detection against its specific history requirement
        final_dets = []
        for d in raw_dets:
            label = d["label"]
            req = get_required_hits(label)
            
            # Count how many of the last N frames had THIS specific label (or similar)
            hits = sum(1 for frame_dets in self._detection_history 
                       if any(label.lower() in fd["label"].lower() or fd["label"].lower() in label.lower() for fd in frame_dets))
            
            if hits >= req:
                final_dets.append(d)
                print(f"[WeaponDetector CONFIRMED] {label} ({d['confidence']:.2f}) with {hits}/{req} hits")

        if final_dets:
            self._last_valid_dets = final_dets
            return final_dets
        else:
            # PERSISTENCE: If we were confirmed in the very recent past, stay alert for a moment
            # This bridges tiny single-frame gaps to keep the red box stable
            confirm_frames = sum(1 for frame_dets in self._detection_history if len(frame_dets) > 0)
            if confirm_frames >= 3 and self._last_valid_dets:
                print(f"[WeaponDetector PERSISTED] Showing last seen {len(self._last_valid_dets)} box(es) — still confirmed ({confirm_frames} total hits)")
                return self._last_valid_dets
            else:
                self._last_valid_dets = []
                if raw_dets:
                     print(f"[WeaponDetector REJECTED] {len(raw_dets)} detection(s) — failed hit requirements")
                return []
