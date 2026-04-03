import threading
import time
from typing import Generator, Tuple, Optional, List, Dict, Any

import numpy as np

from config import CONFIG
from app.camera_stream import CampusCameraStream
from app.detectors.weapon_detector import WeaponDetector
from app.detectors.fight_detector import FightDetector
from app.logging_utils.video_buffer import VideoBuffer
from app.logging_utils.incident_logger import IncidentLogger
from app.risk.risk_engine import RiskEngine, RiskResult, RiskLevel
from app.utils.drawing import draw_weapon_boxes, draw_risk_overlay


class CampusSafetyPipeline:
    """
    End-to-end pipeline for campus safety using an asynchronous architecture:
    - Threaded Camera Stream (lowest latency)
    - Background Detection Worker (non-blocking)
    - Single RiskEngine to avoid conflicting evaluations
    """

    def __init__(self) -> None:
        cam_cfg = CONFIG["camera"]
        self.location_name = cam_cfg["location_name"]
        self.fps = cam_cfg["fps"]

        self.camera = CampusCameraStream()
        self.weapon_detector = WeaponDetector()
        self.fight_detector = FightDetector()

        inc_cfg = CONFIG["incidents"]
        max_buf_sec = inc_cfg["pre_event_sec"] + inc_cfg["post_event_sec"]
        self.video_buffer = VideoBuffer(max_seconds=max_buf_sec, fps=self.fps)
        self.risk_engine = RiskEngine()
        self.logger = IncidentLogger()

        self._is_running = True
        self._lock = threading.Lock()
        self._latest_unprocessed_frame: Optional[np.ndarray] = None

        # Shared state: last fight probability from the background worker
        self._last_fight_prob: float = 0.0
        # Shared state: last weapon detections from the sync loop
        self._last_weapon_dets: List[Dict[str, Any]] = []

        self._prev_risk_level = RiskLevel.NO_RISK
        self._pending_event = None
        self._frame_count = 0

        # Smart Alerting: stores the last incident to be fetched by the web UI
        self.latest_incident: Optional[Dict[str, Any]] = None

        # Background thread for HEAVY detections (violence)
        self._worker_thread = threading.Thread(target=self._background_fight_worker, daemon=True)
        self._worker_thread.start()

    def _background_fight_worker(self) -> None:
        """Runs heavy AI (violence CNN-LSTM) in background. Only updates fight_prob."""
        while self._is_running:
            frame_to_process = None
            with self._lock:
                if self._latest_unprocessed_frame is not None:
                    frame_to_process = self._latest_unprocessed_frame.copy()

            if frame_to_process is not None:
                self.fight_detector.update_frames(frame_to_process)
                fight_prob = self.fight_detector.predict_fight()

                with self._lock:
                    self._last_fight_prob = fight_prob

            # No sleep here: process as fast as the GPU allows to capture high-speed motion
            time.sleep(0.01) 

    def _compute_risk(self, weapon_dets: List[Dict[str, Any]], fight_prob: float) -> RiskResult:
        """
        Single point of risk evaluation — avoids conflicting dual evaluations.
        Combines the latest fight probability and weapon detections.
        """
        return self.risk_engine.update_and_evaluate(
            fight_prob=fight_prob,
            weapon_detections=weapon_dets,
        )

    def _handle_incident_logic(self, risk: RiskResult) -> None:
        """Internal logic for triggering clips and logs based on risk level."""
        now = time.time()
        inc_cfg = CONFIG["incidents"]

        if risk.level != RiskLevel.NO_RISK:
            escalated = (
                (self._prev_risk_level == RiskLevel.MEDIUM_RISK and risk.level in {RiskLevel.HIGH_RISK, RiskLevel.CRITICAL})
                or (self._prev_risk_level == RiskLevel.HIGH_RISK and risk.level == RiskLevel.CRITICAL)
            )
            started = self._prev_risk_level == RiskLevel.NO_RISK

            if (started or escalated) and self._pending_event is None:
                # Determine event type based on CURRENT detections
                has_weapon = risk.num_weapons > 0
                has_fight = risk.fight_prob >= self.risk_engine.fight_threshold

                if has_weapon and has_fight:
                    event_type = "campus_violence_with_weapon"
                elif has_weapon:
                    event_type = "campus_weapon_alert"
                elif has_fight:
                    event_type = "campus_violence_alert"
                else:
                    event_type = "campus_general_alert"

                self._pending_event = {
                    "event_time": now,
                    "event_type": event_type,
                    "risk": risk,
                    "ready_at": now + float(inc_cfg["post_event_sec"]),
                }

        if self._pending_event is not None and now >= self._pending_event["ready_at"]:
            event_time = float(self._pending_event["event_time"])
            event_type = str(self._pending_event["event_type"])
            event_risk: RiskResult = self._pending_event["risk"]
            clip_path = self.logger.save_clip_from_buffer(self.video_buffer, event_time=event_time)
            logged = self.logger.log_incident(
                risk=event_risk,
                event_type=event_type,
                location_name=self.location_name,
                clip_path=clip_path,
            )

            if logged:
                # Update latest_incident for the web UI/Polling
                self.latest_incident = {
                    "timestamp": time.time(),
                    "level": event_risk.level.value,
                    "event_type": event_type,
                    "details": "; ".join(event_risk.reasons)
                }

            self._pending_event = None

        self._prev_risk_level = risk.level

    def start(self) -> None:
        """Start monitoring."""
        if not self._is_running:
            self.camera = CampusCameraStream()
            self._is_running = True

    def stop(self) -> None:
        """Stop/Pause monitoring."""
        if self._is_running:
            self.camera.release()
            self._is_running = False
            self._pending_event = None

    def is_running(self) -> bool:
        return self._is_running

    def frames(self) -> Generator[Tuple[bool, Optional[np.ndarray], Optional[RiskResult]], None, None]:
        """
        Yield (success, annotated_frame, risk_result).
        HYBRID: Weapon detection is SYNC, Violence is ASYNC via background thread.
        Risk evaluation happens ONCE per frame using both inputs.
        """
        while True:
            if not self._is_running:
                time.sleep(0.5)
                yield False, None, None
                continue

            for ok, frame in self.camera.frames():
                if not self._is_running:
                    break
                if not ok or frame is None:
                    yield False, None, None
                    continue

                self._frame_count += 1
                self.video_buffer.add_frame(frame)

                # 1. Run weapon detection on EVERY frame for maximum responsiveness
                weapon_dets = self.weapon_detector.detect_weapons(frame, fight_prob=self._last_fight_prob)
                with self._lock:
                    self._last_weapon_dets = weapon_dets
                    self._latest_unprocessed_frame = frame
                    fight_prob = self._last_fight_prob

                # Debug: log fight probability periodically
                if self._frame_count % 30 == 0:
                    print(f"[Pipeline DEBUG] frame={self._frame_count} fight_prob={fight_prob:.4f} weapons={len(weapon_dets)}")


                # 2. SINGLE risk evaluation combining both detectors
                risk = self._compute_risk(weapon_dets, fight_prob)

                self._handle_incident_logic(risk)

                annotated = frame.copy()
                draw_weapon_boxes(annotated, weapon_dets)
                draw_risk_overlay(annotated, risk, self.location_name)

                yield True, annotated, risk

    def release(self) -> None:
        self._is_running = False
        self.camera.release()
