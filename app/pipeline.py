from typing import Generator, Tuple, Optional

import numpy as np
import time

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
    End-to-end pipeline for campus safety:

    Campus camera -> weapon detector (YOLOv8) + fight detector (CNN-LSTM)
    -> risk engine -> incident logger -> annotated frame for display/streaming.
    """

    def __init__(self) -> None:
        cam_cfg = CONFIG["camera"]
        self.location_name = cam_cfg["location_name"]
        fps = cam_cfg["fps"]

        self.camera = CampusCameraStream()
        self.weapon_detector = WeaponDetector()
        self.fight_detector = FightDetector()

        inc_cfg = CONFIG["incidents"]
        max_buf_sec = inc_cfg["pre_event_sec"] + inc_cfg["post_event_sec"]
        self.video_buffer = VideoBuffer(max_seconds=max_buf_sec, fps=fps)
        self.risk_engine = RiskEngine()
        self.logger = IncidentLogger()

        # Alert state: avoid logging on every frame and ensure post-event frames exist.
        self._prev_risk_level: RiskLevel = RiskLevel.NO_RISK
        self._pending_event: Optional[dict] = None

    def frames(self) -> Generator[Tuple[bool, Optional[np.ndarray], Optional[RiskResult]], None, None]:
        """
        Generator yielding (success, annotated_frame, risk_result).
        """
        for ok, frame in self.camera.frames():
            if not ok or frame is None:
                yield False, None, None
                break

            self.video_buffer.add_frame(frame)

            weapon_dets = self.weapon_detector.detect_weapons(frame)
            self.fight_detector.update_frames(frame)
            fight_prob = self.fight_detector.predict_fight()

            risk = self.risk_engine.update_and_evaluate(
                fight_prob=fight_prob,
                weapon_detections=weapon_dets,
            )

            now = time.time()
            inc_cfg = CONFIG["incidents"]

            # Trigger an "incident episode" only on transition from NO_RISK -> risk,
            # or on escalation (e.g., MEDIUM -> CRITICAL). Clip is saved after post_event_sec.
            if risk.level != RiskLevel.NO_RISK:
                escalated = (
                    (self._prev_risk_level == RiskLevel.MEDIUM_RISK and risk.level in {RiskLevel.HIGH_RISK, RiskLevel.CRITICAL})
                    or (self._prev_risk_level == RiskLevel.HIGH_RISK and risk.level == RiskLevel.CRITICAL)
                )
                started = self._prev_risk_level == RiskLevel.NO_RISK

                if (started or escalated) and self._pending_event is None:
                    event_type = "campus_fight" if risk.num_weapons == 0 else "campus_weapon_or_fight"
                    event_time = now
                    self._pending_event = {
                        "event_time": event_time,
                        "event_type": event_type,
                        "risk": risk,
                        "ready_at": event_time + float(inc_cfg["post_event_sec"]),
                    }

            # If we have a pending event and enough post-event time has passed, save clip and log once.
            if self._pending_event is not None and now >= self._pending_event["ready_at"]:
                event_time = float(self._pending_event["event_time"])
                event_type = str(self._pending_event["event_type"])
                event_risk: RiskResult = self._pending_event["risk"]
                clip_path = self.logger.save_clip_from_buffer(self.video_buffer, event_time=event_time)
                self.logger.log_incident(
                    risk=event_risk,
                    event_type=event_type,
                    location_name=self.location_name,
                    clip_path=clip_path,
                )
                self._pending_event = None

            self._prev_risk_level = risk.level

            annotated = frame.copy()
            draw_weapon_boxes(annotated, weapon_dets)
            draw_risk_overlay(annotated, risk, self.location_name)

            yield True, annotated, risk

    def release(self) -> None:
        """Release camera when campus monitoring stops."""
        self.camera.release()

