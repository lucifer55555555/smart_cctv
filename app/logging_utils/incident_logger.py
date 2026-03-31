import csv
import os
import time
from typing import Optional

import cv2

from config import CONFIG, ensure_directories, timestamp_str
from app.logging_utils.video_buffer import VideoBuffer
from app.risk.risk_engine import RiskResult


class IncidentLogger:
    """
    Logs campus safety incidents and saves short video clips.

    Each alert from a corridor/hostel camera is written to a CSV log
    and optionally saved as an MP4 clip for later review.
    """

    def __init__(self) -> None:
        ensure_directories()
        inc_cfg = CONFIG["incidents"]
        self.log_csv = inc_cfg["log_csv"]
        self.clips_dir = inc_cfg["clips_dir"]
        self.pre_event_sec = inc_cfg["pre_event_sec"]
        self.post_event_sec = inc_cfg["post_event_sec"]

        # Smart Alerting: prevents spamming alerts for the same incident
        self.COOLDOWN_SEC = 30.0
        self._last_alert_time: float = 0.0

        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "risk_level", "event_type", "location", "clip_path", "details"]
                )

    def log_incident(
        self,
        risk: RiskResult,
        event_type: str,
        location_name: str,
        clip_path: Optional[str],
    ) -> bool:
        """
        Append one incident row to the CSV log for the campus.
        Returns True if logged, False if suppressed by cooldown.
        """
        now = time.time()
        if now - self._last_alert_time < self.COOLDOWN_SEC:
            # Still in cooldown; don't log a new row for the same continuous event
            return False

        ts = timestamp_str()
        details = "; ".join(risk.reasons)
        with open(self.log_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts, risk.level.value, event_type, location_name, clip_path or "", details])
        
        self._last_alert_time = now
        print(f"[IncidentLogger] Logged incident at {ts}: {risk.level.value} ({event_type})")
        return True

    def save_clip_from_buffer(
        self,
        buffer: VideoBuffer,
        event_time: Optional[float] = None,
    ) -> Optional[str]:
        """
        Save an MP4 clip from frames around event_time using the ring buffer.
        """
        if event_time is None:
            event_time = time.time()

        frames = buffer.get_frames_around(
            event_time,
            self.pre_event_sec,
            self.post_event_sec,
        )
        if not frames:
            print("[IncidentLogger] No frames to save for clip.")
            return None

        height, width = frames[0].shape[:2]
        fps = CONFIG["camera"]["fps"]

        filename = f"incident_{int(event_time)}.mp4"
        path = os.path.join(self.clips_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)

        writer.release()
        print(f"[IncidentLogger] Saved incident clip to {path}")
        return path

