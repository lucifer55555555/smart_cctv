from typing import List, Dict, Any

import cv2
import time

from app.risk.risk_engine import RiskResult, RiskLevel
from config import CONFIG


def _risk_color(level: RiskLevel) -> tuple[int, int, int]:
    """
    BGR colors for different campus risk levels.
    """
    if level == RiskLevel.CRITICAL:
        return (0, 0, 255)  # red
    if level == RiskLevel.HIGH_RISK:
        return (0, 128, 255)  # orange
    if level == RiskLevel.MEDIUM_RISK:
        return (0, 255, 255)  # yellow
    return (0, 255, 0)  # green


def draw_weapon_boxes(frame, detections: List[Dict[str, Any]]) -> None:
    """
    Draw bounding boxes around detected weapons in the campus frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def draw_risk_overlay(
    frame,
    risk: RiskResult,
    location_name: str,
) -> None:
    """
    Draw text overlay with risk level, campus camera location, and
    a prominent flashing banner for active fight / critical incidents.
    """
    color = _risk_color(risk.level)
    h, w = frame.shape[:2]

    title = f"{location_name} - {risk.level.value}"
    details = ", ".join(risk.reasons) if risk.reasons else ""

    # Location + risk level header
    cv2.putText(
        frame,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    # Fight probability indicator bar
    if risk.fight_prob > 0.0:
        thresh = CONFIG["detection"]["fight_prob_threshold"]
        bar_label = f"Fight: {risk.fight_prob:.0%} (Thr: {thresh:.0%})"
        bar_w = int((w - 20) * min(risk.fight_prob, 1.0))

        # Background bar (dark)
        cv2.rectangle(frame, (10, 35), (w - 10, 50), (40, 40, 40), -1)
        # Fill bar (color matches risk level)
        bar_color = (0, 255, 255) if risk.fight_prob < 0.7 else (0, 0, 255)
        cv2.rectangle(frame, (10, 35), (10 + bar_w, 50), bar_color, -1)
        cv2.putText(
            frame, bar_label, (14, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Prominent flashing banner for MEDIUM_RISK and above (fight/violence)
    if risk.level in (RiskLevel.MEDIUM_RISK, RiskLevel.HIGH_RISK, RiskLevel.CRITICAL):
        # Flash effect: banner visible for ~0.7s out of every 1s
        show_banner = (int(time.time() * 2) % 2 == 0)
        if show_banner:
            banner_text = "⚠ FIGHT DETECTED" if risk.num_weapons == 0 else "⚠ WEAPON + FIGHT"
            if risk.level == RiskLevel.HIGH_RISK and risk.num_weapons > 0:
                banner_text = "⚠ WEAPON DETECTED"

            # Semi-transparent red/orange banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(
                frame,
                banner_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Reason details at bottom (above banner if banner is visible)
    if details:
        detail_y = h - 70 if risk.level != RiskLevel.NO_RISK else h - 10
        cv2.putText(
            frame,
            details,
            (10, detail_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
