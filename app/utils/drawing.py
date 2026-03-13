from typing import List, Dict, Any

import cv2

from app.risk.risk_engine import RiskResult, RiskLevel


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
    Draw text overlay with risk level and campus camera location.
    """
    color = _risk_color(risk.level)
    h, w = frame.shape[:2]

    title = f"{location_name} - {risk.level.value}"
    details = ", ".join(risk.reasons) if risk.reasons else ""

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
    if details:
        cv2.putText(
            frame,
            details,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

