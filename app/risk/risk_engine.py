from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import time

from config import CONFIG


class RiskLevel(str, Enum):
    NO_RISK = "NO_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"     # fight only
    HIGH_RISK = "HIGH_RISK"         # weapon only
    CRITICAL = "CRITICAL"           # weapon + fight, or long fight


@dataclass
class RiskResult:
    level: RiskLevel
    fight_prob: float
    num_weapons: int
    reasons: List[str]


class RiskEngine:
    """
    Combines fight probability and weapon detections into a campus risk level.

    Designed for college corridors, hostel gates, and other campus spaces.
    """

    def __init__(self) -> None:
        cfg = CONFIG["detection"]
        self.fight_threshold = cfg["fight_prob_threshold"]
        self.min_fight_duration = cfg["min_fight_duration_sec"]
        self._fight_start_time: Optional[float] = None

    def update_and_evaluate(
        self,
        fight_prob: float,
        weapon_detections: List[Dict[str, Any]],
    ) -> RiskResult:
        now = time.time()
        num_weapons = len(weapon_detections)
        reasons: List[str] = []

        # Track continuous fight duration in campus scene.
        if fight_prob >= self.fight_threshold:
            if self._fight_start_time is None:
                self._fight_start_time = now
            fight_duration = now - self._fight_start_time
        else:
            fight_duration = 0.0
            self._fight_start_time = None

        has_fight = fight_prob >= self.fight_threshold
        has_long_fight = fight_duration >= self.min_fight_duration
        has_weapon = num_weapons > 0

        if not has_fight and not has_weapon:
            level = RiskLevel.NO_RISK
        elif has_weapon and has_fight:
            level = RiskLevel.CRITICAL
            reasons.append("Weapon and fight detected in campus camera")
        elif has_weapon:
            level = RiskLevel.HIGH_RISK
            labels = ", ".join(set(d["label"] for d in weapon_detections))
            reasons.append(f"Weapon detected ({labels}) in campus camera")
        elif has_long_fight:
            level = RiskLevel.CRITICAL
            reasons.append("Prolonged fight detected in campus camera")
        else:
            level = RiskLevel.MEDIUM_RISK
            reasons.append("Fight detected in campus camera")

        if has_fight:
            reasons.append(f"fight_prob={fight_prob:.2f}")
        if has_weapon:
            reasons.append(f"weapons={num_weapons}")

        return RiskResult(
            level=level,
            fight_prob=fight_prob,
            num_weapons=num_weapons,
            reasons=reasons,
        )

