import torch
import numpy as np
import cv2
from app.detectors.fight_detector import FightDetector
from config import CONFIG

def test_fight_raw():
    detector = FightDetector()
    if detector.model is None:
        print("Model not loaded")
        return

    # Create 30 frames of "random motion" (actually just noise)
    for _ in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detector.update_frames(frame)

    prob = detector.predict_fight()
    print(f"Prob for noise: {prob}")
    print(f"Last raw prob: {detector.last_raw_prob}")

if __name__ == "__main__":
    test_fight_raw()
