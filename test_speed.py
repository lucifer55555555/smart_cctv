import time
import numpy as np
from app.detectors.fight_detector import FightDetector

def test_speed():
    d = FightDetector()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(30):
        d.update_frames(frame)

    start = time.time()
    for _ in range(10):
        d.update_frames(frame)
        d.predict_fight()
    
    elapsed = time.time() - start
    print(f"Time for 10 predictions: {elapsed:.2f}s, meaning {elapsed/10:.2f}s per prediction.")

if __name__ == "__main__":
    test_speed()
