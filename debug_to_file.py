import traceback
from app.detectors.fight_detector import FightDetector
import numpy as np

def run_debug():
    try:
        f_detector = FightDetector()
        for _ in range(f_detector.seq_len):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            f_detector.update_frames(frame)
        
        prob = f_detector.predict_fight()
        print("Success:", prob)
    except Exception as e:
        with open("debug_error.log", "w") as f:
            f.write(traceback.format_exc())
        print("Error written to debug_error.log")

if __name__ == "__main__":
    run_debug()
