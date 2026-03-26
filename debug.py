import traceback
from app.detectors.fight_detector import FightDetector
from app.detectors.weapon_detector import WeaponDetector
import numpy as np

def run_debug():
    try:
        # Test FightDetector
        print("--- Testing FightDetector ---")
        f_detector = FightDetector()
        if f_detector.model is not None:
            for _ in range(f_detector.seq_len):
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                f_detector.update_frames(frame)
            
            prob = f_detector.predict_fight()
            print("FightDetector Prediction Success:", prob)
        else:
            print("FightDetector model failed to load.")

        # Test WeaponDetector
        print("\n--- Testing WeaponDetector ---")
        w_detector = WeaponDetector()
        if w_detector.model is not None:
            frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            dets = w_detector.detect_weapons(frame)
            print("WeaponDetector Detection Success (Dummy data):", dets)
        else:
            print("WeaponDetector model failed to load (possibly ultralytics missing or weights missing).")

    except Exception as e:
        print("Exception caught!")
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()

