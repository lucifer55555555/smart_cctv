import cv2
import urllib.request
import numpy as np
import time
import urllib.error

from app.detectors.weapon_detector import WeaponDetector
from app.detectors.fight_detector import FightDetector
from app.risk.risk_engine import RiskEngine

def run_tests():
    # 1. Test Weapon Detector with a real weapon image
    print("\n--- [1] Testing Weapon Detector ---")
    w_detector = WeaponDetector()
    
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Glock_19_Gen4.jpg/800px-Glock_19_Gen4.jpg"
        print(f"Downloading test weapon image from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        print(f"Image downloaded. Dimensions: {img.shape}")
        
        dets = w_detector.detect_weapons(img)
        print(f"Detected {len(dets)} weapons.")
        for d in dets:
            print(f" - {d['label']} with confidence {d['confidence']:.2f}")
    except Exception as e:
        print(f"Failed to download or process test image: {e}")

    # 2. Test Fight Detector with synthetic sequence
    print("\n--- [2] Testing Fight Detector ---")
    f_detector = FightDetector()
    print(f"Smoothing window: {f_detector.SMOOTHING_WINDOW}")
    
    # Run a few predictions to test smoothing accumulation
    for i in range(5):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        p = f_detector.predict_fight(frame)
        print(f"  Prediction {i+1}: raw={f_detector.last_raw_prob:.4f} smooth={f_detector.last_smoothed_prob:.4f}")
    
    # Test reset
    f_detector.reset()
    print(f"After reset: prob={f_detector.last_smoothed_prob:.4f}")

    # 3. Test Risk Engine Logic
    print("\n--- [3] Testing Risk Engine Logic ---")
    engine = RiskEngine()
    
    # Test no risk
    res1 = engine.update_and_evaluate(fight_prob=0.1, weapon_detections=[])
    print("Test 1 (No incidents):", res1.level)
    
    # Test medium risk (fight only)
    res2 = engine.update_and_evaluate(fight_prob=0.8, weapon_detections=[])
    print("Test 2 (Fight only):", res2.level)
    
    # Test high risk (weapon only)
    res3 = engine.update_and_evaluate(fight_prob=0.1, weapon_detections=[{"label": "gun", "confidence": 0.9}])
    print("Test 3 (Weapon only):", res3.level)
    
    # Test critical risk (weapon + fight)
    res4 = engine.update_and_evaluate(fight_prob=0.9, weapon_detections=[{"label": "gun", "confidence": 0.9}])
    print("Test 4 (Weapon + Fight):", res4.level)
    
    # Test prolonged fight escalation
    print("\nTesting prolonged fight (escalates to CRITICAL over time):")
    engine = RiskEngine()
    for i in range(5):
        res = engine.update_and_evaluate(fight_prob=0.8, weapon_detections=[])
        print(f"  Sec {i}: {res.level}")
        time.sleep(0.5)

if __name__ == "__main__":
    run_tests()
