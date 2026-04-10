import torch
import os
from ultralytics import YOLO
from config import CONFIG

def check_models():
    print("--- Checking Weapon Model ---")
    weapon_path = CONFIG["models"]["weapon_yolo_weights"]
    if os.path.exists(weapon_path):
        model = YOLO(weapon_path)
        print(f"Names: {model.names}")
    else:
        print("Weapon model not found")

    print("\n--- Checking Violence Model ---")
    violence_path = CONFIG["models"]["violence_model_weights"]
    if os.path.exists(violence_path):
        try:
            device = CONFIG["models"]["device"]
            model = torch.jit.load(violence_path, map_location=device)
            # Create a dummy input (1, 30, 3, 224, 224)
            dummy_input = torch.randn(1, 30, 3, 224, 224).to(device)
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Sample output: {output}")
        except Exception as e:
            print(f"Error loading/running violence model: {e}")
    else:
        print("Violence model not found")

if __name__ == "__main__":
    check_models()
