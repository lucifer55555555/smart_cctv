import os
from roboflow import Roboflow
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Get your API key from your Roboflow account settings
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

# 2. Get the workspace and project name from the dataset URL. 
# For example, if the URL is: https://universe.roboflow.com/some-workspace/violence-detection-xyz
WORKSPACE_NAME = "some-workspace"
PROJECT_NAME = "violence-detection-xyz"
VERSION_NUMBER = 1 # Usually 1 unless specified 

def train_custom_model():
    print("Initializing Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    print(f"Downloading dataset '{PROJECT_NAME}'...")
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    dataset = project.version(VERSION_NUMBER).download("yolov8")
    
    # The dataset download creates a folder with a data.yaml file inside
    yaml_path = os.path.join(dataset.location, "data.yaml")
    
    print(f"Dataset downloaded successfully to: {dataset.location}")
    print("Initializing YOLOv8 Nano model for training...")
    
    # Initialize a blank YOLOv8 nano model (fastest inference for CCTV)
    # You can change to 'yolov8s.pt' for better accuracy at the cost of speed
    model = YOLO("yolov8n.pt") 
    
    print("Starting training process. This may take a while depending on your GPU...")
    # Train the model on the dataset
    # Adjust epochs depending on the dataset size (e.g., 50-100)
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device="0" # Use "cpu" if you don't have a GPU, but it will be very slow
    )
    
    print("Training Complete!")
    print(f"Your new specialized model weights are saved in: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_custom_model()
