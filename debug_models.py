#!/usr/bin/env python3
"""Debug script to check why distraction/drowsiness predictions are NULL."""

import cv2
import numpy as np
from pathlib import Path
from src.main import DriverFatigueMonitoringPipeline
from src.config import *

print("Checking model loading and inference...")
print(f"Models directory: {MODELS_DIR}")
print(f"Models exist:")
print(f"  Eye: {EYE_MODEL_PATH.exists()}")
print(f"  Yawn: {YAWN_MODEL_PATH.exists()}")
print(f"  Distraction: {DISTRACTION_MODEL_PATH.exists()}")
print(f"  Drowsiness: {DROWSINESS_MODEL_PATH.exists()}")

# Initialize pipeline
pipeline = DriverFatigueMonitoringPipeline(
    str(EYE_MODEL_PATH),
    str(YAWN_MODEL_PATH),
    str(DISTRACTION_MODEL_PATH),
    str(DROWSINESS_MODEL_PATH)
)

print(f"\nPipeline state:")
print(f"  Safe mode active: {pipeline.safe_mode_active}")
print(f"  Enable distraction alerts: {pipeline.enable_distraction_alerts}")
print(f"  Enable drowsiness alerts: {pipeline.enable_drowsiness_alerts}")
print(f"  Distraction model loaded: {pipeline.distraction_model is not None}")
print(f"  Drowsiness model loaded: {pipeline.drowsiness_model is not None}")
print(f"  Model quality: {pipeline.model_quality}")
print(f"  SHOW_UNVALIDATED_MODEL_SIGNALS: {SHOW_UNVALIDATED_MODEL_SIGNALS}")

# Test with a few frames
video_path = Path('dataset/evaluation_videos/1-FemaleNoGlasses-Normal.avi')
cap = cv2.VideoCapture(str(video_path))

print(f"\nProcessing frames to debug inference...")
print(f"Frame | Eye | Yawn | Distraction | Drowsy | Alert")
print("-" * 60)

for i in range(210):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (480, 360))
    _, state_dict = pipeline.process_frame(frame)
    
    # Only print after calibration and at specific intervals
    if i >= 150 and i % 20 == 0:
        distraction = state_dict.get('distraction_class') or 'NULL'
        drowsy = state_dict.get('drowsy_probability')
        drowsy_str = f"{drowsy*100:.0f}%" if drowsy is not None else "NULL"
        alert = "YES" if state_dict.get('should_alert') else "NO"
        
        print(f"{i:3d}  | {state_dict.get('eye_state')} | {state_dict.get('is_yawning')} | {distraction:12} | {drowsy_str:6} | {alert}")

cap.release()

print("\nIf distraction and drowsy are still NULL, the models may not be making predictions.")
print("Check that:")
print("  1. Models exist and are loadable")
print("  2. Inference is being called (not skipped by frame interval)")
print("  3. Support signals are being detected") 
print("  4. SHOW_UNVALIDATED_MODEL_SIGNALS is enabled")
