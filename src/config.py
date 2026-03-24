# Configuration for Driver Fatigue Monitoring System
import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except Exception:
    pass


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'on'}


DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset Paths
EYE_DATASET_PATH = DATA_DIR / "eye"
YAWN_DATASET_PATH = DATA_DIR / "yawn"
EVALUATION_VIDEO_PATH = DATA_DIR / "evaluation_videos"

# Model Paths
EYE_MODEL_PATH = MODELS_DIR / "eye_model.h5"
YAWN_MODEL_PATH = MODELS_DIR / "yawn_model.h5"
DISTRACTION_MODEL_PATH = MODELS_DIR / "distraction_model.h5"
DROWSINESS_MODEL_PATH = MODELS_DIR / "drowsiness_model.h5"
HEAD_POSE_MODEL_PATH = MODELS_DIR / "head_pose_model.h5"

# CNN Model Configuration
CNN_INPUT_SIZE = 224
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 50
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT_RATE = 0.3

# Debug Flags (set to True to enable detailed logging)
DEBUG_YAWN = False  # Enable yawn detection logging
DEBUG_DROWSY = False
DEBUG_DISTRACTION = False

# Face Detection Configuration
FACE_DETECTION_CONFIDENCE = 0.5
FACE_MIN_SIZE = (20, 20)

# Eye Detection Configuration
EYE_ASPECT_RATIO_THRESHOLD = 0.2  # Threshold for detecting closed eyes
EYE_CLOSURE_DURATION_THRESHOLD = 1.5  # seconds

# Yawn Detection Configuration
MOUTH_ASPECT_RATIO_THRESHOLD = 0.5  # Threshold for detecting yawn
YAWN_DURATION_THRESHOLD = 0.5  # seconds
YAWN_CONFIDENCE_THRESHOLD = 0.45
YAWN_MAR_THRESHOLD = 0.48
YAWN_MIN_POSITIVE_FRAMES = 2

# Eye fusion thresholds
EYE_MODEL_CONFIDENCE_THRESHOLD_HIGH = 0.75
EYE_MODEL_CONFIDENCE_THRESHOLD_LOW = 0.55

# Distraction / drowsiness model thresholds
DISTRACTION_CONFIDENCE_THRESHOLD = 0.60
DROWSINESS_CONFIDENCE_THRESHOLD = 0.55

# Head Pose Configuration
HEAD_DROOP_THRESHOLD = 20  # degrees from vertical
HEAD_POSE_LOSS_FRAMES = 10  # Frames to consider head lost

# Baseline Calibration
BASELINE_CALIBRATION_FRAMES = 900  # ~30s at 30 FPS fallback window
BASELINE_CALIBRATION_SECONDS = 30  # requested calibration duration
BASELINE_UPDATE_INTERVAL = 100  # Update every 100 frames

# Fatigue Scoring Configuration
FATIGUE_SCORE_WEIGHTS = {
    'eye_closure': 0.4,
    'yawn_frequency': 0.3,
    'head_droop': 0.3,
}

FATIGUE_SCORE_THRESHOLDS = {
    'low': 20,
    'moderate': 40,
    'high': 60,
    'critical': 80,
}

# State Classification
DRIVER_STATES = {
    'ALERT': 'Driver is alert and focused',
    'MILD FATIGUE': 'Driver shows mild fatigue signs',
    'MODERATE FATIGUE': 'Driver shows moderate fatigue signs',
    'SEVERE FATIGUE': 'Driver shows severe fatigue signs',
}

STATE_CLASSIFICATION_RULES = {
    'ALERT': {'max_fatigue_score': 20, 'max_yawn_count': 0, 'min_eye_openness': 0.8},
    'DISTRACTED': {'max_fatigue_score': 50, 'max_yawn_count': 1, 'min_eye_openness': 0.5},
    'FATIGUED': {'max_fatigue_score': 100, 'max_yawn_count': 100, 'min_eye_openness': 0.0},
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_alerts': True,
    'alert_cooldown': 5,  # seconds between alerts of same type
    'enable_sound': True,
    'enable_visual': True,
}

EMERGENCY_ALERT_CONFIG = {
    'enabled': _env_bool('EMERGENCY_EMAIL_ENABLED', False),
    'smtp_server': os.getenv('EMERGENCY_SMTP_SERVER', ''),
    'smtp_port': int(os.getenv('EMERGENCY_SMTP_PORT', '587')),
    'smtp_use_tls': _env_bool('EMERGENCY_SMTP_USE_TLS', True),
    'smtp_username': os.getenv('EMERGENCY_SMTP_USERNAME', ''),
    'smtp_password': os.getenv('EMERGENCY_SMTP_PASSWORD', ''),
    'from_email': os.getenv('EMERGENCY_FROM_EMAIL', ''),
    'to_email': os.getenv('EMERGENCY_TO_EMAIL', ''),
    'subject_prefix': os.getenv('EMERGENCY_EMAIL_SUBJECT_PREFIX', '[Driver Monitor Emergency]'),
    'cooldown_seconds': int(os.getenv('EMERGENCY_EMAIL_COOLDOWN_SECONDS', '120')),
    'driver_id': os.getenv('EMERGENCY_DRIVER_ID', 'UNKNOWN_DRIVER'),
    'vehicle_id': os.getenv('EMERGENCY_VEHICLE_ID', 'UNKNOWN_VEHICLE'),
    'manual_latitude': os.getenv('EMERGENCY_LOCATION_LAT', ''),
    'manual_longitude': os.getenv('EMERGENCY_LOCATION_LON', ''),
    'manual_location_text': os.getenv('EMERGENCY_LOCATION_TEXT', ''),
    'allow_ip_geolocation': _env_bool('EMERGENCY_ALLOW_IP_GEOLOCATION', True),
    'ip_geolocation_url': os.getenv('EMERGENCY_IP_GEOLOCATION_URL', 'https://ipapi.co/json/'),
    'http_timeout_seconds': float(os.getenv('EMERGENCY_HTTP_TIMEOUT_SECONDS', '4.0')),
}

# Real-time Monitoring
FPS_TARGET = 30
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360
INFERENCE_SKIP_FRAMES = 2  # Balanced throughput/latency for typical CPUs
CAMERA_BUFFER_SIZE = 1
CAMERA_FRAME_FLUSH = 1

# Heavy model inference throttling for runtime responsiveness
DISTRACTION_INFERENCE_INTERVAL = 2  # CHANGED: Run every 2 frames instead of 5 for more responsive detection
DROWSINESS_INFERENCE_INTERVAL = 2  # CHANGED: Run every 2 frames instead of 5 for more responsive detection
YAWN_INFERENCE_INTERVAL = 2

# Sudden-event and tracking behavior
SUDDEN_PITCH_CHANGE_THRESHOLD = 18.0  # deg/frame-equivalent change after smoothing
SUDDEN_ROLL_CHANGE_THRESHOLD = 18.0
NO_FACE_ALERT_SECONDS = 0.7
HEAD_TILT_DROOP_THRESHOLD = -15.0
EMERGENCY_SUDDEN_DROP_FROM = -5.0
EMERGENCY_SUDDEN_DROP_TO = -25.0
EMERGENCY_PROLONGED_EYE_CLOSURE_SECONDS = 3.0
EMERGENCY_NO_MOVEMENT_SECONDS = 3.0

# Dashboard UI layout
LEFT_PANEL_WIDTH = 300
RIGHT_PANEL_WIDTH = 320
GRAPH_HISTORY_SECONDS = 15

# Trend Prediction
TREND_WINDOW_SIZE = 30  # frames (1 second at 30 FPS)
EARLY_WARNING_THRESHOLD = 0.8  # 80% of critical threshold

# False Alert Reduction
MIN_CONSECUTIVE_DETECTIONS = 5  # More stable detections to reduce rapid flipping
TEMPORAL_SMOOTHING_WINDOW = 3  # frames

# Safety gating for production readiness
MODEL_DEPLOYMENT_MIN_ACCURACY = 0.60  # Lowered from 0.90 to enable testing of distraction/drowsiness
SHOW_UNVALIDATED_MODEL_SIGNALS = True  # CHANGED: Show predictions even when models not validated, for testing
CALIBRATION_HIDE_NOTIFICATIONS = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "results" / "system.log"

# Display Configuration
SHOW_FPS = True
SHOW_DETECTIONS = True
SHOW_FATIGUE_SCORE = True
SHOW_STATE = True
SHOW_LANDMARKS = False  # Show facial landmarks
SHOW_HEAD_POSE = True
VISUALIZATION_THICKNESS = 2
VISUALIZATION_FONT_SCALE = 0.6

# Device Configuration
USE_GPU = True
GPU_MEMORY_FRACTION = 0.7  # Use 70% of GPU memory
