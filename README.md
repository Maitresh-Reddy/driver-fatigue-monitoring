# Driver Fatigue Monitoring System

Real-time driver monitoring for fatigue, distraction,drowsiness and emergency detection with a dashboard UI.

## Features
- Real-time webcam/video monitoring
- Eye/yawn/head-pose based fatigue analytics
- Driver state classification (`ALERT` to `CRITICAL`)
- Explainable alert reasons in UI
- Emergency/sudden-event handling
- Auto critical screenshot capture
- Emergency contact email on critical alerts (with screenshot + location coordinates + condition)
- Live emergency status lines in Session Info (email send status + screenshot notification status)
- Light/Dark UI theme toggle
- In-UI session report on `Exit + Report`

## Project Structure
- `src/main.py` - App entrypoint and runtime pipeline
- `src/system/monitoring.py` - Core monitoring and state logic
- `src/utils/visualization.py` - Dashboard rendering/UI controls
- `src/detection/` - Landmark and facial feature extraction
- `src/training/` - Training/data utilities
- `models/` - Trained model files (`*.h5`)
- `dataset/` - Input datasets and evaluation videos
- `results/` - Runtime/generated artifacts

## Requirements
- Python 3.10 or 3.11 recommended
- Webcam (for live mode)
- Windows/Linux/macOS

Install dependencies from:
- `requirements.txt`

## Setup

### 0) Download Pre-trained Models (Quick Path - Recommended)
Pre-trained models are available as a GitHub Release for immediate use without training:

**[Download Pre-trained Models (v1.0-models)](https://github.com/Maitresh-Reddy/driver-fatigue-monitoring/releases/tag/v1.0-models)**

Extract the 4 files (`eye_model.h5`, `yawn_model.h5`, `distraction_model.h5`, `drowsiness_model.h5`) into the `models/` folder after cloning.

### 1) Clone and enter project folder
```bash
cd driver-fatigue-monitoring
```

### 2) Create and activate virtual environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux/macOS (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Share This Project So Others Can Run It
This repository is configured for security-first sharing.

What is intentionally not tracked in Git:
- Local datasets (`dataset/`, `src/dataset/`)
- Local models (`models/`)
- Local notebooks and runtime outputs (`notebooks/`, `results/`)

To help others reproduce your experience, use one of these paths:

1) **Quick feedback path (recommended)** ⭐
- Download pre-trained models from the **[GitHub Release (v1.0-models)](https://github.com/Maitresh-Reddy/driver-fatigue-monitoring/releases/tag/v1.0-models)**
- Place the 4 `.h5` files in `models/`
- Share a test video file (or ask testers to use their own)
- Testers run: `python src/main.py --mode video --video path/to/test_video.mp4`
- Feedback is immediate, no training required

2) Full rebuild path
- Testers run `python setup_and_train.py` (or `train_models.py`) to build models locally from their own dataset setup.
- This path takes longer but avoids distributing model binaries.

## Run the App

### Webcam mode (recommended)
```bash
python src/main.py --mode webcam
```

### Video file mode
```bash
python src/main.py --mode video --video path/to/video.mp4
```

### Recommended command for shared testing
```bash
python src/main.py --mode video --video path/to/test_video.mp4
```

## Emergency Contact + Location Alert Setup
Copy `.env.example` to `.env`, then fill your values to enable emergency email alerts.

Example `.env`:
```env
EMERGENCY_EMAIL_ENABLED=true

EMERGENCY_SMTP_SERVER=smtp.gmail.com
EMERGENCY_SMTP_PORT=587
EMERGENCY_SMTP_USE_TLS=true
EMERGENCY_SMTP_USERNAME=your_email@gmail.com
EMERGENCY_SMTP_PASSWORD=your_app_password

EMERGENCY_FROM_EMAIL=your_email@gmail.com
EMERGENCY_TO_EMAIL=emergency_contact@example.com
EMERGENCY_EMAIL_SUBJECT_PREFIX=[Driver Monitor Emergency]
EMERGENCY_EMAIL_COOLDOWN_SECONDS=120

EMERGENCY_DRIVER_ID=DRIVER_01
EMERGENCY_VEHICLE_ID=VEHICLE_01

EMERGENCY_LOCATION_LAT=12.9715987
EMERGENCY_LOCATION_LON=77.594566
EMERGENCY_LOCATION_TEXT=MG Road, Bengaluru

EMERGENCY_ALLOW_IP_GEOLOCATION=true
EMERGENCY_IP_GEOLOCATION_URL=https://ipapi.co/json/
EMERGENCY_HTTP_TIMEOUT_SECONDS=4.0
```

Behavior:
- When the system enters `CRITICAL`, it captures a screenshot in `results/critical_events/`.
- It sends one emergency email per critical episode to `EMERGENCY_TO_EMAIL`.
- The email includes driver condition, fatigue/risk status, screenshot attachment, and coordinates.
- If `EMERGENCY_LOCATION_LAT` and `EMERGENCY_LOCATION_LON` are set, those exact coordinates are used.
- Otherwise, it falls back to IP-based location (approximate).
- SMTP password is intentionally not persisted to `results/emergency_settings.json`; use `.env` for secure persistence.

## Controls
- `t` - Toggle Light/Dark theme
- `s` - Open/close Emergency Settings panel
- `q` or `e` - Exit
- `r` - Generate session report overlay
- UI buttons in top bar:
  - Emergency Settings
  - Theme toggle
  - Exit
  - Exit + Report

Emergency Settings panel usage:
- `Tab` / Arrow keys: move between fields
- `Space`: toggle boolean fields
- Type text directly for email/location fields
- `Backspace`: delete character
- `Enter`: save settings
- `Esc`: close panel

Session Info includes emergency diagnostics:
- `Email Status`: `SENT`, `FAILED`, `NOT CONFIGURED`, `COOLDOWN`, or `SKIPPED`
- `Screenshot Status`: whether CRITICAL screenshot capture/notification succeeded
- `Last Email` and `Last Screenshot` timestamps/messages for recent critical events

Important:
- Setting only `Emergency Contact Email` is not enough.
- You must also set SMTP fields (`SMTP Server`, `SMTP Port`, `SMTP Username`, `SMTP Password`, `From Email`) and enable `Email Alerts Enabled`.
- For Gmail, use `smtp.gmail.com`, port `587`, TLS on, and an App Password.

## Optional Utilities
- Full setup + training workflow:
```bash
python setup_and_train.py
```

- Train models only:
```bash
python train_models.py
```

- Multi-video benchmark:
```bash
python evaluate_multi_video.py
```

## Models
Expected model files in `models/`:
- `eye_model.h5`
- `yawn_model.h5`
- `distraction_model.h5`
- `drowsiness_model.h5`

If any are missing, runtime falls back where possible but detection quality may drop.

## Output Files (Generated at Runtime)
When running the app, these may be created automatically:
- `results/system.log`
- `results/event_timeline.jsonl`
- `results/critical_events/*.png`
- `results/emergency_settings.json`
- `results/session_report_*.json`
- `results/session_report_*.md`

## GitHub / Secret Safety
- `results/emergency_settings.json` may include non-secret SMTP metadata (server/username/to-email), but SMTP password is sanitized.
- Do not commit this file to Git.
- If credentials were ever committed or shared, rotate/revoke them immediately and generate a new app password.

Safety protections now included in this repo:
- `.gitignore` excludes `.env` and sensitive runtime artifacts.
- `.env.example` is provided as a safe template.
- Runtime settings persistence sanitizes SMTP password before writing to disk.
- `scripts/secret_guard.py` scans for common secret patterns.
- `.githooks/pre-commit` blocks commits when secrets/sensitive files are detected.

For any new clone, enable hooks once:
```bash
git config core.hooksPath .githooks
```

## Troubleshooting
- If webcam cannot open:
  - Close other apps using camera.
  - Try changing camera index in `src/main.py` (`cv2.VideoCapture(0)`).
- If model load warnings appear:
  - Verify `models/*.h5` files exist.
- If FPS is low:
  - Reduce `VIDEO_WIDTH` / `VIDEO_HEIGHT` in `src/config.py`.
  - Increase `INFERENCE_SKIP_FRAMES` in `src/config.py`.

## Feedback Guide
If you are testing this project, please share:
- OS and Python version
- Run mode used (`webcam` or `video`)
- Whether models were pre-downloaded or trained locally
- What worked as expected
- What failed (include terminal logs / traceback)
- Optional screenshot from dashboard and output artifacts from `results/`

## Safety Note
This project is a monitoring aid and still not a substitute for safe driving practices or certified in-vehicle safety systems.
