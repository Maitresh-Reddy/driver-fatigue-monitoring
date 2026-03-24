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

**Critical alerts are optional.** You can run the app without email setup. To enable emergency alerts:

### Step 1: Enable 2FA & Create App Password (Gmail example)
1. Go to https://myaccount.google.com/security
2. Enable **2-Step Verification** if not already done
3. Go to https://myaccount.google.com/apppasswords
4. Select "Mail" + "Windows Computer" → Generate
5. Copy the 16-character **App Password** (you'll use it in `.env`)

### Step 2: Create `.env` File in Project Root
1. Copy `.env.example` to `.env` (must be in same folder as this README)
2. Open `.env` in any text editor and fill:

```env
EMERGENCY_EMAIL_ENABLED=true

EMERGENCY_SMTP_SERVER=smtp.gmail.com
EMERGENCY_SMTP_PORT=587
EMERGENCY_SMTP_USE_TLS=true
EMERGENCY_SMTP_USERNAME=your_gmail@gmail.com
EMERGENCY_SMTP_PASSWORD=xxxx xxxx xxxx xxxx          # ← Paste App Password from Step 1

EMERGENCY_FROM_EMAIL=your_gmail@gmail.com            # ← Same as USERNAME
EMERGENCY_TO_EMAIL=emergency_contact@example.com     # ← Who gets the alert
EMERGENCY_EMAIL_SUBJECT_PREFIX=[Driver Monitor]
EMERGENCY_EMAIL_COOLDOWN_SECONDS=300

EMERGENCY_DRIVER_ID=DRIVER_01
EMERGENCY_VEHICLE_ID=VEHICLE_01

EMERGENCY_LOCATION_TEXT=My Location                  # ← Optional
# Leave LAT/LON blank for IP-based location lookup; otherwise set exact coordinates:
EMERGENCY_LOCATION_LAT=
EMERGENCY_LOCATION_LON=
```

3. **Save and do NOT commit `.env` to Git** (it's in `.gitignore`)

### Step 3: Run & Test
```bash
python src/main.py --mode webcam
```

When system reaches `CRITICAL` state:
- One email is sent to `EMERGENCY_TO_EMAIL` with screenshot, driver condition, and location
- Check Session Info panel (`Email Status`) to see if configured correctly
- See `results/critical_events/` for captured images and `results/emergency_settings.json` for diagnostics

**For non-Gmail SMTP:** Fill your provider's SMTP details (server, port, TLS setting, username/password)

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
