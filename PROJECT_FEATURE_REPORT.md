# Driver Fatigue Monitoring - Project Feature Report

## 1) Project Overview
This project is a real-time driver monitoring system that detects fatigue, distraction, and emergency states using webcam/video input.

Core goals achieved:
- Real-time monitoring pipeline
- Fatigue/distraction/emergency detection
- Explainable UI with reasons and risk indicators
- Critical-event screenshot capture
- Session analytics/report export

## 2) Implemented Features (Detailed)

### 2.1 Perception & Detection
- Face/landmark detection using MediaPipe (468 landmarks).
- Eye-region and mouth-region extraction from landmarks.
- Head pose estimation (pitch/yaw/roll) and head tilt tracking.

Primary code:
- `src/detection/facial_features.py`

### 2.2 ML Models & Inference
- Eye state model (`open/closed`) loading and inference.
- Yawn model (`yawn/non-yawn`) loading and inference.
- Distraction model loading and inference.
- Drowsiness model loading and inference.
- Runtime inference throttling for performance stability.

Primary code:
- `src/main.py`
- `src/training/models.py`

### 2.3 Baseline Calibration (Driver-specific)
- Initial calibration phase (`CALIBRATING`) to learn baseline behavior.
- Continuous baseline adaptation for long sessions.
- Baseline-aware deviation used in fatigue and posture logic.

Primary code:
- `src/system/monitoring.py` (`BaselineCalibration`)

### 2.4 Fatigue Scoring & Trend
- Composite fatigue score (0-100) with decay.
- Signals include eye closure duration, yawn frequency, and head droop/tilt.
- Trend extraction (`increasing/stable/decreasing`) from score history.

Primary code:
- `src/system/monitoring.py` (`FatigueScoring`)

### 2.5 Driver State Classification
States produced in runtime:
- `ALERT`
- `MILD FATIGUE`
- `MODERATE FATIGUE`
- `SEVERE FATIGUE`
- `CRITICAL`
- `CALIBRATING`
- `TRACKING_LOST`

Primary code:
- `src/system/monitoring.py` (`DriverStateClassifier` + `DriverMonitoringSystem`)

### 2.6 Emergency & Sudden Event Logic
- Emergency detection for prolonged eye closure/head drop conditions.
- Sudden event detection from abrupt pose changes and collapse-style events.
- Immediate `CRITICAL` behavior and recovery transition handling.

Primary code:
- `src/system/monitoring.py`

### 2.7 Explainable Alerts
- Alerts include explicit reasons (e.g., eye closure duration, yawn rate, pose/drop signals).
- Category-aware behavior for distraction and drowsiness alerting.

Primary code:
- `src/system/monitoring.py`
- `src/main.py`

### 2.8 Critical Screenshot Capture (Auto)
- Auto screenshot capture when critical condition is active.
- Per-episode handling and cooldown to avoid spam.
- Saved under `results/critical_events/` at runtime.

Primary code:
- `src/main.py` (`_maybe_capture_critical_screenshot`, `_save_critical_screenshot`)

### 2.9 Emergency Contact Email + Location Dispatch
- On `CRITICAL`, the system sends an emergency email to configured contact.
- Email includes:
  - Driver condition/state and risk context
  - Critical screenshot attachment
  - Coordinates (exact when configured manually)
  - Location label and map link
- One email is sent per critical episode with cooldown protection.
- Full SMTP configuration is supported from in-app Emergency Settings (server/port/TLS/username/password/from/to).
- Runtime diagnostics track and display latest emergency outcomes:
  - Email status (`SENT` / `FAILED` / `NOT CONFIGURED` / `COOLDOWN` / `SKIPPED`)
  - Screenshot status and latest event details in Session Info.

Primary code:
- `src/main.py` (`_maybe_send_emergency_email` and critical flow integration)
- `src/utils/emergency.py` (`EmergencyAlertNotifier`)

### 2.10 Dashboard/UI Features
- Full dashboard rendering for live monitoring.
- Driver status card, alert level, safety features, session info, fatigue analytics.
- "WHY THIS ALERT?" reason panel.
- Visible calibration banner and system-check panel.
- Light/Dark mode toggle (top bar button + keyboard `t`).
- Exit and Exit+Report controls.
- In-UI session report overlay on `Exit + Report`.

Primary code:
- `src/utils/visualization.py`
- `src/main.py`

### 2.11 Logging, Timeline, Reporting
- Event timeline logging (`event_timeline.jsonl` at runtime).
- Session report generation (JSON + Markdown) when report action is used.
- Performance/resource metrics collection.

Primary code:
- `src/main.py`
- `src/utils/visualization.py` (supporting display)

## 3) Scripts and What They Do
- `src/main.py` - Main runtime entrypoint (webcam/video modes).
- `setup_and_train.py` - Dataset setup + training workflow automation.
- `train_models.py` - Model training utility.
- `evaluate_multi_video.py` - Multi-video benchmark evaluator.
- `final_verification.py` - Project verification utility.
- `debug_models.py` - Model diagnostics helper.
- `smoke_test.py` - Lightweight runtime smoke test helper.

## 4) Data/Model Layout
- Input datasets in `dataset/`.
- Trained model files in `models/`.
- Runtime/generated outputs in `results/`.

## 5) Cleanup Audit (Current)
Performed safe cleanup without changing source logic or feature behavior:
- Removed non-essential generated runtime artifacts (logs/reports/screenshots/temp outputs).
- Removed previous ad-hoc test scripts and cache files (`__pycache__` / `.pyc`).
- Kept essential runtime assets and source files.
- Kept model files in `models/`.
- Kept essential JSON result baselines:
  - `results/training_results.json`
  - `results/test_results.json`

## 6) Runtime Integrity Check
After cleanup, core runtime files were compile-validated:
- `src/main.py`
- `src/system/monitoring.py`
- `src/utils/visualization.py`

Status: compile checks pass.

## 7) Notes
Some runtime artifacts (for example logs, timeline files, critical screenshots, and session reports) are designed to be recreated automatically when the app runs. That is expected behavior.

Security note:
- Runtime emergency settings are stored in `results/emergency_settings.json`, but SMTP password is sanitized before write.
- Secret persistence is expected via `.env` (template: `.env.example`).
- Sensitive runtime artifacts and `.env` are excluded via `.gitignore` to reduce credential exposure risk.
- Automated secret scanning is available via `scripts/secret_guard.py` and enforced at commit time via `.githooks/pre-commit`.
