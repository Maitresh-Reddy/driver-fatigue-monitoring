# Driver Fatigue Monitoring System
## Operations Guide & Team Structure

---

## Table of Contents
1. [Quick Start Commands](#quick-start-commands)
2. [Project Glossary](#project-glossary)
3. [Feature Functionality Matrix](#feature-functionality-matrix)
4. [Complete System Architecture](#complete-system-architecture)
5. [Team Roles and Responsibilities](#team-roles-and-responsibilities)

---

## Quick Start Commands

### Prerequisites
- Python 3.9+ with virtual environment activated
- Webcam or video file available
- All dependencies installed: `pip install -r requirements.txt`
- Models available in `models/` directory (or download from GitHub Release v1.0-models)
- `.env` file configured for emergency email (optional but recommended)

### Running in Webcam Mode (Live Detection)
```bash
# Basic: Real-time detection from webcam
python main.py --mode webcam

# Verbose output with detailed logging
python main.py --mode webcam --verbose

# Custom model directory
python main.py --mode webcam --models-dir path/to/custom/models
```

### Running in Video File Mode (Post-Analysis)
```bash
# Process a video file
python main.py --mode video --video path/to/video.mp4

# Custom model path with logging
python main.py --mode video --video dataset/evaluation_videos/test.mp4 --verbose

# Disable emergency alerts for testing
python main.py --mode video --video sample.mp4 --no-alerts
```

### Interactive Controls During Runtime
| Key | Action |
|-----|--------|
| **q** or **e** | Quit application |
| **t** | Toggle dashboard theme (light/dark) |
| **s** | Open emergency settings overlay |
| **r** | Generate and display session report |
| **Esc/Enter/Space** | Close modal overlays (settings/report) |

### Example Workflows

#### Quick Test (30 seconds)
```bash
# Start monitoring, let it run ~30 sec, press 'q' to exit
python main.py --mode webcam
# Result: Outputs session report to results/session_report_YYYYMMDD_HHMMSS.json
```

#### Evaluate Recording
```bash
# Analyze saved video from evaluation dataset
python main.py --mode video --video dataset/evaluation_videos/driver_session.mp4
# Result: Session report and screenshots of critical events
```

#### Setup Training (Offline)
```bash
# Prepare dataset and train models (one-time setup)
python setup_and_train.py
# This trains eye, yawn, distraction, and drowsiness models

# Or train individual models
python train_models.py --model eye
python train_models.py --model yawn
python train_models.py --model distraction
python train_models.py --model drowsiness
```

#### Verify Models
```bash
# Run smoke test on all models
python smoke_test.py
# Expected output: "All models passed smoke tests"
```

---

## Project Glossary

### Core Concepts

#### Driver Fatigue States
| Term | Definition | Score Range |
|------|-----------|--------------|
| **Alert** | Driver is well-rested, no fatigue signs | 0–20 |
| **Caution** | Mild fatigue detected, monitor closely | 21–50 |
| **Warning** | Moderate fatigue, recommend rest | 51–80 |
| **Critical** | Severe fatigue, immediate intervention needed | 81–100 |

#### Detection Metrics

| Metric | Description | Standard Threshold |
|--------|-------------|-------------------|
| **Eye Aspect Ratio (EAR)** | Distance between eyelids; lower = more closed | < 0.25 indicates eye closure |
| **Yawn Duration** | Seconds mouth open continuously | > 0.5 sec = yawn detection |
| **Head Pose** | Roll, pitch, yaw angles of head orientation | > 25° = distraction |
| **Distraction Score** | Model confidence driver is distracted (0–100) | > 70% = distracted |
| **Drowsiness Score** | Model confidence driver is drowsy (0–100) | > 60% = drowsy |

#### Fatigue Score Calculation
**Fatigue Score** = Weighted combination of:
- Eye Aspect Ratio (EAR) weight: 30%
- Yawn detection (binary) weight: 25%
- Head pose (distraction) weight: 20%
- Drowsiness model confidence weight: 25%

Final score clipped to [0, 100] and mapped to driver state (Alert/Caution/Warning/Critical).

#### Event Types
| Event | Trigger | Action |
|-------|---------|--------|
| **Eye Closure** | EAR < threshold for 1+ frame | Log, update fatigue score |
| **Yawn Detection** | Mouth open > 0.5 sec | Log, trigger alert if critical |
| **Distraction** | Head pose > 25° or model > 70% confidence | Log, highlight dashboard |
| **Drowsiness** | Drowsiness model > 60% confidence | Log, trigger email alert if critical |
| **Critical Event** | Fatigue score ≥ 81 | Screenshot, send emergency email, log to critical_events/ |

### System Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Pipeline** | `src/main.py` | Main loop: read frames → detect → update state → render |
| **Config** | `src/config.py` | Centralized settings (thresholds, model paths, feature flags) |
| **Visualizer** | `src/utils/visualization.py` | Render dashboard, overlays, metrics |
| **Facial Features** | `src/detection/facial_features.py` | Extract landmarks for eye/yawn/pose |
| **Monitoring** | `src/system/monitoring.py` | Fatigue scoring, state classification, alert logic |
| **Emergency** | `src/utils/emergency.py` | Email alerts with screenshot + location data |
| **Models** | `models/*.h5` | Trained Keras models for detection |
| **Dataset** | `dataset/` | Training images (eye open/closed, yawn, etc.) |

### Configuration Parameters

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `EYE_ASPECT_RATIO_THRESHOLD` | `config.py` | 0.25 | Threshold below which eyes considered closed |
| `YAWN_DURATION_THRESHOLD` | `config.py` | 0.5 sec | Mouth open duration to trigger yawn |
| `DISTRACTION_CONFIDENCE_THRESHOLD` | `config.py` | 70% | Model confidence required for distraction |
| `DROWSINESS_CONFIDENCE_THRESHOLD` | `config.py` | 60% | Model confidence required for drowsiness |
| `FATIGUE_SCORE_WEIGHTS` | `config.py` | [0.3, 0.25, 0.2, 0.25] | EAR, yawn, head-pose, drowsiness weights |
| `CRITICAL_FATIGUE_THRESHOLD` | `config.py` | 81 | Score above which triggers emergency alert |
| `EMERGENCY_EMAIL_ENABLED` | `.env` | false | Enable/disable email alerts |
| `EMERGENCY_EMAIL_COOLDOWN_SECONDS` | `.env` | 120 | Minimum seconds between emails (anti-spam) |

### Model Information

| Model | Purpose | Input | Output | Path |
|-------|---------|-------|--------|------|
| **eye_model.h5** | Binary eye state classifier | 128×128 grayscale eye ROI | Open (0) or Closed (1) | `models/eye_model.h5` |
| **yawn_model.h5** | Binary yawn classifier | 128×128 mouth ROI | Yawn (0) or No-Yawn (1) | `models/yawn_model.h5` |
| **distraction_model.h5** | Binary distraction classifier | Full face with landmarks | Distracted (0) or Focused (1) | `models/distraction_model.h5` |
| **drowsiness_model.h5** | Regression drowsiness score | 64×64 full face | Drowsiness probability [0, 1] | `models/drowsiness_model.h5` |

---

## Feature Functionality Matrix

### Real-Time Detection Pipeline

```
┌─────────────────┐
│  Input Source   │ ← Webcam or Video File
└────────┬────────┘
         │
┌────────▼────────────────────────────────────┐
│ Frame Capture (30 FPS)                      │
│ - Resize to config dimensions               │
│ - Convert BGR to RGB                        │
└────────┬─────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────┐
│ Facial Landmark Detection                    │
│ - 468-point mediapipe mesh (face_mesh.py)   │
│ - Extract eye ROI, mouth ROI, head pose      │
└────────┬──────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ Multi-Model Inference (Parallel)                 │
│ ├─ Eye Model → Open/Closed (EAR calculation)     │
│ ├─ Yawn Model → Yawn/No-Yawn                     │
│ ├─ Distraction Model → Distracted/Focused        │
│ └─ Drowsiness Model → Drowsiness Score [0, 1]   │
└────────┬───────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────┐
│ State Classification & Fatigue Scoring           │
│ - Compute weighted fatigue score [0, 100]       │
│ - Map to driver state: Alert/Caution/...        │
│ - Detect critical events (score ≥ 81)           │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│ Event Logging & Alert Dispatch                       │
│ - Log to event_timeline.jsonl (all frames)           │
│ - Screenshot critical events → critical_events/      │
│ - Send emergency email (if enabled + cooldown OK)    │
│ - Update dashboard state                            │
└────────┬───────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│ Dashboard Rendering                                  │
│ - Metrics: Fatigue score, driver state, FPS, etc.   │
│ - Landmarks overlay (face mesh, eye/mouth ROI)      │
│ - Alerts & status indicators                        │
│ - Theme toggle (light/dark)                         │
│ - Modal overlays (settings, reports)                │
└────────┬───────────────────────────────────────────────┘
         │
        Loop (next frame)
```

### Feature Breakdown

#### 1. **Real-Time Video Processing**
- Reads frames at 30 FPS from webcam or video file
- Parallel landmark + multi-model inference
- < 100ms per-frame latency (on modern GPU/CPU)
- Handles lighting variations (auto-adjusted thresholds)

#### 2. **Multi-Modal Detection**
- **Eye Detection**: Binary classifier + EAR calculation
- **Yawn Detection**: Binary classifier + duration tracking
- **Head Pose**: Landmark-based roll/pitch/yaw computation
- **Distraction Detection**: Model + landmark-based angle thresholding
- **Drowsiness Detection**: Regression model + confidence scoring

#### 3. **Fatigue Scoring**
- Weighted combination of 4 signals (eye, yawn, head-pose, drowsiness)
- Normalized to [0, 100] scale
- Hysteresis to prevent score flickering
- Time-weighted averaging across frames

#### 4. **Driver State Classification**
- **Alert** (score 0–20): All clear, continue driving
- **Caution** (21–50): Monitor closely, minor fatigue
- **Warning** (51–80): Recommend rest break
- **Critical** (81–100): Intervention required, emergency protocol

#### 5. **Emergency Alert System**
- SMTP email with screenshot + conditions + location
- Cooldown timer (default 120 sec) to prevent email spam
- Optional geolocation (IP-based or manual coordinates)
- Recipient customizable via `.env`

#### 6. **Session Reporting**
- JSON reports with aggregate statistics
- Markdown summary for human review
- Event timeline (all detections with timestamps)
- Critical event screenshots
- Output to `results/session_report_YYYYMMDD_HHMMSS.*`

#### 7. **Dashboard Visualization**
- Live metrics: Fatigue score, driver state, FPS, lighting
- Facial landmarks overlay (468-point mesh)
- ROI highlighting (eyes, mouth)
- Alert indicators
- Theme toggle (light/dark mode)

#### 8. **Configuration Management**
- Centralized `config.py` for all thresholds
- `.env` for secrets (SMTP, credentials, emergency settings)
- Feature flags for emergency alerts, detailed logging, etc.

---

## Complete System Architecture

### Directory Structure and Responsibilities

```
driver-fatigue-monitoring/
│
├── main.py                           # CLI entry point, --help for all options
├── setup_and_train.py                # One-time: prepare dataset, train all models
├── train_models.py                   # Train individual models by type
├── evaluate_multi_video.py           # Batch evaluate videos, generate metrics
├── smoke_test.py                     # Validate all models can be loaded and infer
│
├── src/                              # Core application source code
│   ├── __init__.py
│   ├── main.py                       # DriverFatigueMonitoringPipeline (main loop)
│   ├── config.py                     # Centralized configuration (thresholds, paths, flags)
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── facial_features.py        # Landmark extraction, EAR, yawn duration, head pose
│   │
│   ├── system/
│   │   ├── __init__.py
│   │   └── monitoring.py             # Fatigue scoring, state classification, alert logic
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── models.py                 # Model architecture definitions (CNN, etc.)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py          # Visualizer (dashboard, overlays, themes, modals)
│   │   └── emergency.py              # EmergencyAlertNotifier (SMTP email, geolocation)
│   │
│   └── results/
│       └── training_results.json     # Cached model performance metrics
│
├── models/                           # Pre-trained .h5 files
│   ├── eye_model.h5
│   ├── yawn_model.h5
│   ├── distraction_model.h5
│   └── drowsiness_model.h5
│
├── dataset/                          # Training and evaluation data
│   ├── raw/                          # Raw datasets (CEW, NTU, YAWDD, etc.)
│   ├── eye/                          # Processed eye crops {open/, closed/}
│   ├── yawn/                         # Processed yawn crops {yawn/, non_yawn/}
│   └── evaluation_videos/            # Real-world test videos
│
├── results/                          # Runtime outputs
│   ├── session_report_*.json         # JSON reports from sessions
│   ├── session_report_*.md           # Markdown summaries
│   ├── event_timeline.jsonl          # All-frame event log
│   ├── training_results.json         # Cached training metrics
│   └── critical_events/              # Screenshots of critical fatigue events
│
├── scripts/
│   ├── secret_guard.py               # Utility to check .env for leaked secrets
│   └── ... (offline utilities)
│
├── notebooks/
│   └── demo.ipynb                    # Jupyter notebook demo/exploration
│
├── .env.example                      # Safe template for .env (committed)
├── .env                              # Runtime secrets (NOT committed, in .gitignore)
├── .gitignore                        # Excludes cache, secrets, runtime artifacts
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview, setup, deployment info
├── SECURITY.md                       # Security best practices, cleanup guidance
└── PROJECT_SUMMARY.txt               # Feature summary (text version)
```

### Data Flow Example: Critical Event Detection

```
Frame Input (webcam, 30 fps)
    ↓
[1] Facial Landmark Detection (mediapipe)
    ├─ Face detected? → Continue : Skip frame
    ├─ Extract eye ROI (landmarks 33, 133, etc.)
    ├─ Extract mouth ROI (landmarks 13, 14, etc.)
    └─ Calculate head pose (roll, pitch, yaw)
    ↓
[2] Parallel Model Inference
    ├─ Eye Model(eye_roi) → Confidence(closed)
    ├─ Yawn Model(mouth_roi) → Confidence(yawn)
    ├─ Distraction Model(full_face) → Confidence(distracted)
    └─ Drowsiness Model(full_face) → Drowsiness_score
    ↓
[3] Feature Extraction
    ├─ EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)  [Eye Aspect Ratio]
    ├─ Yawn_duration += 1 frame (if mouth_open > threshold)
    ├─ Head_angle = max(|roll|, |pitch|, |yaw|)
    └─ Drowsiness_confidence = model_output
    ↓
[4] Fatigue Score Calculation
    ├─ ear_signal = (1 - EAR) if EAR < 0.25 else 0
    ├─ yawn_signal = 1 if yawn_duration > 0.5 else 0
    ├─ pose_signal = (head_angle / 45) if angle > 25° else 0
    ├─ drowsiness_signal = drowsiness_confidence
    ├─ fatigue_score = 30*ear + 25*yawn + 20*pose + 25*drowsy
    └─ fatigue_score = clip(fatigue_score, 0, 100)
    ↓
[5] State Classification & Alerting
    ├─ if fatigue_score >= 81 → CRITICAL
    │  ├─ Take screenshot → critical_events/frame_TIMESTAMP.png
    │  ├─ Log event to event_timeline.jsonl
    │  └─ Send emergency email (if enabled && cooldown OK)
    │
    ├─ if fatigue_score >= 51 → WARNING
    │  └─ Highlight dashboard, log event
    │
    ├─ if fatigue_score >= 21 → CAUTION
    │  └─ Update dashboard, log event
    │
    └─ else → ALERT (normal)
    ↓
[6] Dashboard Rendering
    ├─ Render video frame with landmarks
    ├─ Overlay metrics: fatigue_score, state, FPS
    ├─ Draw alert indicators (color-coded by state)
    ├─ Render modal overlays if active (settings, report)
    └─ Display help text (keyboard shortcuts)
    ↓
Frame Output (OpenCV window)
```

### Configuration Dependency Graph

```
.env (secrets & emergency settings)
  ├─ EMERGENCY_EMAIL_ENABLED → monitoring.py: skip alerts if false
  ├─ EMERGENCY_SMTP_* → emergency.py: SMTP connection params
  ├─ EMERGENCY_TO_EMAIL → emergency.py: alert recipient
  ├─ EMERGENCY_DRIVER_ID, VEHICLE_ID → emergency.py: email body
  ├─ EMERGENCY_LOCATION_* → emergency.py: geolocation (or IP fallback)
  └─ EMERGENCY_EMAIL_COOLDOWN_SECONDS → monitoring.py: rate-limit alerts

config.py (detection thresholds & model paths)
  ├─ EYE_ASPECT_RATIO_THRESHOLD → facial_features.py: ear_signal calc
  ├─ YAWN_DURATION_THRESHOLD → monitoring.py: yawn detection
  ├─ DISTRACTION_CONFIDENCE_THRESHOLD → monitoring.py: distraction alert
  ├─ DROWSINESS_CONFIDENCE_THRESHOLD → monitoring.py: drowsiness alert
  ├─ FATIGUE_SCORE_WEIGHTS → monitoring.py: weighted fatigue calc
  ├─ CRITICAL_FATIGUE_THRESHOLD → monitoring.py: trigger emergency protocol
  ├─ MODEL_PATHS → main.py: load .h5 files
  └─ FEATURE_FLAGS → main.py: enable/disable logging, detailed output
```

---

## Team Roles and Responsibilities

### Project Overview for Team Structure
The Driver Fatigue Monitoring System spans **computer vision**, **machine learning**, **real-time systems**, **UI/dashboard design**, **backend state management**, **infrastructure**, and **security**. A 4-person team can cover these domains effectively with strategic role definition.

---

### Role 1: Computer Vision & Detection Lead
**Primary Domains:** Facial landmark extraction, multi-modal detection pipelines, model integration  
**Files Owned:**
- `src/detection/facial_features.py` (landmark extraction, EAR, yawn duration, head pose)
- `src/training/models.py` (model architecture design)
- `dataset/` (dataset preparation, cleaning, augmentation)
- Training scripts: `setup_and_train.py`, `train_models.py`

**Key Responsibilities:**
1. **Facial Landmark Extraction**
   - Maintain mediapipe integration for 468-point face mesh
   - Implement robust eye, mouth, and head-pose ROI extraction
   - Handle edge cases (partial faces, shadows, occlusions)
   - Optimize landmark calculation latency

2. **Multi-Modal Model Management**
   - Train and evaluate eye, yawn, distraction, drowsiness classifiers
   - Version models and track performance metrics in `results/training_results.json`
   - Implement quantization/optimization for real-time inference
   - Maintain model accuracy benchmarks across dataset variations

3. **Dataset Stewardship**
   - Organize training data in `dataset/` (eye/open, eye/closed, yawn/non_yawn)
   - Curate evaluation_videos with ground-truth labels
   - Implement data augmentation pipelines (rotation, lighting, etc.)
   - Generate performance reports via `evaluate_multi_video.py`

4. **Detection Algorithm Development**
   - Compute Eye Aspect Ratio (EAR) robustly
   - Implement yawn duration tracking with state machine
   - Develop distraction detection via head pose + model confidence
   - Integrate drowsiness model predictions

**Success Metrics:**
- Models achieve >90% accuracy on test set
- EAR calculation is robust to head rotation
- Per-frame inference latency < 50ms
- Critical events captured with >85% recall

**Dependencies/Collaboration:**
- Works with **Monitoring Lead** on fatigue scoring formula
- Works with **Dashboard Lead** on landmark visualization overlays
- Works with **DevOps Lead** on model versioning and release management

---

### Role 2: Backend & Monitoring Logic Lead
**Primary Domains:** Fatigue scoring, state classification, alert protocol, session management  
**Files Owned:**
- `src/system/monitoring.py` (core fatigue scoring, state classification, emergency protocol)
- `src/main.py` (pipeline orchestration, frame-loop coordination)
- `src/config.py` (threshold management, feature flags)
- `src/utils/emergency.py` (alert dispatch logic)
- Results pipeline: logging, reporting, artifact management

**Key Responsibilities:**
1. **Fatigue Score Calculation**
   - Implement weighted combination of EAR, yawn, head-pose, drowsiness signals
   - Maintain normalization to [0, 100] scale
   - Apply hysteresis/smoothing to prevent score flickering
   - Expose configurable weights in `config.py`

2. **Driver State Classification**
   - Implement state machine: Alert → Caution → Warning → Critical
   - Define state transition thresholds (0–20, 21–50, 51–80, 81–100)
   - Handle edge cases (rapid state transitions, noisy signals)
   - Generate state change events with timestamps

3. **Emergency Alert Protocol**
   - Define critical event triggers (fatigue ≥ 81, specific yawn/eye patterns)
   - Implement cooldown logic to prevent email spam (default 120 sec)
   - Coordinate with `emergency.py` to dispatch alerts
   - Log all alerts to `event_timeline.jsonl`

4. **Event Logging & Timeline**
   - Write per-frame events to `event_timeline.jsonl` (timestamp, state, signals, confidence)
   - Screenshot critical events → `results/critical_events/`
   - Aggregate session statistics (duration, state distribution, critical events count)
   - Generate JSON + Markdown session reports

5. **Session Management**
   - Track session start/end, total duration, event counts
   - Compute summary statistics (avg fatigue score, time in each state, etc.)
   - Output to `results/session_report_YYYYMMDD_HHMMSS.*`
   - Expose session data for dashboard display

**Success Metrics:**
- Fatigue score is stable (low frame-to-frame variance)
- State transitions are responsive (<500ms latency)
- Critical events trigger alerts reliably (>95% true-positive rate)
- Session reports accurately reflect driving conditions

**Dependencies/Collaboration:**
- Works with **CV Lead** on detection signal integration
- Works with **Dashboard Lead** on real-time metric updates
- Works with **DevOps Lead** on configuration and .env secrets

---

### Role 3: UI/Dashboard & User Experience Lead
**Primary Domains:** Real-time visualization, interactive overlays, theme management, user input handling  
**Files Owned:**
- `src/utils/visualization.py` (Visualizer class: dashboard rendering, overlays, themes, modal handling)
- Keyboard input handling in `src/main.py`
- Theme assets (colors, fonts, styling)

**Key Responsibilities:**
1. **Dashboard Rendering**
   - Render live video with overlay metrics (fatigue score, driver state, FPS, lighting)
   - Display facial landmarks (468-point mesh) with ROI highlighting
   - Draw alert indicators (color-coded by state: green/yellow/red)
   - Support smooth rendering at 30 FPS without frame drops

2. **Theme Management**
   - Implement light/dark theme toggle (keyboard: 't')
   - Define color palettes for each state (Alert/Caution/Warning/Critical)
   - Ensure readable text contrast in both themes
   - Persist theme preference (optional, via config)

3. **Interactive Overlays**
   - **Emergency Settings Modal** (keyboard: 's')
     - Display editable fields: email, recipient, cooldown, location
     - Validate user input before submission
     - Save settings locally or to `.env`
   - **Session Report Modal** (keyboard: 'r')
     - Display aggregate statistics and critical events list
     - Close via Esc/Enter/Space/Q/E keys
     - Format for readability (timestamps, state durations, etc.)

4. **Keyboard Input Handling**
   - Map keys to actions: 'q'/'e' (quit), 't' (theme), 's' (settings), 'r' (report)
   - Implement modal context: overlays intercept keys before background handlers
   - Provide on-screen help text (visible in top-left corner)
   - Handle Esc/Enter/Space to close modals

5. **Visual Feedback**
   - Provide clear visual indicators of detection results (green eyes = open, red = closed)
   - Highlight areas of interest (distraction angle, yawn mouth)
   - Show real-time FPS and system status
   - Blink or flash alerts for critical events (optional)

**Success Metrics:**
- Dashboard renders smoothly at 30 FPS with no dropped frames
- Modal overlays are intuitive and responsive to keyboard input
- Theme toggle is instant with no visual artifacts
- Help text is clear and accessible

**Dependencies/Collaboration:**
- Works with **Monitoring Lead** on metrics display format
- Works with **CV Lead** on landmark visualization
- Works with **DevOps Lead** on config-driven styling (colors, fonts)

---

### Role 4: DevOps, Infrastructure & Configuration Lead
**Primary Domains:** Deployment, configuration management, environment setup, testing, security  
**Files Owned:**
- `requirements.txt` (dependency management)
- `.env.example` (safe template for secrets)
- `.env` (actual secrets, never committed)
- `.gitignore` (prevent accidental leaks)
- `config.py` (centralized configuration defaults)
- `scripts/secret_guard.py` (validation utility)
- `README.md`, `SECURITY.md` (documentation)
- Testing: `smoke_test.py`, `final_verification.py`

**Key Responsibilities:**
1. **Environment & Dependency Management**
   - Maintain `requirements.txt` with pinned versions
   - Document Python version requirements (3.9+)
   - Provide setup instructions for Windows/macOS/Linux
   - Test in isolated virtual environment (`venv/`)

2. **Configuration Management**
   - Define safe defaults in `config.py` (thresholds, model paths, feature flags)
   - Maintain `.env.example` as safe template with comments
   - Implement `.env` loading via python-dotenv
   - Validate .env syntax (no dangling statements, quoted values)
   - Document all configuration parameters in README

3. **Security & Credential Protection**
   - Never commit `.env` or credentials to Git
   - Maintain `.gitignore` patterns for secrets: *.pem, *.key, credentials.json
   - Implement `secret_guard.py` to audit for leaked secrets
   - Document safe workspace cleanup (remove caches, logs, screenshots)
   - Add SECURITY.md with best practices

4. **Model Versioning & Deployment**
   - Provide pre-trained models in GitHub Release (v1.0-models)
   - Document model download & placement process
   - Version models with metadata (accuracy, dataset, training date)
   - Plan multi-model deployment strategy (local vs. cloud)

5. **Testing & Quality Assurance**
   - Implement `smoke_test.py`: validate all models load and infer
   - Implement `final_verification.py`: end-to-end integration test
   - Test webcam and video file modes
   - Verify emergency email sending (dry-run if needed)
   - Check dashboard rendering and theme toggle

6. **Documentation & Knowledge Transfer**
   - Maintain README.md with quick-start, features, architecture
   - Document CLI options and keyboard shortcuts
   - Keep SECURITY.md updated with cleanup procedures
   - Write this operations guide for team onboarding

7. **Development Workflow**
   - Define branching strategy (main, develop, feature branches)
   - Set up CI/CD for testing and deployment (if applicable)
   - Create issue/PR templates for team collaboration
   - Maintain project roadmap and changelog

**Success Metrics:**
- Setup time for new developer: < 15 minutes (following README)
- All tests pass: smoke_test.py, final_verification.py
- No secrets leaked to Git (verified via secret_guard.py)
- Environment is reproducible across machines
- Documentation is complete and up-to-date

**Dependencies/Collaboration:**
- Supports all roles: CV Lead, Monitoring Lead, Dashboard Lead
- Works with entire team on configuration changes and releases
- Owns deployment and environment consistency

---

### Cross-Role Collaboration Points

| Scenario | CV Lead | Monitoring Lead | Dashboard Lead | DevOps Lead | Action |
|----------|---------|-----------------|----------------|-------------|--------|
| New detection signal added | ✓ (implement) | ✓ (integrate into fatigue) | ✓ (visualize) | ✓ (config param) | Meeting to define threshold & weight |
| Critical event triggered | ✓ (detection) | ✓ (alert dispatch) | ✓ (highlight) | ✓ (logging/storage) | Ensure all components notified |
| Theme/UI updated | — | — | ✓ (implement) | ✓ (config colors) | Review and test both themes |
| Model retrained | ✓ (train & test) | — | — | ✓ (version & release) | Version control and deployment |
| Bug found in production | ? (investigator) | ? (investigator) | ? (investigator) | ✓ (hotfix & deploy) | Debug, fix, test, release cycle |

---

### Example Workflow: Implement "Blink Detection" Feature

1. **CV Lead** designs blink detection algorithm (eye closed for 1–2 frames)
2. **CV Lead** extracts blink detection signal from existing eye model
3. **Monitoring Lead** integrates blink signal into fatigue score (e.g., 5% weight)
4. **Monitoring Lead** adjusts fatigue thresholds to account for new signal
5. **Dashboard Lead** adds blink counter to dashboard display
6. **Dashboard Lead** highlights eyes when blink detected
7. **DevOps Lead** adds config parameter: `BLINK_DETECTION_ENABLED`
8. **DevOps Lead** tests end-to-end in webcam mode
9. All review and merge via PR

---

### Onboarding Checklist for New Team Members

**Week 1:**
- [ ] Clone repository and follow README setup (< 15 min)
- [ ] Run `python main.py --mode webcam` and verify dashboard renders
- [ ] Read this OPERATIONS_AND_TEAM_GUIDE.md
- [ ] Read PROJECT_SUMMARY.txt and SECURITY.md
- [ ] Understand role assignment and code ownership
- [ ] Set up development environment (IDE, Git, virtual env)

**Week 2:**
- [ ] Study assigned role's code files (facial_features.py, monitoring.py, visualization.py, or config.py)
- [ ] Run `python smoke_test.py` and `python final_verification.py`
- [ ] Trace one critical event through the entire pipeline
- [ ] Attend team sync to ask role-specific questions

**Week 3:**
- [ ] Pick a small task from backlog (e.g., add a new metric, improve threshold)
- [ ] Implement, test locally, create PR
- [ ] Collaborate with peers on code review
- [ ] Merge and deploy

---

### Communication Guidelines

**Daily Standup (15 min):**
- Each role reports: completed yesterday, doing today, blockers
- CV Lead: model training progress, dataset status
- Monitoring Lead: fatigue scoring refinements, alert logic
- Dashboard Lead: UI improvements, user feedback
- DevOps Lead: environment, testing, deployment status

**Weekly Sync (1 hour):**
- Review performance metrics (accuracy, latency, error rates)
- Discuss cross-role dependencies and alignment
- Plan next sprint or milestone
- Escalate blockers

**On-Demand Pairing:**
- Whenever one role blocks another (e.g., Dashboard needs monitoring.py API)
- Pair-code for complex features spanning multiple modules
- Review pull requests before merge

---

## Summary

This guide provides:
1. **Quick Start Commands** for running the system in any mode
2. **Project Glossary** with all technical terms and thresholds
3. **Feature Functionality Matrix** covering the full pipeline
4. **Complete System Architecture** showing module relationships and data flow
5. **4-Person Team Roles** with clear ownership, responsibilities, and collaboration points

Use this as a reference for operations, onboarding, and team coordination.

