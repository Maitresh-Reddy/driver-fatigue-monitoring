# Driver Fatigue Monitoring System - Model Decisions FAQ

This document answers the common "why this model, why this approach, and why this feature" questions for the project.

## 1) Why did you use separate models for eye, yawn, distraction, and drowsiness detection?
We split the problem into smaller signals because driver fatigue is not a single visual cue. Eye closure, yawning, distraction, and drowsiness do not behave the same way, so separate detectors are easier to tune, debug, and improve independently.

This also makes the runtime more resilient. If one signal is unavailable or weak, the system can still rely on the others and keep producing a useful fatigue estimate.

## 2) Why use MobileNetV2 for the eye and yawn classifiers?
MobileNetV2 is a good fit for real-time monitoring because it is lightweight, fast, and works well for transfer learning. The project needs inference during live webcam/video processing, so model size and latency matter.

The code in [src/training/models.py](src/training/models.py) uses MobileNetV2 as the base for both the eye state and yawn models, then adds a small custom classification head. That gives a practical balance between speed and accuracy.

## 3) Why not use one large end-to-end model for everything?
A single large model would be harder to train, harder to debug, and less transparent when something fails. With separate models, it is easier to answer questions like:
- Is the eye detector wrong?
- Is the yawn classifier missing events?
- Is the distraction model too sensitive?
- Is the drowsiness score unstable?

That separation also makes future updates simpler because each detector can be retrained without rebuilding the whole pipeline.

## 4) Why is MediaPipe used for facial landmarks and head pose?
MediaPipe gives a fast facial landmark pipeline that is well suited for real-time applications. The project uses those landmarks to extract the eye region, mouth region, and head pose signals that feed the fatigue logic.

That choice keeps the system explainable. Instead of relying only on black-box predictions, the app can show why an alert was raised, which is important for driver-facing feedback and debugging.

## 5) Why combine model outputs with landmark-based rules?
The project mixes learned signals and rule-based signals on purpose. The learned models capture visual patterns, while the landmark and timing rules capture behavior over time, such as eye closure duration, yawn frequency, and head droop.

That hybrid design reduces false alerts and makes the system more stable in difficult conditions like lighting changes, partial occlusions, or momentary detection noise.

## 6) Why is there a baseline calibration phase?
Drivers have different normal head posture, blink patterns, and mouth movement habits. The calibration phase lets the system learn a personal baseline before it starts interpreting deviations as fatigue.

This reduces overreaction. A generic threshold may flag one driver too early and another too late, so a driver-specific baseline improves practical accuracy.

## 7) Why use a composite fatigue score instead of a direct label from one model?
A single label like "fatigued" is too coarse for a safety-oriented app. A composite score lets the system combine multiple symptoms and express severity on a 0-100 scale.

That gives better control over state transitions such as `ALERT`, `MILD FATIGUE`, `MODERATE FATIGUE`, `SEVERE FATIGUE`, and `CRITICAL`. It also makes the dashboard and reporting clearer.

## 8) Why are alerts explainable with reasons?
Because the driver and the reviewer need to understand what caused the warning. The app exposes reasons like long eye closure, multiple yawns, and head droop so alerts are not just "the model says so".

Explainability also helps with debugging. If the system behaves badly, you can inspect the triggering signals instead of guessing.

## 9) Why do you keep emergency email and screenshot capture?
Those features are for the most severe state only. When the system reaches `CRITICAL`, the goal is to preserve evidence and notify a contact quickly.

The screenshot provides context, and the email gives a remote notification path if the driver cannot respond. That makes the system more useful as a safety prototype rather than just a dashboard demo.

## 10) Why throttle heavy model inference?
The app needs to stay responsive in real time, so not every expensive model runs on every frame. The runtime uses inference intervals and smoothing so the dashboard remains usable even on modest hardware.

That is a tradeoff: slightly delayed model updates in exchange for stable frame rate and a smoother user experience.

## 11) Why are some models optional at runtime?
The pipeline is designed to degrade gracefully. If a pre-trained model is missing, the app can still run the rest of the monitoring stack and report what is available.

That makes development and testing easier, especially when datasets or trained weights are not present on every machine.

## 12) Why does the app support both webcam mode and video mode?
Webcam mode is for live monitoring, while video mode is better for testing, demos, and benchmark evaluation. Supporting both lets you validate the same pipeline in real time and on saved footage.

That matters for reproducibility. Video mode makes it easier to compare changes across runs.

## 13) Why separate configuration into `src/config.py`?
Centralizing thresholds, model paths, timing values, and feature flags keeps the project maintainable. It reduces the chance of hidden magic numbers scattered across the codebase.

It also makes it easier to tune the system without rewriting the main logic.

## 14) Why store runtime artifacts in `results/`?
The app creates reports, logs, event timelines, and critical screenshots while it runs. Keeping those outputs in `results/` makes the workspace cleaner and keeps generated data separate from source code.

This also makes session analysis and cleanup easier.

## 15) Why does the project expose both state labels and scores?
The score is useful for trend analysis and thresholding, while the state label is easier for the user to read. Together they give both machine-friendly and human-friendly views of the same condition.

That is why the dashboard can show the current state while the report still records the underlying score history.

## 16) Why is the project focused on practical runtime behavior instead of just training accuracy?
A fatigue monitor has to work live, not just score well offline. That means latency, stability, fallback behavior, alert cooldowns, and readable explanations all matter as much as model accuracy.

The implementation reflects that by pairing model inference with monitoring logic, calibration, alerting, and reporting.

## 17) Why is the system split into detection, monitoring, training, and visualization modules?
Each layer has a different job:
- Detection extracts signals from frames.
- Monitoring turns signals into fatigue state and alerts.
- Training defines and loads the models.
- Visualization presents the results to the user.

That separation keeps the code easier to understand and safer to modify.

## 18) Why this overall design?
Because the project is trying to solve a real-time safety problem, not just a classification task. The design prioritizes speed, explainability, resilience, and operational clarity over a single elegant model.

That is the reason for the hybrid approach: small specialized models, landmark-based logic, baseline calibration, composite scoring, and visible alert reasons.
