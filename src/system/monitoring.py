import numpy as np
import time
from datetime import datetime, timedelta
from collections import deque
from src.config import (
    FATIGUE_SCORE_WEIGHTS,
    FATIGUE_SCORE_THRESHOLDS,
    DRIVER_STATES,
    STATE_CLASSIFICATION_RULES,
    BASELINE_CALIBRATION_FRAMES,
    EYE_CLOSURE_DURATION_THRESHOLD,
    YAWN_DURATION_THRESHOLD,
    HEAD_DROOP_THRESHOLD,
    MIN_CONSECUTIVE_DETECTIONS,
    TEMPORAL_SMOOTHING_WINDOW,
    EARLY_WARNING_THRESHOLD,
    BASELINE_CALIBRATION_SECONDS,
    SUDDEN_PITCH_CHANGE_THRESHOLD,
    SUDDEN_ROLL_CHANGE_THRESHOLD,
    NO_FACE_ALERT_SECONDS,
    HEAD_TILT_DROOP_THRESHOLD,
    EMERGENCY_SUDDEN_DROP_FROM,
    EMERGENCY_SUDDEN_DROP_TO,
    EMERGENCY_PROLONGED_EYE_CLOSURE_SECONDS,
    EMERGENCY_NO_MOVEMENT_SECONDS,
)


class BaselineCalibration:
    """Learns driver-specific baseline behaviors."""

    def __init__(self, calibration_frames=BASELINE_CALIBRATION_FRAMES):
        self.calibration_frames = calibration_frames
        self.calibration_seconds = BASELINE_CALIBRATION_SECONDS
        self.calibration_data = {
            'blink_rates': deque(maxlen=100),
            'head_positions': deque(maxlen=100),
            'mouth_movements': deque(maxlen=100),
        }
        self.baseline = {
            'avg_blink_rate': 0,
            'avg_head_position': (0, 0, 0),
            'avg_mouth_opening': 0,
        }
        self.is_calibrated = False
        self.frame_count = 0
        self.start_time = datetime.now()

    def update(self, eye_aspect_ratio, head_pose, mouth_aspect_ratio):
        """Update calibration with new readings."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed < self.calibration_seconds and self.frame_count < self.calibration_frames:
            self.calibration_data['blink_rates'].append(eye_aspect_ratio)
            self.calibration_data['head_positions'].append(head_pose)
            self.calibration_data['mouth_movements'].append(mouth_aspect_ratio)
            self.frame_count += 1
        else:
            if not self.is_calibrated:
                self._compute_baseline()
                self.is_calibrated = True
            # Continuous adaptation (slow update)
            self.calibration_data['blink_rates'].append(eye_aspect_ratio)
            self.calibration_data['head_positions'].append(head_pose)
            self.calibration_data['mouth_movements'].append(mouth_aspect_ratio)

            # Adaptive threshold evolution (requested):
            # NewBaseline = 0.9 * OldBaseline + 0.1 * CurrentValue
            old_blink = float(self.baseline['avg_blink_rate'])
            old_mouth = float(self.baseline['avg_mouth_opening'])
            old_head = np.array(self.baseline['avg_head_position'], dtype=np.float32)
            current_head = np.array(head_pose, dtype=np.float32)

            self.baseline['avg_blink_rate'] = float(0.9 * old_blink + 0.1 * float(eye_aspect_ratio))
            self.baseline['avg_mouth_opening'] = float(0.9 * old_mouth + 0.1 * float(mouth_aspect_ratio))
            self.baseline['avg_head_position'] = tuple((0.9 * old_head + 0.1 * current_head).tolist())

    def get_status(self):
        """Return calibration status for UI."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = min(1.0, elapsed / max(self.calibration_seconds, 0.1))
        remaining = max(0.0, self.calibration_seconds - elapsed)
        return {
            'is_calibrated': self.is_calibrated,
            'elapsed_seconds': elapsed,
            'remaining_seconds': remaining,
            'progress': progress,
        }

    def _compute_baseline(self):
        """Compute baseline from calibration data."""
        if not self.calibration_data['blink_rates']:
            return

        self.baseline['avg_blink_rate'] = np.mean(list(self.calibration_data['blink_rates']))

        head_poses = np.array(list(self.calibration_data['head_positions']))
        self.baseline['avg_head_position'] = tuple(np.mean(head_poses, axis=0))

        self.baseline['avg_mouth_opening'] = np.mean(list(self.calibration_data['mouth_movements']))

    def get_deviation(self, eye_aspect_ratio, head_pose, mouth_aspect_ratio):
        """Calculate deviation from baseline."""
        if not self.is_calibrated:
            return {'eye': 0, 'head': 0, 'mouth': 0}

        eye_deviation = abs(eye_aspect_ratio - self.baseline['avg_blink_rate']) / max(self.baseline['avg_blink_rate'], 0.01)

        head_dev = np.array(head_pose) - np.array(self.baseline['avg_head_position'])
        head_deviation = np.linalg.norm(head_dev)

        mouth_deviation = abs(mouth_aspect_ratio - self.baseline['avg_mouth_opening']) / max(self.baseline['avg_mouth_opening'], 0.01)

        return {
            'eye': eye_deviation,
            'head': head_deviation,
            'mouth': mouth_deviation,
        }


class FatigueScoring:
    """Computes composite fatigue score."""

    def __init__(self, fps=30):
        self.fps = fps
        self.fatigue_score = 0.0
        self.eye_closure_duration = 0
        self.yawn_count = 0.0
        self.head_droop_duration = 0
        self.score_history = deque(maxlen=1800)  # 60 seconds at 30 FPS
        self.previous_eye_state = True  # True = open, False = closed
        self.previous_yawn_state = False
        self.yawn_timestamps = deque(maxlen=256)

    def update(self, eye_state, is_yawning, head_droop, head_tilt=0.0, baseline_deviation=None):
        """
        Update fatigue score based on current detections.

        eye_state: bool, True if eyes open
        is_yawning: bool
        head_droop: bool
        baseline_deviation: dict with 'eye', 'head', 'mouth' keys
        """
        if baseline_deviation is None:
            baseline_deviation = {'eye': 0, 'head': 0, 'mouth': 0}

        # Track eye closure duration
        if not eye_state:
            self.eye_closure_duration += 1 / self.fps
        else:
            self.eye_closure_duration = 0

        # Track yawn events in the last minute
        now = datetime.now()
        if is_yawning and not self.previous_yawn_state:
            self.yawn_timestamps.append(now)
        self.previous_yawn_state = is_yawning

        # Keep only last 60 seconds of yawn timestamps
        while self.yawn_timestamps and (now - self.yawn_timestamps[0]).total_seconds() > 60:
            self.yawn_timestamps.popleft()
        self.yawn_count = float(len(self.yawn_timestamps))

        # Track head droop
        if head_droop:
            self.head_droop_duration += 1 / self.fps
        else:
            self.head_droop_duration = 0

        # Requested additive logic + decay
        score_delta = 0.0

        # Eye closure event (>2 sec)
        if self.eye_closure_duration > 2.0:
            score_delta += 15.0 / max(self.fps, 1)

        # Yawning frequency event (>=3 yawns in last minute)
        if self.yawn_count >= 3:
            if (not eye_state) or head_droop or head_tilt < -10.0:
                score_delta += 10.0 / max(self.fps, 1)
            else:
                score_delta += 2.0 / max(self.fps, 1)

        # Head droop/tilt event
        if self.head_droop_duration > 0.6 and (head_tilt < -15.0 or head_droop):
            score_delta += 10.0 / max(self.fps, 1)

        # Baseline-aware slight boost
        if baseline_deviation.get('head', 0) > 10 or baseline_deviation.get('eye', 0) > 0.5:
            score_delta *= 1.1

        self.fatigue_score += score_delta

        # Decay (important)
        self.fatigue_score -= 1.0 / max(self.fps, 1)
        self.fatigue_score = float(np.clip(self.fatigue_score, 0, 100))

        self.score_history.append(self.fatigue_score)

        return self.fatigue_score

    def get_trend(self):
        """Get fatigue trend (increasing, stable, decreasing)."""
        if len(self.score_history) < 20:
            return 'stable'

        recent_scores = list(self.score_history)[-10:]
        older_scores = list(self.score_history)[-20:-10]

        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)

        diff = recent_avg - older_avg
        if diff > 2:
            return 'increasing'
        elif diff < -2:
            return 'decreasing'
        else:
            return 'stable'

    def get_score_history(self, seconds=10):
        """Get score history for the last N seconds."""
        frames = int(seconds * self.fps)
        return list(self.score_history)[-frames:]


class DriverStateClassifier:
    """Classifies driver state as Alert, Distracted, or Fatigued."""

    def __init__(self):
        self.state = 'ALERT'
        self.state_history = deque(maxlen=TEMPORAL_SMOOTHING_WINDOW)

    def classify(self, fatigue_score, yawn_count, eye_openness, head_pose=None, baseline_deviation=None):
        """
        Classify driver state based on current metrics.
        
        head_pose: (pitch, yaw, roll) tuple in degrees
        Returns: state ('ALERT', 'DISTRACTED', 'FATIGUED')
        """
        if baseline_deviation is None:
            baseline_deviation = {'eye': 0, 'head': 0, 'mouth': 0}
        if head_pose is None:
            head_pose = (0, 0, 0)

        # CRITICAL: Add CRITICAL state for severe/incapacitation scenarios
        if fatigue_score >= 90:
            state = 'CRITICAL'
        elif fatigue_score < 10:
            state = 'ALERT'
        elif fatigue_score < 25:
            state = 'MILD FATIGUE'
        elif fatigue_score < 40:
            state = 'MODERATE FATIGUE'
        else:
            state = 'SEVERE FATIGUE'

        self.state_history.append(state)

        # Apply temporal smoothing (majority voting)
        if len(self.state_history) >= TEMPORAL_SMOOTHING_WINDOW:
            state_counts = {
                'ALERT': sum(1 for s in self.state_history if s == 'ALERT'),
                'DISTRACTED': sum(1 for s in self.state_history if s == 'DISTRACTED'),
                'MILD FATIGUE': sum(1 for s in self.state_history if s == 'MILD FATIGUE'),
                'MODERATE FATIGUE': sum(1 for s in self.state_history if s == 'MODERATE FATIGUE'),
                'SEVERE FATIGUE': sum(1 for s in self.state_history if s == 'SEVERE FATIGUE'),
                'CRITICAL': sum(1 for s in self.state_history if s == 'CRITICAL'),
            }
            self.state = max(state_counts, key=state_counts.get)
        else:
            self.state = state

        return self.state

    def _get_score_level(self, score):
        """Categorize score into level."""
        if score < FATIGUE_SCORE_THRESHOLDS['low']:
            return 'low'
        elif score < FATIGUE_SCORE_THRESHOLDS['moderate']:
            return 'moderate'
        elif score < FATIGUE_SCORE_THRESHOLDS['high']:
            return 'high'
        else:
            return 'critical'

    def get_confidence(self):
        """Get classification confidence based on history consistency."""
        if not self.state_history:
            return 0.0

        current_state_count = sum(1 for s in self.state_history if s == self.state)
        return current_state_count / len(self.state_history)


class AlertSystem:
    """Generates explainable alerts based on driver state."""

    def __init__(self, fps=30):
        self.fps = fps
        self.last_alert_time = None
        self.alert_cooldown = 5  # seconds
        self.consecutive_detections = {}
        self.ongoing_issues = {}

    def check_alert(self, fatigue_score, eye_closure_duration, yawn_count, head_droop, state, head_tilt=0.0, immediate_reason=None):
        """
        Check if alert should be triggered.
        Returns: (should_alert, alert_text, reasons)
        """
        reasons = []
        should_alert = False

        if immediate_reason:
            reasons.append(immediate_reason)
            should_alert = True

        # Explainable reason collection
        if eye_closure_duration > 2.0:
            reasons.append(f"Eye closure for {eye_closure_duration:.2f} seconds")
            self.consecutive_detections['eye_closure'] = self.consecutive_detections.get('eye_closure', 0) + 1
        else:
            self.consecutive_detections['eye_closure'] = 0

        if yawn_count >= 3:
            reasons.append(f"Multiple yawns detected ({int(yawn_count)} in last minute)")
            self.consecutive_detections['yawning'] = self.consecutive_detections.get('yawning', 0) + 1
        else:
            self.consecutive_detections['yawning'] = 0

        if head_droop or head_tilt < -15.0:
            reasons.append(f"Head droop detected (tilt {head_tilt:.1f}°)")
            self.consecutive_detections['head_droop'] = self.consecutive_detections.get('head_droop', 0) + 1
        else:
            self.consecutive_detections['head_droop'] = 0

        # Apply multi-condition validation (false alert reduction)
        indicator_count = 0
        if eye_closure_duration > 2.0:
            indicator_count += 1
        if yawn_count >= 3:
            indicator_count += 1
        if head_droop or head_tilt < -15.0:
            indicator_count += 1

        if fatigue_score > 40 and indicator_count >= 2:
            reasons.append(f"Fatigue score high: {fatigue_score:.1f}/100")
            should_alert = True

        if state in ('SEVERE FATIGUE', 'CRITICAL'):
            reasons.append(f"Driver classified as: {state}")
            should_alert = True

        consecutive_confirmed = any(
            count >= MIN_CONSECUTIVE_DETECTIONS
            for count in self.consecutive_detections.values()
        )

        if not should_alert and state in ('MODERATE FATIGUE', 'SEVERE FATIGUE', 'CRITICAL') and indicator_count >= 2 and consecutive_confirmed:
            should_alert = True

        if not should_alert:
            return False, "", []

        # CRITICAL FIX: Separate cooldown per alert type to prevent blocking other alerts
        # Different alert categories use different cooldown keys
        alert_category = 'critical' if state == 'CRITICAL' else 'fatigue' if state in ('SEVERE FATIGUE', 'MODERATE FATIGUE') else 'general'
        cooldown_key = f'{alert_category}_alert_time'
        
        last_category_alert = getattr(self, cooldown_key, None)
        if last_category_alert:
            time_since_alert = (datetime.now() - last_category_alert).total_seconds()
            if time_since_alert < self.alert_cooldown:
                # Same category cooldown still active; suppress
                return False, "", []
        
        # Alert triggered and passed cooldown - update category timer
        setattr(self, cooldown_key, datetime.now())

        if state == 'CRITICAL':
            alert_text = "CRITICAL: DRIVER INCAPACITATED"
        elif state == 'SEVERE FATIGUE':
            alert_text = "SEVERE FATIGUE DETECTED"
        elif fatigue_score > 40:
            alert_text = "FATIGUE ALERT"
        else:
            alert_text = "ATTENTION REQUIRED"

        return True, alert_text, reasons

    def get_explainable_alert(self, alert_text, reasons):
        """Format alert with explanations."""
        explanation = f"{alert_text}\n\nReasons:\n"
        for reason in reasons:
            explanation += f"  • {reason}\n"

        return f"\n{explanation}\nPlease take a break and rest.\n"


class DriverMonitoringSystem:
    """Integrates all components for comprehensive driver monitoring."""

    def __init__(self, fps=30):
        self.fps = fps
        self.baseline = BaselineCalibration()
        self.fatigue_scorer = FatigueScoring(fps)
        self.state_classifier = DriverStateClassifier()
        self.alert_system = AlertSystem(fps)
        self.last_head_pose = (0.0, 0.0, 0.0)
        self.no_face_duration = 0.0
        self.no_movement_duration = 0.0
        self.eye_closed_duration_emergency = 0.0
        self.head_drop_duration_emergency = 0.0
        self.last_head_tilt = 0.0
        self.last_eye_aspect_ratio = 0.0
        self.last_mouth_aspect_ratio = 0.0
        self.last_face_center = None
        self.no_face_started_at = None
        self.was_emergency_last_frame = False
        self.pose_distraction_duration = 0.0
        self.detection_buffer = {
            'eye_closed': deque(maxlen=MIN_CONSECUTIVE_DETECTIONS),
            'yawning': deque(maxlen=MIN_CONSECUTIVE_DETECTIONS),
            'head_droop': deque(maxlen=MIN_CONSECUTIVE_DETECTIONS),
        }

    def _calibration_only_state(self, eye_state, is_yawning, head_pose, eye_aspect_ratio, mouth_aspect_ratio):
        calibration_status = self.baseline.get_status()
        return {
            'fatigue_score': 0.0,
            'eye_state': eye_state,
            'eye_aspect_ratio': eye_aspect_ratio,
            'eye_closure_duration': 0.0,
            'is_yawning': is_yawning,
            'yawn_count': 0.0,
            'head_pose': head_pose,
            'head_tilt': 0.0,
            'head_droop': False,
            'mouth_aspect_ratio': mouth_aspect_ratio,
            'state': 'CALIBRATING',
            'should_alert': False,
            'alert_text': '',
            'alert_reasons': [],
            'fatigue_trend': 'stable',
            'early_warning': False,
            'baseline_calibrated': False,
            'calibration_progress': calibration_status['progress'],
            'calibration_remaining_seconds': calibration_status['remaining_seconds'],
            'status_message': 'Custom calibration is now running. Please keep your normal posture.',
            'sudden_event': False,
            'no_face_duration': self.no_face_duration,
            'monitoring_confidence': 0.0,
            'emergency_flag': False,
            'critical_alarm': False,
            'no_movement_duration': self.no_movement_duration,
            'state_reason': 'Calibration in progress to learn your baseline behavior',
            'state_reasons': ['Calibration in progress to learn your baseline behavior'],
        }

    def update_missing_face(self):
        """Handle frames where face is not detected."""
        now = time.time()
        if self.no_face_started_at is None:
            self.no_face_started_at = now
        self.no_face_duration = max(0.0, now - self.no_face_started_at)
        # CRITICAL: Reset fatigue and timers when face is lost
        self.fatigue_scorer.fatigue_score = 0.0
        self.fatigue_scorer.eye_closure_duration = 0.0
        self.fatigue_scorer.yawn_count = 0.0
        self.fatigue_scorer.head_droop_duration = 0.0
        self.last_eye_aspect_ratio = 0.0
        self.last_mouth_aspect_ratio = 0.0
        self.eye_closed_duration_emergency = 0.0
        self.head_drop_duration_emergency = 0.0
        self.was_emergency_last_frame = False
        
        prolonged_missing_face = self.no_face_duration >= EMERGENCY_NO_MOVEMENT_SECONDS
        should_alert = self.no_face_duration >= NO_FACE_ALERT_SECONDS
        if prolonged_missing_face:
            alert_text = f'CRITICAL: DRIVER OUT OF FRAME > {EMERGENCY_NO_MOVEMENT_SECONDS:.0f}s'
            reasons = [f'Face not detected for {self.no_face_duration:.1f} seconds']
            state = 'CRITICAL'
            emergency_flag = True
        else:
            alert_text = 'FACE LOST - PLEASE FACE THE CAMERA' if should_alert else ''
            reasons = ['Face not detected continuously'] if should_alert else []
            state = 'TRACKING_LOST'
            emergency_flag = False
        return {
            'fatigue_score': 0.0,
            'eye_state': None,
            'eye_aspect_ratio': None,
            'eye_closure_duration': 0.0,
            'is_yawning': None,
            'yawn_count': 0.0,
            'head_pose': (None, None, None),
            'head_tilt': None,
            'head_droop': None,
            'mouth_aspect_ratio': None,
            'state': state,
            'should_alert': should_alert,
            'alert_text': alert_text,
            'alert_reasons': reasons,
            'fatigue_trend': self.fatigue_scorer.get_trend(),
            'early_warning': False,
            'baseline_calibrated': self.baseline.is_calibrated,
            'calibration_progress': self.baseline.get_status()['progress'],
            'calibration_remaining_seconds': self.baseline.get_status()['remaining_seconds'],
            'status_message': 'No face detected. Re-align to camera.',
            'sudden_event': False,
            'no_face_duration': self.no_face_duration,
            'monitoring_confidence': 0.0,
            'emergency_flag': emergency_flag,
            'critical_alarm': emergency_flag,
            'no_movement_duration': self.no_movement_duration,
            'state_reason': reasons[0] if reasons else 'Face not detected continuously',
            'state_reasons': reasons if reasons else ['Face not detected continuously'],
        }

    def update(self, eye_state, is_yawning, head_pose, eye_aspect_ratio, mouth_aspect_ratio,
               head_tilt=0.0, monitoring_confidence=0.0, face_center=None,
               distraction_class=None, distraction_confidence=None):
        """
        Update the monitoring system with new detections.

        eye_state: bool, True if eyes open
        is_yawning: bool
        head_pose: (pitch, yaw, roll) tuple
        eye_aspect_ratio: float
        mouth_aspect_ratio: float

        Returns: state_dict with all relevant information
        """
        had_no_face = self.no_face_duration >= NO_FACE_ALERT_SECONDS
        self.no_face_duration = 0.0
        self.no_face_started_at = None

        # Emergency eye-closure timer (independent of fatigue logic)
        if not eye_state:
            self.eye_closed_duration_emergency += 1.0 / max(self.fps, 1)
        else:
            self.eye_closed_duration_emergency = 0.0

        pitch, yaw, roll = head_pose
        baseline_head_pose = self.baseline.baseline.get('avg_head_position', (0.0, 0.0, 0.0))
        baseline_pitch_reference = float(baseline_head_pose[0])
        pitch_delta_for_emergency = float(pitch) - baseline_pitch_reference

        if self.baseline.is_calibrated:
            deep_tilt_drop = head_tilt < -24.0
            moderate_combo_drop = (pitch_delta_for_emergency < -12.0) and (head_tilt < -10.0)
            deep_pitch_drop = (pitch_delta_for_emergency < -16.0) and (abs(float(yaw)) < 22.0)
            head_drop_condition = deep_tilt_drop or moderate_combo_drop or deep_pitch_drop
        else:
            # During calibration, avoid false emergency escalation from unstable early pose estimates.
            head_drop_condition = False

        if head_drop_condition:
            self.head_drop_duration_emergency += 1.0 / max(self.fps, 1)
        else:
            self.head_drop_duration_emergency = 0.0

        # CRITICAL: Sudden collapse detection (eyes closed >5 seconds)
        emergency_flag = self.eye_closed_duration_emergency >= EMERGENCY_PROLONGED_EYE_CLOSURE_SECONDS
        if emergency_flag:
            # Immediate fatigue boost for emergency
            self.fatigue_scorer.fatigue_score = min(100.0, self.fatigue_scorer.fatigue_score + (5.0 / max(self.fps, 1)))

        # Motion tracking for incapacitation detection
        motion_detected = False
        face_shift = 0.0
        if self.last_face_center is not None and face_center is not None:
            face_shift = float(np.linalg.norm(np.array(face_center) - np.array(self.last_face_center)))
            if face_shift > 12.0:
                motion_detected = True

        if abs(head_tilt - self.last_head_tilt) > 7.0:
            motion_detected = True
        if abs(eye_aspect_ratio - self.last_eye_aspect_ratio) > 0.06:
            motion_detected = True
        if abs(mouth_aspect_ratio - self.last_mouth_aspect_ratio) > 0.06:
            motion_detected = True

        if motion_detected:
            self.no_movement_duration = max(0.0, self.no_movement_duration - (0.5 / max(self.fps, 1)))
        else:
            self.no_movement_duration += 1.0 / max(self.fps, 1)

        sudden_drop = self.last_head_tilt > EMERGENCY_SUDDEN_DROP_FROM and head_tilt < EMERGENCY_SUDDEN_DROP_TO
        prolonged_closure = self.eye_closed_duration_emergency >= EMERGENCY_PROLONGED_EYE_CLOSURE_SECONDS
        prolonged_head_drop = self.head_drop_duration_emergency >= EMERGENCY_PROLONGED_EYE_CLOSURE_SECONDS

        emergency = prolonged_closure or prolonged_head_drop
        emergency_sudden_event = sudden_drop or emergency

        # Do not enter CRITICAL while calibration is still running.
        if not self.baseline.is_calibrated:
            emergency = False

        # Emergency recovery: transition OUT of critical immediately when emergency condition clears
        emergency_just_cleared = self.was_emergency_last_frame and not emergency
        
        # Full recovery: reset fatigue accumulation when eyes reopen and posture improves
        recovered_from_emergency = (
            emergency_just_cleared
            and eye_state
            and float(eye_aspect_ratio) > 0.22
            and head_tilt > -12.0
        )
        
        if recovered_from_emergency:
            # Full reset on solid recovery
            self.fatigue_scorer.fatigue_score = 0.0
            self.fatigue_scorer.eye_closure_duration = 0.0
            self.eye_closed_duration_emergency = 0.0
            self.head_drop_duration_emergency = 0.0
            self.state_classifier.state_history.clear()
        elif emergency_just_cleared:
            # Partial reset when emergency ends but posture is still recovering
            self.fatigue_scorer.fatigue_score = min(40.0, self.fatigue_scorer.fatigue_score)
            self.eye_closed_duration_emergency = 0.0
            self.head_drop_duration_emergency = 0.0

        # update references
        self.last_head_tilt = head_tilt
        self.last_eye_aspect_ratio = eye_aspect_ratio
        self.last_mouth_aspect_ratio = mouth_aspect_ratio
        self.last_face_center = face_center

        # Update baseline
        self.baseline.update(eye_aspect_ratio, head_pose, mouth_aspect_ratio)
        baseline_dev = self.baseline.get_deviation(eye_aspect_ratio, head_pose, mouth_aspect_ratio)

        if not self.baseline.is_calibrated:
            self.was_emergency_last_frame = False
            self.last_head_pose = head_pose
            cal_state = self._calibration_only_state(
                eye_state=eye_state,
                is_yawning=is_yawning,
                head_pose=head_pose,
                eye_aspect_ratio=eye_aspect_ratio,
                mouth_aspect_ratio=mouth_aspect_ratio,
            )
            # After calibration is done, mark it in state so it doesn't show again
            if self.baseline.is_calibrated:
                cal_state['baseline_calibrated'] = True
            return cal_state

        # Detect head droop relative to driver baseline (reduces fixed-angle false positives)
        baseline_pitch = float(self.baseline.baseline.get('avg_head_position', (0.0, 0.0, 0.0))[0])
        pitch_delta = float(pitch) - baseline_pitch
        head_droop = (head_tilt < -20.0) or (pitch_delta < -12.0 and head_tilt < -5.0)

        previous_pitch, _, previous_roll = self.last_head_pose
        sudden_pitch_change = abs(pitch - previous_pitch) > SUDDEN_PITCH_CHANGE_THRESHOLD
        sudden_roll_change = abs(roll - previous_roll) > SUDDEN_ROLL_CHANGE_THRESHOLD
        sudden_event = sudden_pitch_change or sudden_roll_change or emergency_sudden_event
        self.last_head_pose = head_pose

        # Update fatigue score
        fatigue_score = self.fatigue_scorer.update(
            eye_state,
            is_yawning,
            head_droop,
            head_tilt=head_tilt,
            baseline_deviation=baseline_dev,
        )

        # Classify state
        yawn_count = self.fatigue_scorer.yawn_count
        # Classifier rules are normalized in [0,1], so feed semantic openness.
        eye_openness = 1.0 if eye_state else 0.0
        state = self.state_classifier.classify(
            fatigue_score,
            yawn_count,
            eye_openness,
            head_pose=head_pose,  # ADDED: Now passes head pose for distraction detection
            baseline_deviation=baseline_dev,
        )

        pose_distraction_now = abs(float(yaw)) > 20.0 or abs(float(roll)) > 20.0
        if pose_distraction_now:
            self.pose_distraction_duration += 1.0 / max(self.fps, 1)
        else:
            self.pose_distraction_duration = max(0.0, self.pose_distraction_duration - (1.5 / max(self.fps, 1)))

        # Context-aware distraction state override when fatigue is not severe
        if state in ('ALERT', 'MILD FATIGUE', 'MODERATE FATIGUE'):
            model_distraction = (
                distraction_class is not None
                and distraction_class != 'safe'
                and (distraction_confidence is not None and float(distraction_confidence) >= 0.60)
            )
            pose_distraction = self.pose_distraction_duration >= 0.6
            if model_distraction or pose_distraction:
                state = 'DISTRACTED'

        state_reasons = []
        if state == 'DISTRACTED':
            if distraction_class not in (None, 'safe') and distraction_confidence is not None:
                state_reasons.append(f"Distraction detected: {distraction_class} ({float(distraction_confidence) * 100:.0f}%)")
            if abs(float(yaw)) > 18.0 or abs(float(roll)) > 18.0:
                state_reasons.append(f"Head turned away (yaw {float(yaw):.1f}°, roll {float(roll):.1f}°)")
        if state in ('MILD FATIGUE', 'MODERATE FATIGUE', 'SEVERE FATIGUE', 'CRITICAL'):
            if self.fatigue_scorer.eye_closure_duration > 1.0:
                state_reasons.append(f"Eyes closed for {self.fatigue_scorer.eye_closure_duration:.1f}s")
            if yawn_count >= 1:
                state_reasons.append(f"Yawn frequency elevated ({int(yawn_count)} in last minute)")
            if head_droop:
                state_reasons.append(f"Head droop pattern detected (tilt {head_tilt:.1f}°)")
            state_reasons.append(f"Fatigue score is {fatigue_score:.1f}/100")
        if state == 'ALERT':
            state_reasons.append('Low fatigue score with stable eyes and head pose')

        trend = self.fatigue_scorer.get_trend()
        early_warning = (
            trend == 'increasing'
            and fatigue_score >= FATIGUE_SCORE_THRESHOLDS['critical'] * EARLY_WARNING_THRESHOLD
            and state != 'SEVERE FATIGUE'
        )

        # High-priority incapacitation check (overrides fatigue)
        if emergency:
            state = 'CRITICAL'
            reasons = []
            if prolonged_closure:
                reasons.append(f"No eye movement detected ({self.eye_closed_duration_emergency:.1f} sec)")
            if prolonged_head_drop or sudden_drop:
                reasons.append(f"Head drop detected ({self.head_drop_duration_emergency:.1f} sec)")

            should_alert = True
            alert_text = "⚠ CRITICAL ALERT ⚠ Driver Unresponsive Detected"
            state_reasons = reasons[:] if reasons else ['Emergency condition detected']
        elif recovered_from_emergency:
            # Recently recovered from emergency - reset state to ALERT
            state = 'ALERT'
            should_alert = False
            alert_text = ''
            reasons = []
            state_reasons = ['Driver posture recovered - emergency cleared']
        else:
            # Check normal fatigue alerts
            should_alert, alert_text, reasons = self.alert_system.check_alert(
                fatigue_score,
                self.fatigue_scorer.eye_closure_duration,
                yawn_count,
                head_droop,
                state,
                head_tilt=head_tilt,
                immediate_reason='Sudden head movement/drop detected' if sudden_event else None,
            )

        if had_no_face and not emergency:
            self.state_classifier.state_history.clear()
            self.fatigue_scorer.eye_closure_duration = 0.0
            self.eye_closed_duration_emergency = 0.0
            self.head_drop_duration_emergency = 0.0
            state = 'ALERT'
            should_alert = False
            alert_text = ''
            reasons = []
            state_reasons = ['Driver returned to frame - tracking restored']

        self.was_emergency_last_frame = emergency

        calibration_status = self.baseline.get_status()

        return {
            'fatigue_score': fatigue_score,
            'eye_state': eye_state,
            'eye_aspect_ratio': eye_aspect_ratio,
            'eye_closure_duration': self.fatigue_scorer.eye_closure_duration,
            'is_yawning': is_yawning,
            'yawn_count': yawn_count,
            'head_pose': head_pose,
            'head_pose_delta': (
                float(pitch) - float(baseline_head_pose[0]),
                float(yaw) - float(baseline_head_pose[1]),
                float(roll) - float(baseline_head_pose[2]),
            ),
            'baseline_head_pose': (
                float(baseline_head_pose[0]),
                float(baseline_head_pose[1]),
                float(baseline_head_pose[2]),
            ),
            'head_tilt': head_tilt,
            'head_droop': head_droop,
            'mouth_aspect_ratio': mouth_aspect_ratio,
            'state': state,
            'should_alert': should_alert,
            'alert_text': alert_text,
            'alert_reasons': reasons,
            'fatigue_trend': trend,
            'early_warning': early_warning,
            'baseline_calibrated': self.baseline.is_calibrated,
            'calibration_progress': calibration_status['progress'],
            'calibration_remaining_seconds': calibration_status['remaining_seconds'],
            'status_message': 'Monitoring active',
            'sudden_event': sudden_event,
            'no_face_duration': self.no_face_duration,
            'monitoring_confidence': float(np.clip(monitoring_confidence, 0, 100)),
            'emergency_flag': emergency,
            'critical_alarm': emergency,
            'no_movement_duration': self.no_movement_duration,
            'state_reason': state_reasons[0] if state_reasons else 'Monitoring active',
            'state_reasons': state_reasons,
        }
