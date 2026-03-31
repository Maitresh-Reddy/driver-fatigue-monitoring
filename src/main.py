import cv2
import numpy as np
import time
import tensorflow as tf
import json
from collections import deque
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import platform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import *
from src.detection import FaceDetector, EyeExtractor, MouthExtractor, HeadPoseEstimator
from src.training import EyeStateModel, YawDetectionModel
from src.system import DriverMonitoringSystem
from src.utils import (
    Visualizer,
    AlertLogger,
    PerformanceMonitor,
    EventTimelineLogger,
    ResourceUsageMonitor,
    EmergencyAlertNotifier,
)


class DriverFatigueMonitoringPipeline:
    """Real-time driver fatigue monitoring system."""

    def __init__(self, eye_model_path=None, yawn_model_path=None, distraction_model_path=None, drowsiness_model_path=None):
        print("Initializing Driver Fatigue Monitoring System...")

        # Initialize detectors
        self.face_detector = FaceDetector()
        self.eye_extractor = EyeExtractor(eye_size=CNN_INPUT_SIZE)
        self.mouth_extractor = MouthExtractor(mouth_size=CNN_INPUT_SIZE)
        self.head_pose_estimator = HeadPoseEstimator()

        # Initialize CNN models
        self.eye_model = EyeStateModel(CNN_INPUT_SIZE)
        self.yawn_model = YawDetectionModel(CNN_INPUT_SIZE)
        self.distraction_model = None
        self.drowsiness_model = None
        self.distraction_labels = ['safe', 'texting', 'phone', 'radio', 'drinking', 'reaching', 'makeup', 'passenger']

        # Load pre-trained models if available
        if eye_model_path and Path(eye_model_path).exists():
            print(f"Loading eye model from {eye_model_path}")
            self.eye_model.load(eye_model_path)
        else:
            print("Warning: Eye model not found. Using untrained model.")

        if yawn_model_path and Path(yawn_model_path).exists():
            print(f"Loading yawn model from {yawn_model_path}")
            self.yawn_model.load(yawn_model_path)
        else:
            print("Warning: Yawn model not found. Using untrained model.")

        if distraction_model_path and Path(distraction_model_path).exists():
            print(f"Loading distraction model from {distraction_model_path}")
            self.distraction_model = tf.keras.models.load_model(distraction_model_path)
            try:
                output_dim = int(self.distraction_model.output_shape[-1])
            except Exception:
                output_dim = 8

            if output_dim == 2:
                self.distraction_labels = ['safe', 'distracted']
            else:
                self.distraction_labels = ['safe', 'texting', 'phone', 'radio', 'drinking', 'reaching', 'makeup', 'passenger']
        else:
            print("Warning: Distraction model not found.")

        if drowsiness_model_path and Path(drowsiness_model_path).exists():
            print(f"Loading drowsiness model from {drowsiness_model_path}")
            self.drowsiness_model = tf.keras.models.load_model(drowsiness_model_path)
        else:
            print("Warning: Drowsiness model not found.")

        # Initialize monitoring system
        self.monitoring_system = DriverMonitoringSystem(fps=FPS_TARGET)

        # Initialize utilities
        self.visualizer = Visualizer(self)
        self.alert_logger = AlertLogger(LOG_FILE)
        self.performance_monitor = PerformanceMonitor()
        self.timeline_logger = EventTimelineLogger(RESULTS_DIR / 'event_timeline.jsonl')
        self.resource_monitor = ResourceUsageMonitor()
        self.emergency_notifier = EmergencyAlertNotifier(EMERGENCY_ALERT_CONFIG, self.timeline_logger)
        self.emergency_settings_file = RESULTS_DIR / 'emergency_settings.json'
        self.emergency_settings = self._load_emergency_settings()
        self.emergency_settings_open = False
        self.emergency_settings_field_index = 0
        self.last_state_dict = None
        self.last_face_bbox = None
        self.fatigue_timeline = []
        self.timestamp_timeline = []
        self.yawn_positive_frames = 0
        self.yawn_candidate_frames = 0
        self.frame_index = 0
        self.last_distraction_prediction = None
        self.last_drowsy_probability = None
        self.last_yawn_prediction = None
        self.distraction_positive_frames = 0
        self.drowsy_positive_frames = 0
        self.distraction_signal_history = deque(maxlen=15)
        self.stable_distraction_class = 'safe'
        self.smoothed_drowsy_probability = None
        self.eye_model_disabled = True
        self.yawn_model_disabled = True
        self.window_name = 'Driver Fatigue Monitoring'
        self.capture_active = False
        self.request_exit = False
        self.request_report_exit = False
        self.report_overlay_data = None
        self.frame_drop_count = 0
        self.session_start_time = datetime.now()
        self.last_alert_active = False
        self.active_alert_start_time = None
        self.last_recovery_time_seconds = None
        self.last_state_name = None
        self.last_critical_active = False
        self.last_critical_screenshot_time = None
        self.critical_episode_screenshot_taken = False
        self.critical_episode_email_sent = False
        self.screenshot_last_status = 'NONE'
        self.screenshot_last_message = 'No critical screenshot yet'
        self.screenshot_last_time = None
        self.screenshot_last_path = None
        self.emergency_email_last_status = 'NOT SENT'
        self.emergency_email_last_message = 'No emergency email attempted yet'
        self.emergency_email_last_time = None
        self.system_check_status = None
        self.self_check_visible_until = None
        self.failsafe_mode = False
        self.model_unstable_counter = 0
        self.drowsy_delta_history = deque(maxlen=25)
        self.distraction_conf_history = deque(maxlen=25)

        self.model_quality = self._load_model_quality()
        self.enable_distraction_alerts = self.model_quality.get('distraction', 0.0) >= MODEL_DEPLOYMENT_MIN_ACCURACY
        self.enable_drowsiness_alerts = self.model_quality.get('drowsiness', 0.0) >= MODEL_DEPLOYMENT_MIN_ACCURACY
        self.safe_mode_active = not (self.enable_distraction_alerts and self.enable_drowsiness_alerts)

    def _load_emergency_settings(self):
        defaults = {
            'enabled': bool(EMERGENCY_ALERT_CONFIG.get('enabled', False)),
            'smtp_server': str(EMERGENCY_ALERT_CONFIG.get('smtp_server', '')).strip() or 'smtp.gmail.com',
            'smtp_port': str(EMERGENCY_ALERT_CONFIG.get('smtp_port', '587')),
            'smtp_use_tls': bool(EMERGENCY_ALERT_CONFIG.get('smtp_use_tls', True)),
            'smtp_username': str(EMERGENCY_ALERT_CONFIG.get('smtp_username', '')),
            'smtp_password': str(EMERGENCY_ALERT_CONFIG.get('smtp_password', '')),
            'from_email': str(EMERGENCY_ALERT_CONFIG.get('from_email', '')),
            'to_email': str(EMERGENCY_ALERT_CONFIG.get('to_email', '')),
            'driver_id': str(EMERGENCY_ALERT_CONFIG.get('driver_id', 'UNKNOWN_DRIVER')),
            'vehicle_id': str(EMERGENCY_ALERT_CONFIG.get('vehicle_id', 'UNKNOWN_VEHICLE')),
            'manual_latitude': str(EMERGENCY_ALERT_CONFIG.get('manual_latitude', '')),
            'manual_longitude': str(EMERGENCY_ALERT_CONFIG.get('manual_longitude', '')),
            'manual_location_text': str(EMERGENCY_ALERT_CONFIG.get('manual_location_text', '')),
            'allow_ip_geolocation': bool(EMERGENCY_ALERT_CONFIG.get('allow_ip_geolocation', True)),
        }
        try:
            if self.emergency_settings_file.exists():
                with open(self.emergency_settings_file, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    for key, value in defaults.items():
                        if key == 'smtp_password':
                            continue
                        defaults[key] = payload.get(key, value)
            if str(defaults.get('smtp_server', '')).strip() == '':
                defaults['smtp_server'] = 'smtp.gmail.com'
            if str(defaults.get('smtp_port', '')).strip() == '':
                defaults['smtp_port'] = '587'
        except Exception:
            pass
        self._apply_emergency_settings(defaults)
        return defaults

    def _apply_emergency_settings(self, settings_dict):
        smtp_port_raw = str(settings_dict.get('smtp_port', '587')).strip()
        try:
            smtp_port = int(smtp_port_raw)
        except Exception:
            smtp_port = 587

        self.emergency_notifier.config.update({
            'enabled': bool(settings_dict.get('enabled', False)),
            'smtp_server': str(settings_dict.get('smtp_server', '')),
            'smtp_port': smtp_port,
            'smtp_use_tls': bool(settings_dict.get('smtp_use_tls', True)),
            'smtp_username': str(settings_dict.get('smtp_username', '')),
            'smtp_password': str(settings_dict.get('smtp_password', '')),
            'from_email': str(settings_dict.get('from_email', '')),
            'to_email': str(settings_dict.get('to_email', '')),
            'driver_id': str(settings_dict.get('driver_id', 'UNKNOWN_DRIVER')),
            'vehicle_id': str(settings_dict.get('vehicle_id', 'UNKNOWN_VEHICLE')),
            'manual_latitude': str(settings_dict.get('manual_latitude', '')),
            'manual_longitude': str(settings_dict.get('manual_longitude', '')),
            'manual_location_text': str(settings_dict.get('manual_location_text', '')),
            'allow_ip_geolocation': bool(settings_dict.get('allow_ip_geolocation', True)),
        })

    def _save_emergency_settings(self):
        try:
            self.emergency_settings_file.parent.mkdir(parents=True, exist_ok=True)
            persisted_settings = dict(self.emergency_settings)
            persisted_settings['smtp_password'] = ''
            with open(self.emergency_settings_file, 'w', encoding='utf-8') as f:
                json.dump(persisted_settings, f, indent=2)
            self._apply_emergency_settings(self.emergency_settings)
            self.timeline_logger.log('emergency_settings_updated', {
                'enabled': bool(self.emergency_settings.get('enabled', False)),
                'to_email': str(self.emergency_settings.get('to_email', '')),
            })
            return True
        except Exception as ex:
            self.timeline_logger.log('emergency_settings_update_failed', {'error': str(ex)[:180]})
            return False

    def _emergency_settings_fields(self):
        return [
            ('enabled', 'Email Alerts Enabled', 'bool'),
            ('smtp_server', 'SMTP Server', 'text'),
            ('smtp_port', 'SMTP Port', 'text'),
            ('smtp_use_tls', 'SMTP Use TLS', 'bool'),
            ('smtp_username', 'SMTP Username', 'text'),
            ('smtp_password', 'SMTP Password', 'password'),
            ('from_email', 'From Email', 'text'),
            ('to_email', 'Emergency Contact Email', 'text'),
            ('driver_id', 'Driver ID', 'text'),
            ('vehicle_id', 'Vehicle ID', 'text'),
            ('manual_latitude', 'Latitude (exact)', 'text'),
            ('manual_longitude', 'Longitude (exact)', 'text'),
            ('manual_location_text', 'Location Label', 'text'),
            ('allow_ip_geolocation', 'Fallback IP Geolocation', 'bool'),
        ]

    def _move_emergency_field(self, step):
        fields = self._emergency_settings_fields()
        if not fields:
            self.emergency_settings_field_index = 0
            return
        self.emergency_settings_field_index = (self.emergency_settings_field_index + step) % len(fields)

    def _handle_emergency_settings_key(self, key):
        fields = self._emergency_settings_fields()
        if not fields:
            return

        selected_key, _, field_type = fields[self.emergency_settings_field_index]

        if key == 27:  # ESC
            self.emergency_settings_open = False
            return
        if key in (9,):  # TAB
            self._move_emergency_field(1)
            return
        if key in (2424832,):  # LEFT
            self._move_emergency_field(-1)
            return
        if key in (2490368,):  # UP
            self._move_emergency_field(-1)
            return
        if key in (2621440,):  # DOWN
            self._move_emergency_field(1)
            return
        if key in (13, 10):  # ENTER
            self._save_emergency_settings()
            return

        if field_type == 'bool':
            if key in (32, ord(' '), ord('t'), ord('T')):
                self.emergency_settings[selected_key] = not bool(self.emergency_settings.get(selected_key, False))
            return

        current = str(self.emergency_settings.get(selected_key, ''))
        if key in (8, 127):
            self.emergency_settings[selected_key] = current[:-1]
            return
        if 32 <= key <= 126:
            self.emergency_settings[selected_key] = (current + chr(key))[:120]

    @staticmethod
    def _prepare_face_input(frame, face_bbox):
        if face_bbox is None:
            return None
        x, y, w, h = face_bbox
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CNN_INPUT_SIZE, CNN_INPUT_SIZE)).astype(np.float32) / 255.0
        return np.expand_dims(resized, axis=0)

    @staticmethod
    def _prepare_frame_input(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CNN_INPUT_SIZE, CNN_INPUT_SIZE)).astype(np.float32) / 255.0
        return np.expand_dims(resized, axis=0)

    @staticmethod
    def _load_model_quality():
        try:
            path = RESULTS_DIR / 'training_results.json'
            if not path.exists():
                return {}
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            quality = {}
            for key in ('distraction', 'drowsiness', 'eye', 'yawn'):
                if key in data and isinstance(data[key], dict):
                    quality[key] = float(data[key].get('accuracy', 0.0))
            return quality
        except Exception:
            return {}

    @staticmethod
    def _fuse_eye_open_state(class_idx, confidence, ear_value):
        model_open = (class_idx == 0)
        ear_open = ear_value > (EYE_ASPECT_RATIO_THRESHOLD + 0.03)

        if confidence >= EYE_MODEL_CONFIDENCE_THRESHOLD_HIGH:
            return model_open
        if confidence <= EYE_MODEL_CONFIDENCE_THRESHOLD_LOW:
            return ear_open
        if model_open == ear_open:
            return model_open
        return ear_value > EYE_ASPECT_RATIO_THRESHOLD

    def _play_voice_alert(self, text):
        try:
            if platform.system().lower() == 'windows':
                import winsound
                freq = 1400 if 'critical' in text.lower() else 1100
                winsound.Beep(freq, 220)
            else:
                print('\a', end='')
        except Exception:
            pass

    def _save_critical_screenshot(self, frame, state_dict):
        try:
            out_dir = RESULTS_DIR / 'critical_events'
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            risk_val = int(float(state_dict.get('driver_risk_index', 0.0)))
            file_path = out_dir / f'critical_{ts}_risk{risk_val}.png'
            written = cv2.imwrite(str(file_path), frame)
            if not written:
                self.screenshot_last_status = 'FAILED'
                self.screenshot_last_message = 'Screenshot write failed'
                self.screenshot_last_time = datetime.now().isoformat()
                return None
            self.last_critical_screenshot_time = time.time()
            self.screenshot_last_status = 'CAPTURED'
            self.screenshot_last_path = str(file_path)
            self.screenshot_last_time = datetime.now().isoformat()
            self.screenshot_last_message = f'Captured {Path(file_path).name}'
            self.timeline_logger.log('critical_screenshot', {'path': str(file_path)})
            return str(file_path)
        except Exception as ex:
            self.screenshot_last_status = 'FAILED'
            self.screenshot_last_message = f'Screenshot error: {str(ex)[:120]}'
            self.screenshot_last_time = datetime.now().isoformat()
            return None

    def _maybe_send_emergency_email(self, state_dict, screenshot_path):
        if self.critical_episode_email_sent:
            self.emergency_email_last_status = 'SKIPPED'
            self.emergency_email_last_message = 'Already sent for current critical episode'
            return
        sent, message = self.emergency_notifier.send_critical_alert(state_dict=state_dict, screenshot_path=screenshot_path)
        self.emergency_email_last_time = datetime.now().isoformat()
        if sent:
            self.critical_episode_email_sent = True
            self.emergency_email_last_status = 'SENT'
            self.emergency_email_last_message = str(message)
            print(f"Emergency email sent: {message}")
        else:
            msg = str(message)
            lowered = msg.lower()
            if 'disabled' in lowered or 'missing smtp' in lowered:
                self.emergency_email_last_status = 'NOT CONFIGURED'
            elif 'cooldown' in lowered:
                self.emergency_email_last_status = 'COOLDOWN'
            else:
                self.emergency_email_last_status = 'FAILED'
            self.emergency_email_last_message = msg
            if 'cooldown' not in str(message).lower() and 'disabled' not in str(message).lower():
                print(f"Emergency email skipped: {message}")

    def _maybe_capture_critical_screenshot(self, frame, state_dict, was_critical_before):
        critical_active = bool(state_dict.get('state') == 'CRITICAL' or state_dict.get('emergency_flag', False))
        if not critical_active:
            self.critical_episode_screenshot_taken = False
            self.critical_episode_email_sent = False
            return None

        now_ts = time.time()
        on_transition = not was_critical_before
        if on_transition:
            self.critical_episode_screenshot_taken = False
            self.critical_episode_email_sent = False

        if not self.critical_episode_screenshot_taken:
            saved = self._save_critical_screenshot(frame, state_dict)
            if saved is not None:
                self.critical_episode_screenshot_taken = True
                self._maybe_send_emergency_email(state_dict=state_dict, screenshot_path=saved)
            return saved

        cooldown_elapsed = (
            self.last_critical_screenshot_time is None
            or (now_ts - self.last_critical_screenshot_time) >= 2.0
        )
        if cooldown_elapsed:
            saved = self._save_critical_screenshot(frame, state_dict)
            if saved is not None and not self.critical_episode_email_sent:
                self._maybe_send_emergency_email(state_dict=state_dict, screenshot_path=saved)
            return saved
        return None

    def _run_startup_self_check(self, cap):
        model_loaded_ok = (
            self.eye_model is not None
            and self.yawn_model is not None
            and self.distraction_model is not None
            and self.drowsiness_model is not None
        )
        measured_fps = cap.get(cv2.CAP_PROP_FPS)
        if measured_fps is None or measured_fps <= 1.0:
            measured_fps = FPS_TARGET

        brightness_value = 0.0
        ok, test_frame = cap.read()
        if ok and test_frame is not None:
            gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            brightness_value = float(np.mean(gray))

        lighting_label = 'GOOD' if brightness_value >= 65.0 else 'LOW'
        status = 'READY' if model_loaded_ok and lighting_label == 'GOOD' else 'DEGRADED'
        self.system_check_status = {
            'camera': 'OK' if cap.isOpened() else 'FAIL',
            'models': 'OK' if model_loaded_ok else 'PARTIAL',
            'fps': int(round(float(measured_fps))),
            'lighting': lighting_label,
            'status': status,
        }
        self.self_check_visible_until = time.time() + 12.0

        print('\nSystem Check:')
        print(f"Camera: {self.system_check_status['camera']}")
        print(f"Model Loaded: {self.system_check_status['models']}")
        print(f"FPS: {self.system_check_status['fps']}")
        print(f"Lighting: {self.system_check_status['lighting']}")
        print(f"System Status: {self.system_check_status['status']}")
        self.timeline_logger.log('system_check', self.system_check_status)

    def _compute_risk_index(self, fatigue_score):
        driving_time_seconds = max((datetime.now() - self.session_start_time).total_seconds(), 0.0)
        recent_alerts_count = self.timeline_logger.get_recent_alert_count(minutes=5)
        risk_index = (
            0.4 * float(fatigue_score) +
            0.3 * (driving_time_seconds / 60.0) +
            0.3 * float(recent_alerts_count)
        )
        risk_index = float(np.clip(risk_index, 0.0, 100.0))
        if risk_index >= 70.0:
            level = 'HIGH RISK'
        elif risk_index >= 40.0:
            level = 'MEDIUM RISK'
        else:
            level = 'LOW RISK'
        return risk_index, level, driving_time_seconds, recent_alerts_count

    def _detect_failure_modes(self, state_dict):
        issues = []
        state_name = state_dict.get('state')
        confidence = float(state_dict.get('monitoring_confidence', 0.0) or 0.0)

        if state_name in ('TRACKING_LOST',) or state_dict.get('eye_state') is None:
            issues.append('Face not clearly visible')
        if confidence < 35.0:
            issues.append('Camera blocked or poor framing')
        if confidence < 45.0:
            issues.append('Low lighting conditions')

        if self.drowsy_delta_history:
            unstable_ratio = float(sum(self.drowsy_delta_history)) / float(len(self.drowsy_delta_history))
            if unstable_ratio > 0.45:
                issues.append('CNN predictions unstable')

        unreliable = len(issues) > 0
        return unreliable, issues

    def _update_timeline_events(self, state_dict):
        state_name = state_dict.get('state', 'UNKNOWN')
        should_alert = bool(state_dict.get('should_alert', False))
        critical_active = bool(state_dict.get('state') == 'CRITICAL' or state_dict.get('emergency_flag', False))

        if self.last_state_name != state_name:
            self.timeline_logger.log('state_change', {'from': self.last_state_name, 'to': state_name})
            self.last_state_name = state_name

        if should_alert and not self.last_alert_active:
            self.active_alert_start_time = datetime.now()
            self.timeline_logger.log('alert_start', {
                'alert_text': state_dict.get('alert_text', 'ALERT'),
                'reasons': state_dict.get('alert_reasons', []),
            })
            self._play_voice_alert(str(state_dict.get('alert_text', 'Warning: Driver fatigue detected')))
        elif not should_alert and self.last_alert_active:
            if self.active_alert_start_time is not None:
                self.last_recovery_time_seconds = max((datetime.now() - self.active_alert_start_time).total_seconds(), 0.0)
                self.timeline_logger.log('alert_recovered', {
                    'recovery_time_seconds': round(self.last_recovery_time_seconds, 3),
                })
            self.active_alert_start_time = None

        if critical_active and not self.last_critical_active:
            self.timeline_logger.log('critical_start', {'reason': state_dict.get('state_reason', 'Critical condition')})
            self.critical_episode_screenshot_taken = False
        elif not critical_active and self.last_critical_active:
            self.timeline_logger.log('critical_end', {'state': state_name})
            self.critical_episode_screenshot_taken = False

        self.last_alert_active = should_alert
        self.last_critical_active = critical_active

    def _generate_session_report(self, frame_count, processed_count):
        report_time = datetime.now()
        stats = self.performance_monitor.get_stats()
        recent_alerts = self.alert_logger.get_recent_alerts(minutes=120)
        avg_cpu = float(np.mean(self.resource_monitor.cpu_history)) if self.resource_monitor.cpu_history else 0.0
        avg_ram = float(np.mean(self.resource_monitor.ram_history_mb)) if self.resource_monitor.ram_history_mb else 0.0

        report = {
            'generated_at': report_time.isoformat(),
            'session_start': self.session_start_time.isoformat(),
            'session_duration_seconds': (report_time - self.session_start_time).total_seconds(),
            'avg_fps': stats.get('avg_fps', 0.0),
            'avg_frame_time_ms': stats.get('avg_frame_time_ms', 0.0),
            'avg_inference_time_ms': stats.get('avg_inference_time_ms', 0.0),
            'frames_processed_total': int(frame_count),
            'frames_with_inference': int(processed_count),
            'frames_skipped': int(self.frame_drop_count),
            'alerts_count': int(len(recent_alerts)),
            'last_recovery_time_seconds': self.last_recovery_time_seconds,
            'avg_cpu_percent': avg_cpu,
            'avg_ram_mb': avg_ram,
            'failsafe_mode_activated': bool(self.failsafe_mode),
            'event_timeline_file': str(RESULTS_DIR / 'event_timeline.jsonl'),
        }

        out_json = RESULTS_DIR / f"session_report_{report_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        out_md = RESULTS_DIR / f"session_report_{report_time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write('# Driver Monitoring Analytics Report\n\n')
            f.write(f"- Generated: {report['generated_at']}\n")
            f.write(f"- Session Duration: {report['session_duration_seconds']:.1f} sec\n")
            f.write(f"- Average FPS: {report['avg_fps']:.2f}\n")
            f.write(f"- Inference Frames: {report['frames_with_inference']} / {report['frames_processed_total']}\n")
            f.write(f"- Frames Skipped: {report['frames_skipped']}\n")
            f.write(f"- Alerts Count: {report['alerts_count']}\n")
            f.write(f"- Last Recovery Time: {report['last_recovery_time_seconds']} sec\n")
            f.write(f"- CPU: {report['avg_cpu_percent']:.1f}%\n")
            f.write(f"- RAM: {report['avg_ram_mb']:.0f} MB\n")
            f.write(f"- Event Timeline: {report['event_timeline_file']}\n")
        return out_json, out_md, report

    def process_frame(self, frame):
        """
        Process a single frame and return detection results.

        frame: numpy array (BGR format from OpenCV)
        Returns: processed_frame, state_dict
        """
        frame_start = time.time()

        # Get fresh face landmarks first (single authoritative source per frame)
        landmarks = self.face_detector.get_face_landmarks(frame)
        if landmarks is None:
            state_dict = self.monitoring_system.update_missing_face()
            state_dict['distraction_class'] = None
            state_dict['distraction_confidence'] = None
            state_dict['drowsy_probability'] = None
            # CRITICAL: Reset model prediction buffers when face is lost
            self.yawn_positive_frames = 0
            self.yawn_candidate_frames = 0
            self.distraction_positive_frames = 0
            self.drowsy_positive_frames = 0
            self.distraction_signal_history.clear()
            self.stable_distraction_class = 'safe'

            risk_index, risk_level, driving_time_seconds, recent_alerts_count = self._compute_risk_index(
                float(state_dict.get('fatigue_score', 0.0))
            )
            state_dict['driver_risk_index'] = float(risk_index)
            state_dict['driver_risk_level'] = risk_level
            state_dict['driving_time_seconds'] = float(driving_time_seconds)
            state_dict['recent_alerts_count'] = int(recent_alerts_count)
            state_dict['recovery_time_seconds'] = self.last_recovery_time_seconds

            unreliable, issues = self._detect_failure_modes(state_dict)
            state_dict['system_unreliable'] = unreliable
            state_dict['unreliable_reasons'] = issues
            state_dict['failsafe_mode'] = bool(self.failsafe_mode)
            state_dict['emergency_email_status'] = str(self.emergency_email_last_status)
            state_dict['emergency_email_message'] = str(self.emergency_email_last_message)
            state_dict['emergency_email_time'] = self.emergency_email_last_time
            state_dict['screenshot_status'] = str(self.screenshot_last_status)
            state_dict['screenshot_message'] = str(self.screenshot_last_message)
            state_dict['screenshot_time'] = self.screenshot_last_time
            state_dict['screenshot_path'] = self.screenshot_last_path

            now_epoch = time.time()
            if self.system_check_status is not None and self.self_check_visible_until is not None and now_epoch <= self.self_check_visible_until:
                state_dict['self_check_status'] = self.system_check_status
            else:
                state_dict['self_check_status'] = None

            self.last_state_dict = state_dict
            was_critical_before = self.last_critical_active
            self._update_timeline_events(state_dict)
            self._maybe_capture_critical_screenshot(frame, state_dict, was_critical_before)
            return frame, state_dict

        face_bbox = self.face_detector.get_face_bbox_from_landmarks(landmarks, frame.shape)

        # Extract eyes
        left_eye, right_eye, _, _ = self.eye_extractor.extract_eyes(frame, landmarks)

        # Extract mouth
        mouth, _ = self.mouth_extractor.extract_mouth(frame, landmarks)

        # CRITICAL: Normalize to [0, 1] to match training preprocessing (Rescaling layer expects this)
        if left_eye is not None:
            left_eye = left_eye.astype(np.float32) / 255.0
        if right_eye is not None:
            right_eye = right_eye.astype(np.float32) / 255.0
        if mouth is not None:
            mouth = mouth.astype(np.float32) / 255.0

        # Calculate aspect ratios
        eye_aspect_ratio_left = self.eye_extractor.calculate_eye_aspect_ratio(landmarks, self.eye_extractor.LEFT_EYE_INDICES)
        eye_aspect_ratio_right = self.eye_extractor.calculate_eye_aspect_ratio(landmarks, self.eye_extractor.RIGHT_EYE_INDICES)
        eye_aspect_ratio = (eye_aspect_ratio_left + eye_aspect_ratio_right) / 2

        mouth_aspect_ratio = self.mouth_extractor.calculate_mouth_aspect_ratio(landmarks)

        self.frame_index += 1

        # Eye state: geometry-first (model was unstable/inverted in production)
        eye_state = eye_aspect_ratio >= (EYE_ASPECT_RATIO_THRESHOLD + 0.06)

        # Predict yawn (geometry-first for robust live webcam behavior)
        # Dataset class mapping: 'yawn' is index 0, 'non_yawn' is index 1
        # So: yawn_idx==0 means YAWNING (True), yawn_idx==1 means NOT_YAWNING (False)
        is_yawning = False
        if mouth is not None:
            raw_baseline_mar = float(self.monitoring_system.baseline.baseline.get('avg_mouth_opening', 0.0))
            baseline_mar = float(np.clip(raw_baseline_mar if raw_baseline_mar > 0 else 0.42, 0.36, 0.48))
            dynamic_yawn_threshold = float(max(YAWN_MAR_THRESHOLD - 0.03, baseline_mar + 0.045))
            mar_delta = float(mouth_aspect_ratio - baseline_mar)
            mar_yawn_score = float(np.clip((mouth_aspect_ratio - dynamic_yawn_threshold) / 0.14, 0.0, 1.0))

            yawn_model_prob = None
            if self.yawn_model is not None and (self.frame_index % YAWN_INFERENCE_INTERVAL == 0 or self.last_yawn_prediction is None):
                try:
                    pred = self.yawn_model.model.predict(np.expand_dims(mouth, axis=0), verbose=0)[0]
                    if len(pred) >= 2:
                        yawn_model_prob = float(pred[0])
                    else:
                        yawn_model_prob = float(pred[0])
                    self.last_yawn_prediction = yawn_model_prob
                except Exception:
                    yawn_model_prob = self.last_yawn_prediction
            else:
                yawn_model_prob = self.last_yawn_prediction

            if yawn_model_prob is None:
                yawn_combined_score = mar_yawn_score
            else:
                yawn_combined_score = float(0.40 * float(yawn_model_prob) + 0.60 * mar_yawn_score)

            strong_mar_signal = (mouth_aspect_ratio >= dynamic_yawn_threshold) and (mar_delta >= 0.03)
            very_strong_mar_signal = (mouth_aspect_ratio >= (dynamic_yawn_threshold + 0.04)) and (mar_delta >= 0.05)
            model_support = yawn_model_prob is not None and float(yawn_model_prob) >= 0.42
            yawn_signal = (
                very_strong_mar_signal
                or (strong_mar_signal and (model_support or mar_yawn_score >= 0.35))
                or (yawn_combined_score >= 0.44)
            )

            if DEBUG_YAWN and self.frame_index % 10 == 0:  # Log every 10 frames to avoid spam
                print(f"[FRAME {self.frame_index}] YAWN DBG: MAR={mouth_aspect_ratio:.3f} baseline={baseline_mar:.3f} threshold={dynamic_yawn_threshold:.3f} delta={mar_delta:.3f} mar_score={mar_yawn_score:.3f} model_prob={yawn_model_prob} combined={yawn_combined_score:.3f} strong={strong_mar_signal} very_strong={very_strong_mar_signal} signal={yawn_signal} frames={self.yawn_positive_frames}")

            if yawn_signal:
                self.yawn_candidate_frames = min(self.yawn_candidate_frames + 1, 30)
                boost = 2 if very_strong_mar_signal else 1
                self.yawn_positive_frames = min(self.yawn_positive_frames + boost, 30)
            else:
                self.yawn_candidate_frames = max(self.yawn_candidate_frames - 2, 0)
                decay_step = 2 if mouth_aspect_ratio < (baseline_mar + 0.015) else 1
                self.yawn_positive_frames = max(self.yawn_positive_frames - decay_step, 0)

            activation_frames = max(2, YAWN_MIN_POSITIVE_FRAMES)
            is_yawning = (
                self.yawn_positive_frames >= activation_frames
                or (very_strong_mar_signal and self.yawn_candidate_frames >= 1)
            )
        else:
            # When mouth can't be extracted, decay yawn state
            self.yawn_candidate_frames = max(self.yawn_candidate_frames - 1, 0)
            self.yawn_positive_frames = max(self.yawn_positive_frames - 1, 0)
            is_yawning = self.yawn_positive_frames >= max(2, YAWN_MIN_POSITIVE_FRAMES)

        face_center = None
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            face_center = (fx + fw / 2.0, fy + fh / 2.0)

        # Estimate head pose
        pitch, yaw, roll = self.head_pose_estimator.estimate_head_pose(landmarks, frame.shape)
        head_tilt = self.head_pose_estimator.calculate_head_tilt(landmarks)

        # Monitoring confidence (simple explainable confidence score)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            face_area_ratio = (fw * fh) / max(1.0, float(frame.shape[0] * frame.shape[1]))
        else:
            face_area_ratio = 0.0

        size_score = float(np.clip((face_area_ratio - 0.03) / 0.12, 0.0, 1.0))
        pose_score = float(np.clip(1.0 - ((abs(float(yaw)) + abs(float(roll))) / 95.0), 0.0, 1.0))
        light_score = float(np.clip(1.0 - abs(mean_brightness - 128.0) / 128.0, 0.0, 1.0))
        landmark_score = 1.0 if landmarks is not None and len(landmarks) >= 300 else 0.45

        monitoring_confidence = 100.0 * (
            0.35 * size_score +
            0.25 * pose_score +
            0.20 * light_score +
            0.20 * landmark_score
        )

        if not eye_state:
            strong_pose = (
                float(pitch) < -12.0
                or abs(float(yaw)) > 18.0
                or abs(float(roll)) > 18.0
                or head_tilt < -18.0
            )
            one_eye_visible_open = max(
                float(eye_aspect_ratio_left),
                float(eye_aspect_ratio_right),
            ) > (EYE_ASPECT_RATIO_THRESHOLD + 0.045)
            large_eye_asymmetry = abs(float(eye_aspect_ratio_left) - float(eye_aspect_ratio_right)) > 0.07
            unreliable_eye_closure = strong_pose and monitoring_confidence < 75.0 and (one_eye_visible_open or large_eye_asymmetry)
            if unreliable_eye_closure:
                eye_state = True

        # Additional context classifiers (distraction and drowsiness)
        distraction_class = None
        distraction_confidence = None
        drowsy_probability = None
        previous_drowsy_probability = self.last_drowsy_probability
        previous_distraction_confidence = (
            None if self.last_distraction_prediction is None else float(self.last_distraction_prediction[1])
        )

        frame_input = self._prepare_frame_input(frame)
        face_input = self._prepare_face_input(frame, face_bbox)
        if frame_input is not None and self.distraction_model is not None and self.enable_distraction_alerts:
            if self.frame_index % DISTRACTION_INFERENCE_INTERVAL == 0 or self.last_distraction_prediction is None:
                try:
                    pred = self.distraction_model.predict(frame_input, verbose=0)[0]
                    cls_idx = int(np.argmax(pred))
                    self.last_distraction_prediction = (
                        self.distraction_labels[cls_idx],
                        float(pred[cls_idx])
                    )
                except Exception as ex:
                    self.failsafe_mode = True
                    self.timeline_logger.log('failsafe_mode', {'source': 'distraction_model', 'error': str(ex)[:140]})
                    self.last_distraction_prediction = None
            if self.last_distraction_prediction is not None:
                distraction_class, distraction_confidence = self.last_distraction_prediction

        if face_input is not None and self.drowsiness_model is not None and self.enable_drowsiness_alerts:
            if self.frame_index % DROWSINESS_INFERENCE_INTERVAL == 0 or self.last_drowsy_probability is None:
                try:
                    pred = self.drowsiness_model.predict(face_input, verbose=0)[0]
                    if len(pred) >= 2:
                        self.last_drowsy_probability = float(pred[1])
                    else:
                        self.last_drowsy_probability = float(pred[0])
                except Exception as ex:
                    self.failsafe_mode = True
                    self.timeline_logger.log('failsafe_mode', {'source': 'drowsiness_model', 'error': str(ex)[:140]})
                    self.last_drowsy_probability = None
            drowsy_probability = self.last_drowsy_probability

        if drowsy_probability is not None and previous_drowsy_probability is not None:
            delta = abs(float(drowsy_probability) - float(previous_drowsy_probability))
            self.drowsy_delta_history.append(1 if delta > 0.45 else 0)
        if distraction_confidence is not None:
            if previous_distraction_confidence is not None:
                delta_conf = abs(float(distraction_confidence) - float(previous_distraction_confidence))
                self.drowsy_delta_history.append(1 if delta_conf > 0.55 else 0)
            self.distraction_conf_history.append(float(distraction_confidence))

        pose_distraction_signal = abs(float(yaw)) > 15.0 or abs(float(roll)) > 18.0
        strong_pose_distraction = abs(float(yaw)) >= 18.0 or abs(float(roll)) >= 20.0

        # Heuristic fallbacks only when validated model path is enabled
        if distraction_class is None and self.enable_distraction_alerts:
            yaw_abs = abs(float(yaw))
            roll_abs = abs(float(roll))
            if pose_distraction_signal:
                distraction_class = 'distracted'
                distraction_confidence = float(min(0.95, max(yaw_abs / 40.0, roll_abs / 45.0)))
            else:
                distraction_class = 'safe'
                distraction_confidence = float(max(0.5, 1.0 - (yaw_abs / 50.0)))

        if drowsy_probability is None and self.enable_drowsiness_alerts:
            fatigue_proxy = 0.0
            fatigue_proxy += 0.45 if not eye_state else 0.0
            fatigue_proxy += min(0.25, max(0.0, (self.yawn_positive_frames / 10.0)))
            if abs(float(head_tilt)) > 22.0 or abs(float(roll)) > 24.0:
                fatigue_proxy += 0.45
            elif abs(float(head_tilt)) > 12.0 or abs(float(roll)) > 15.0:
                fatigue_proxy += 0.25
            fatigue_proxy += min(0.1, max(0.0, (self.monitoring_system.fatigue_scorer.eye_closure_duration / 4.0)))
            drowsy_probability = float(np.clip(fatigue_proxy, 0.0, 1.0))

        if drowsy_probability is not None:
            if self.smoothed_drowsy_probability is None:
                self.smoothed_drowsy_probability = float(drowsy_probability)
            else:
                self.smoothed_drowsy_probability = float(0.82 * self.smoothed_drowsy_probability + 0.18 * float(drowsy_probability))

            baseline_offset = 0.42
            normalized_model = float(np.clip((self.smoothed_drowsy_probability - baseline_offset) / max(0.1, (1.0 - baseline_offset)), 0.0, 1.0))
            cue_score = 0.0
            if not eye_state:
                cue_score += 0.45
            if abs(float(head_tilt)) > 22.0 or abs(float(roll)) > 24.0:
                cue_score += 0.45
            elif abs(float(head_tilt)) > 10.0 or abs(float(roll)) > 12.0:
                cue_score += 0.25
            if self.yawn_positive_frames >= max(3, YAWN_MIN_POSITIVE_FRAMES):
                cue_score += 0.25
            if self.monitoring_system.fatigue_scorer.eye_closure_duration > 1.0:
                cue_score += 0.15
            cue_score = float(np.clip(cue_score, 0.0, 1.0))

            drowsy_probability = float(np.clip(0.55 * normalized_model + 0.45 * cue_score, 0.0, 1.0))

            # False-positive guard: if eyes are open, no yawn buildup, and head is upright,
            # cap drowsiness so alert does not fire in clearly alert posture.
            if (
                eye_state
                and self.yawn_positive_frames < max(3, YAWN_MIN_POSITIVE_FRAMES)
                and abs(float(head_tilt)) < 10.0
                and abs(float(roll)) < 12.0
            ):
                drowsy_probability = min(drowsy_probability, 0.44)

        if self.safe_mode_active and not SHOW_UNVALIDATED_MODEL_SIGNALS:
            distraction_class = None
            distraction_confidence = None
            drowsy_probability = None

        drowsy_posture_signal = (
            float(pitch) < -10.0
            or abs(float(head_tilt)) > 10.0
            or abs(float(roll)) > 14.0
            or self.monitoring_system.fatigue_scorer.eye_closure_duration >= 0.8
        )

        raw_distraction_signal = (
            distraction_class is not None
            and distraction_class != 'safe'
            and distraction_confidence is not None
            and float(distraction_confidence) >= 0.52
        )
        if strong_pose_distraction and monitoring_confidence >= 45.0:
            raw_distraction_signal = True
        if drowsy_posture_signal and drowsy_probability is not None and drowsy_probability >= 0.78 and not strong_pose_distraction:
            raw_distraction_signal = False

        self.distraction_signal_history.append(1 if raw_distraction_signal else 0)
        distraction_ratio = (
            float(sum(self.distraction_signal_history)) / float(len(self.distraction_signal_history))
            if self.distraction_signal_history else 0.0
        )

        if len(self.distraction_signal_history) >= 5:
            if self.stable_distraction_class == 'safe' and distraction_ratio >= 0.45:
                self.stable_distraction_class = 'distracted'
            elif self.stable_distraction_class != 'safe' and distraction_ratio <= 0.25:
                self.stable_distraction_class = 'safe'

        if strong_pose_distraction and distraction_ratio >= 0.30 and monitoring_confidence >= 45.0:
            self.stable_distraction_class = 'distracted'

        if drowsy_posture_signal and drowsy_probability is not None and drowsy_probability >= 0.78 and distraction_ratio < 0.85 and not strong_pose_distraction:
            self.stable_distraction_class = 'safe'

        if self.stable_distraction_class == 'safe':
            distraction_class = 'safe'
            distraction_confidence = float(np.clip(1.0 - distraction_ratio, 0.45, 0.99))
        else:
            distraction_class = 'distracted'
            base_conf = max(distraction_ratio, float(distraction_confidence or 0.0))
            if strong_pose_distraction:
                base_conf = max(base_conf, 0.72)
            distraction_confidence = float(np.clip(base_conf, 0.55, 0.99))

        # Update monitoring system
        state_dict = self.monitoring_system.update(
            eye_state=eye_state,
            is_yawning=is_yawning,
            head_pose=(pitch, yaw, roll),
            eye_aspect_ratio=eye_aspect_ratio,
            mouth_aspect_ratio=mouth_aspect_ratio,
            head_tilt=head_tilt,
            monitoring_confidence=monitoring_confidence,
            face_center=face_center,
            distraction_class=distraction_class,
            distraction_confidence=distraction_confidence,
        )

        state_dict['distraction_class'] = distraction_class
        state_dict['distraction_confidence'] = distraction_confidence
        state_dict['drowsy_probability'] = drowsy_probability
        state_dict['safe_mode_active'] = self.safe_mode_active

        # Fuse new model outputs into alert reasoning
        # CRITICAL FIX: Distraction/drowsiness alerts are INDEPENDENT from fatigue alerts
        distraction_alert_triggered = False
        drowsiness_alert_triggered = False
        
        if (
            self.enable_distraction_alerts
            and state_dict.get('state') != 'CALIBRATING'
            and distraction_class is not None
            and distraction_confidence is not None
        ):
            distraction_signal = (
                (
                    distraction_class != 'safe'
                    and distraction_confidence >= max(0.50, DISTRACTION_CONFIDENCE_THRESHOLD - 0.10)
                )
                or strong_pose_distraction
            )
            if drowsy_posture_signal and drowsy_probability is not None and drowsy_probability >= 0.78 and not strong_pose_distraction:
                distraction_signal = False
            if distraction_signal:
                self.distraction_positive_frames = min(self.distraction_positive_frames + 1, 30)
            else:
                self.distraction_positive_frames = max(self.distraction_positive_frames - 1, 0)

            if self.distraction_positive_frames >= 2 and distraction_signal:
                state_dict.setdefault('alert_reasons', []).append(
                    f"Distraction detected: {distraction_class} ({distraction_confidence * 100:.0f}%)"
                )
                # Set distraction alert independently
                distraction_alert_triggered = True
                state_dict['should_alert'] = True
                if not state_dict.get('alert_text') or state_dict.get('alert_text') in ('FATIGUE ALERT', 'ATTENTION REQUIRED'):
                    state_dict['alert_text'] = 'DISTRACTION ALERT'
        else:
            self.distraction_positive_frames = max(self.distraction_positive_frames - 1, 0)

        if (
            self.enable_drowsiness_alerts
            and state_dict.get('state') != 'CALIBRATING'
            and drowsy_probability is not None
        ):
            sustained_eye_closure = self.monitoring_system.fatigue_scorer.eye_closure_duration >= max(
                1.2,
                EYE_CLOSURE_DURATION_THRESHOLD,
            )
            drowsy_support_signal = (
                sustained_eye_closure
                or (not eye_state)
                or (eye_aspect_ratio < (EYE_ASPECT_RATIO_THRESHOLD * 0.92))
                or (abs(float(head_tilt)) > 12.0)
                or (abs(float(roll)) > 15.0)
                or (self.yawn_positive_frames >= (YAWN_MIN_POSITIVE_FRAMES + 1))
            )
            model_signal = drowsy_probability >= max(0.45, DROWSINESS_CONFIDENCE_THRESHOLD - 0.08)
            physiology_signal = drowsy_support_signal and drowsy_probability >= max(
                0.34,
                DROWSINESS_CONFIDENCE_THRESHOLD - 0.12,
            )

            if model_signal or physiology_signal:
                self.drowsy_positive_frames = min(self.drowsy_positive_frames + 1, 30)
            else:
                self.drowsy_positive_frames = max(self.drowsy_positive_frames - 1, 0)

            if self.drowsy_positive_frames >= 2:
                state_dict.setdefault('alert_reasons', []).append(
                    f"Drowsiness probability high ({drowsy_probability * 100:.0f}%)"
                )
                # Set drowsiness alert independently
                drowsiness_alert_triggered = True
                state_dict['should_alert'] = True
                if not state_dict.get('emergency_flag', False) and state_dict.get('state') != 'CRITICAL':
                    state_dict['alert_text'] = 'DROWSINESS ALERT'
                self.distraction_positive_frames = max(self.distraction_positive_frames - 2, 0)
        else:
            self.drowsy_positive_frames = max(self.drowsy_positive_frames - 1, 0)

        # Log alerts
        if state_dict['should_alert']:
            alert_type = 'CRITICAL_UNRESPONSIVE' if state_dict.get('emergency_flag', False) else 'FATIGUE_WARNING'
            self.alert_logger.log_alert(
                alert_type=alert_type,
                reasons=state_dict['alert_reasons'],
            )

        risk_index, risk_level, driving_time_seconds, recent_alerts_count = self._compute_risk_index(
            float(state_dict.get('fatigue_score', 0.0))
        )
        state_dict['driver_risk_index'] = float(risk_index)
        state_dict['driver_risk_level'] = risk_level
        state_dict['driving_time_seconds'] = float(driving_time_seconds)
        state_dict['recent_alerts_count'] = int(recent_alerts_count)

        state_dict['recovery_time_seconds'] = self.last_recovery_time_seconds
        state_dict['failsafe_mode'] = bool(self.failsafe_mode)
        state_dict['emergency_email_status'] = str(self.emergency_email_last_status)
        state_dict['emergency_email_message'] = str(self.emergency_email_last_message)
        state_dict['emergency_email_time'] = self.emergency_email_last_time
        state_dict['screenshot_status'] = str(self.screenshot_last_status)
        state_dict['screenshot_message'] = str(self.screenshot_last_message)
        state_dict['screenshot_time'] = self.screenshot_last_time
        state_dict['screenshot_path'] = self.screenshot_last_path

        unreliable, issues = self._detect_failure_modes(state_dict)
        state_dict['system_unreliable'] = unreliable
        state_dict['unreliable_reasons'] = issues

        now_epoch = time.time()
        if self.system_check_status is not None and self.self_check_visible_until is not None and now_epoch <= self.self_check_visible_until:
            state_dict['self_check_status'] = self.system_check_status
        else:
            state_dict['self_check_status'] = None

        self.last_state_dict = state_dict
        self.last_face_bbox = face_bbox

        # Store fatigue trend timeline
        self.fatigue_timeline.append(float(state_dict.get('fatigue_score', 0.0)))
        self.timestamp_timeline.append(datetime.now())

        was_critical_before = self.last_critical_active

        self._update_timeline_events(state_dict)
        self._maybe_capture_critical_screenshot(frame, state_dict, was_critical_before)

        # Record performance metrics
        frame_time = (time.time() - frame_start) * 1000
        self.performance_monitor.record_frame_time(frame_time)
        self.performance_monitor.record_inference_time(frame_time)

        return frame, state_dict

    def _save_fatigue_trend_graph(self):
        """Save fatigue trend graph to results directory."""
        if not self.fatigue_timeline:
            return

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.plot(self.fatigue_timeline, color='tab:red', linewidth=1.5)
        plt.title('Fatigue Trend Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Fatigue Score')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'fatigue_trend.png', dpi=120, bbox_inches='tight')
        plt.close()

    def _build_dashboard(self, frame, state_dict, fps):
        """Build final dashboard frame for display."""
        return self.visualizer.render_dashboard(
            frame=frame,
            state_dict=state_dict,
            face_bbox=self.last_face_bbox,
            fps=fps,
        )

    def _on_mouse_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        action = self.visualizer.get_button_action(x, y)
        if action == 'toggle_theme':
            self.visualizer.toggle_theme()
        elif action == 'emergency_settings':
            self.emergency_settings_open = not self.emergency_settings_open
        elif action == 'exit':
            self.request_exit = True
        elif action == 'exit_report':
            self.request_report_exit = True

    def run_webcam(self):
        """Run real-time monitoring from webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        self._run_startup_self_check(cap)

        print("Starting real-time monitoring (Press 'q' to quit, 't' to toggle theme)...")

        frame_count = 0
        start_time = time.time()
        processed_count = 0
        self.capture_active = True
        self.request_exit = False
        self.request_report_exit = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

        try:
            while True:
                for _ in range(max(CAMERA_FRAME_FLUSH - 1, 0)):
                    cap.grab()
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame for faster processing
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

                # Process frame based on skip setting
                if frame_count % INFERENCE_SKIP_FRAMES == 0:
                    processed_frame, state_dict = self.process_frame(frame)
                    processed_count += 1
                else:
                    processed_frame = frame
                    state_dict = self.last_state_dict
                    self.frame_drop_count += 1

                cpu_pct, ram_mb = self.resource_monitor.update()

                if state_dict is None:
                    state_dict = {}
                state_dict['frames_skipped'] = int(self.frame_drop_count)
                state_dict['cpu_usage_percent'] = float(cpu_pct)
                state_dict['ram_usage_mb'] = float(ram_mb)
                state_dict['failsafe_mode'] = bool(self.failsafe_mode)

                if self.request_report_exit:
                    report_json, report_md, report_payload = self._generate_session_report(frame_count, processed_count)
                    self.timeline_logger.log('report_generated', {
                        'json': str(report_json),
                        'markdown': str(report_md),
                    })
                    self.report_overlay_data = {
                        'generated_at': report_payload.get('generated_at'),
                        'session_duration_seconds': float(report_payload.get('session_duration_seconds', 0.0) or 0.0),
                        'avg_fps': float(report_payload.get('avg_fps', 0.0) or 0.0),
                        'alerts_count': int(report_payload.get('alerts_count', 0) or 0),
                        'frames_processed_total': int(report_payload.get('frames_processed_total', 0) or 0),
                        'frames_skipped': int(report_payload.get('frames_skipped', 0) or 0),
                        'avg_cpu_percent': float(report_payload.get('avg_cpu_percent', 0.0) or 0.0),
                        'avg_ram_mb': float(report_payload.get('avg_ram_mb', 0.0) or 0.0),
                        'json_path': str(report_json),
                        'md_path': str(report_md),
                    }
                    self.request_report_exit = False

                if self.report_overlay_data is not None:
                    state_dict['report_overlay'] = self.report_overlay_data

                if self.emergency_settings_open:
                    fields = self._emergency_settings_fields()
                    missing_required = self.emergency_notifier.get_missing_required_fields()
                    state_dict['emergency_settings_overlay'] = {
                        'fields': [
                            {
                                'key': field_key,
                                'label': field_label,
                                'value': self.emergency_settings.get(field_key, ''),
                                'type': field_type,
                            }
                            for field_key, field_label, field_type in fields
                        ],
                        'selected_index': int(self.emergency_settings_field_index),
                        'smtp_ready': bool(self.emergency_notifier.is_enabled()),
                        'missing_required': missing_required,
                        'settings_file': str(self.emergency_settings_file),
                    }
                else:
                    state_dict['emergency_settings_overlay'] = None

                # Draw FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                dashboard = self._build_dashboard(processed_frame, state_dict, fps)

                # Display frame
                cv2.imshow(self.window_name, dashboard)

                # Handle user input
                key_code = cv2.waitKeyEx(1)
                key = key_code & 0xFF

                # Report overlay behaves as a modal exit confirmation.
                if self.report_overlay_data is not None:
                    if key_code in (27, 13, 10, 32) or key in (ord('q'), ord('e')):
                        self.request_exit = True
                        break
                    if key == ord('t'):
                        self.visualizer.toggle_theme()
                    continue

                if self.emergency_settings_open:
                    if key_code != -1:
                        self._handle_emergency_settings_key(key_code)
                    continue

                if key == ord('q'):
                    self.request_exit = True
                    break
                if key == ord('t'):
                    self.visualizer.toggle_theme()
                if key == ord('s'):
                    self.emergency_settings_open = not self.emergency_settings_open
                if key == ord('e'):
                    self.request_exit = True
                    break
                if key == ord('r'):
                    self.request_report_exit = True
                    continue

                if self.request_exit:
                    break

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.capture_active = False
            cap.release()
            cv2.destroyAllWindows()
            self._save_fatigue_trend_graph()

            if self.request_report_exit:
                report_json, report_md, report_payload = self._generate_session_report(frame_count, processed_count)
                self.timeline_logger.log('report_generated', {
                    'json': str(report_json),
                    'markdown': str(report_md),
                })
                self.report_overlay_data = {
                    'generated_at': report_payload.get('generated_at'),
                    'session_duration_seconds': float(report_payload.get('session_duration_seconds', 0.0) or 0.0),
                    'avg_fps': float(report_payload.get('avg_fps', 0.0) or 0.0),
                    'alerts_count': int(report_payload.get('alerts_count', 0) or 0),
                    'frames_processed_total': int(report_payload.get('frames_processed_total', 0) or 0),
                    'frames_skipped': int(report_payload.get('frames_skipped', 0) or 0),
                    'avg_cpu_percent': float(report_payload.get('avg_cpu_percent', 0.0) or 0.0),
                    'avg_ram_mb': float(report_payload.get('avg_ram_mb', 0.0) or 0.0),
                    'json_path': str(report_json),
                    'md_path': str(report_md),
                }

    def run_video(self, video_path):
        """Run monitoring on a video file."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        print(f"Processing video: {video_path}")

        frame_count = 0
        start_time = time.time()
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

                # Process frame based on skip setting
                if frame_count % INFERENCE_SKIP_FRAMES == 0:
                    processed_frame, state_dict = self.process_frame(frame)
                    processed_count += 1
                else:
                    processed_frame = frame
                    state_dict = self.last_state_dict

                # Draw FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                dashboard = self._build_dashboard(processed_frame, state_dict, fps)

                # Display frame
                cv2.imshow('Driver Fatigue Monitoring', dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._save_fatigue_trend_graph()

            stats = self.performance_monitor.get_stats()
            print("\n=== Performance Statistics ===")
            print(f"Average FPS: {stats['avg_fps']:.2f}")
            print(f"Total Frames Processed: {frame_count}")
            print(f"Inference Frames Processed: {processed_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Driver Fatigue Monitoring System")
    parser.add_argument("--mode", type=str, default="webcam", choices=["webcam", "video"],
                       help="Mode: 'webcam' for live monitoring or 'video' for video file")
    parser.add_argument("--video", type=str, help="Path to video file (required if mode='video')")
    parser.add_argument("--eye_model", type=str, default=str(EYE_MODEL_PATH),
                       help="Path to eye state model")
    parser.add_argument("--yawn_model", type=str, default=str(YAWN_MODEL_PATH),
                       help="Path to yawn detection model")
    parser.add_argument("--distraction_model", type=str, default=str(DISTRACTION_MODEL_PATH),
                       help="Path to distraction model")
    parser.add_argument("--drowsiness_model", type=str, default=str(DROWSINESS_MODEL_PATH),
                       help="Path to drowsiness model")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DriverFatigueMonitoringPipeline(
        eye_model_path=args.eye_model,
        yawn_model_path=args.yawn_model,
        distraction_model_path=args.distraction_model,
        drowsiness_model_path=args.drowsiness_model,
    )

    # Run in selected mode
    if args.mode == "webcam":
        pipeline.run_webcam()
    elif args.mode == "video":
        if not args.video:
            print("Error: --video argument required when mode='video'")
            sys.exit(1)
        pipeline.run_video(args.video)
