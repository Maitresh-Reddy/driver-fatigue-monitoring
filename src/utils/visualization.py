import cv2
import numpy as np
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import time
import ctypes
from ctypes import wintypes
from src.config import (
    VISUALIZATION_THICKNESS, VISUALIZATION_FONT_SCALE,
    SHOW_FPS, SHOW_DETECTIONS, SHOW_FATIGUE_SCORE, SHOW_STATE, SHOW_HEAD_POSE,
    GRAPH_HISTORY_SECONDS, FPS_TARGET, EYE_ASPECT_RATIO_THRESHOLD,
)


class Visualizer:
    """Handles visualization of detection results and system state."""

    COLORS = {
        'alert': (70, 255, 170),
        'distracted': (0, 188, 255),
        'fatigued': (80, 80, 255),
        'critical': (0, 0, 255),
        'text': (235, 242, 255),
        'face': (180, 220, 255),
        'info': (255, 220, 120),
        'panel': (22, 25, 36),
        'panel_alt': (42, 47, 66),
        'grid': (68, 78, 102),
    }

    LIGHT_COLORS = {
        'alert': (20, 130, 60),
        'distracted': (0, 145, 255),
        'fatigued': (0, 90, 210),
        'critical': (0, 0, 255),
        'text': (28, 28, 35),
        'face': (70, 110, 170),
        'info': (120, 85, 30),
        'panel': (248, 248, 252),
        'panel_alt': (224, 226, 236),
        'grid': (184, 188, 206),
    }

    def __init__(self, config):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.theme_mode = 'dark'
        self.button_regions = {}
        history_len = max(int(FPS_TARGET * GRAPH_HISTORY_SECONDS), 30)
        self.fatigue_history = deque(maxlen=history_len)

    def toggle_theme(self):
        self.theme_mode = 'light' if self.theme_mode == 'dark' else 'dark'

    def _palette(self):
        return self.COLORS if self.theme_mode == 'dark' else self.LIGHT_COLORS

    def _state_color(self, state):
        colors = self._palette()
        return {
            'ALERT': colors['alert'],
            'DISTRACTED': colors['distracted'],
            'DROWSINESS': colors['fatigued'],
            'MILD FATIGUE': colors['distracted'],
            'MODERATE FATIGUE': colors['distracted'],
            'SEVERE FATIGUE': colors['fatigued'],
            'CRITICAL': colors['critical'],
            'CALIBRATING': colors['info'],
            'TRACKING_LOST': colors['fatigued'],
        }.get(state, colors['text'])

    def get_button_action(self, x, y):
        for key, rect in self.button_regions.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return key
        return None

    def _draw_button(self, canvas, key, text, rect, primary=False):
        x1, y1, x2, y2 = rect
        fill = (0, 0, 165) if primary else (58, 66, 88)
        border = (0, 0, 230) if primary else (90, 102, 135)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        cv2.putText(canvas, text, (x1 + 10, y1 + 22), self.font, 0.5, (240, 246, 255), 1)
        self.button_regions[key] = rect

    def render_dashboard(self, frame, state_dict, face_bbox=None, fps=0.0):
        colors = self._palette()
        state_dict = state_dict or {}

        src_h, src_w = frame.shape[:2]
        dashboard_w, dashboard_h = 1536, 900
        canvas = np.zeros((dashboard_h, dashboard_w, 3), dtype=np.uint8)
        canvas[:] = (8, 10, 16) if self.theme_mode == 'dark' else (244, 246, 252)
        self.button_regions = {}

        margin = 24
        top_h = 54
        content_top = margin + top_h + 12
        left_w = 940
        right_w = dashboard_w - (margin * 3 + left_w)
        right_x1 = margin * 2 + left_w
        right_x2 = right_x1 + right_w
        live_h = 560

        state_name = str(state_dict.get('state', 'UNKNOWN'))
        alert_text = str(state_dict.get('alert_text', ''))
        fatigue_score = float(state_dict.get('fatigue_score', 0.0) or 0.0)
        confidence = float(state_dict.get('monitoring_confidence', 0.0) or 0.0)
        risk_index = float(state_dict.get('driver_risk_index', 0.0) or 0.0)
        eye_state = state_dict.get('eye_state', None)
        eye_aspect_ratio = state_dict.get('eye_aspect_ratio', None)
        if eye_state is None and eye_aspect_ratio is not None:
            eye_state = float(eye_aspect_ratio) >= (EYE_ASPECT_RATIO_THRESHOLD + 0.02)
        if eye_state is None:
            eye_state = not (state_name == 'CRITICAL' or state_dict.get('emergency_flag', False))
        eye_text = "OPEN" if bool(eye_state) else "CLOSED"
        yawn_text = "DETECTED" if state_dict.get('is_yawning', False) else "NOT DETECTED"
        distraction_class = state_dict.get('distraction_class', None)
        distraction_conf = float(state_dict.get('distraction_confidence', 0.0) or 0.0)
        drowsy_prob = float(state_dict.get('drowsy_probability', 0.0) or 0.0)
        sudden_event = state_dict.get('sudden_event', False)
        head_pose = state_dict.get('head_pose', (0.0, 0.0, 0.0))
        head_pose_delta = state_dict.get('head_pose_delta', head_pose)
        pitch = float(head_pose[0]) if head_pose is not None and head_pose[0] is not None else 0.0
        pitch_delta = float(head_pose_delta[0]) if head_pose_delta is not None and head_pose_delta[0] is not None else pitch
        self_check_status = state_dict.get('self_check_status')
        calibration_progress = float(state_dict.get('calibration_progress', 0.0) or 0.0)
        calibration_remaining = float(state_dict.get('calibration_remaining_seconds', 0.0) or 0.0)

        # Context-aware recommendation
        recommendation = "Drive safely"
        if state_dict.get('sudden_event', False) or state_dict.get('emergency_flag', False):
            recommendation = "EMERGENCY: Pull over immediately"
        elif state_name == 'CRITICAL':
            recommendation = "Stop vehicle and rest immediately"
        elif state_dict.get('should_alert', False):
            recommendation = "Take a short break"

        effective_state_name = state_name
        if state_name in ('ALERT', 'MILD FATIGUE', 'MODERATE FATIGUE'):
            distraction_active = (
                alert_text == 'DISTRACTION ALERT'
                or (
                    distraction_class not in (None, 'safe')
                    and distraction_conf >= 0.55
                )
            )
            drowsiness_active = alert_text == 'DROWSINESS ALERT' or drowsy_prob >= 0.55
            if distraction_active:
                effective_state_name = 'DISTRACTED'
            elif drowsiness_active and state_name == 'ALERT':
                effective_state_name = 'DROWSINESS'

        state_color = self._state_color(effective_state_name)

        emergency_email_status = str(state_dict.get('emergency_email_status', 'NOT SENT'))
        emergency_email_message = str(state_dict.get('emergency_email_message', ''))
        emergency_email_time = state_dict.get('emergency_email_time')
        screenshot_status = str(state_dict.get('screenshot_status', 'NONE'))
        screenshot_message = str(state_dict.get('screenshot_message', ''))
        screenshot_time = state_dict.get('screenshot_time')
        screenshot_path = state_dict.get('screenshot_path')

        self.fatigue_history.append(np.clip(fatigue_score / 100.0, 0, 1))

        # Top bar
        cv2.rectangle(canvas, (margin, margin), (dashboard_w - margin, margin + top_h), (12, 16, 28), -1)
        cv2.rectangle(canvas, (margin, margin), (dashboard_w - margin, margin + top_h), (36, 44, 62), 1)
        cv2.circle(canvas, (margin + 16, margin + 18), 8, (0, 70, 220), -1)
        cv2.putText(canvas, "Driver Fatigue Monitoring System", (margin + 36, margin + 25), self.font, 0.72, (230, 236, 245), 2)

        pill_color = (20, 120, 70) if effective_state_name == 'ALERT' else ((0, 120, 180) if 'CALIBRAT' in effective_state_name else (0, 75, 190) if 'CRITICAL' not in effective_state_name else (0, 0, 200))
        cv2.rectangle(canvas, (margin + 430, margin + 10), (margin + 570, margin + 38), pill_color, -1)
        cv2.putText(canvas, effective_state_name[:14], (margin + 440, margin + 30), self.font, 0.52, (235, 245, 255), 1)

        elapsed_txt = "00:00:00"
        if getattr(self.config, 'session_start_time', None) is not None:
            elapsed = datetime.now() - self.config.session_start_time
            total_secs = max(0, int(elapsed.total_seconds()))
            hh = total_secs // 3600
            mm = (total_secs % 3600) // 60
            ss = total_secs % 60
            elapsed_txt = f"{hh:02d}:{mm:02d}:{ss:02d}"
        btn_settings_rect = (dashboard_w - 642, margin + 10, dashboard_w - 452, margin + 42)
        btn_theme_rect = (dashboard_w - 446, margin + 10, dashboard_w - 292, margin + 42)
        btn_exit_rect = (dashboard_w - 286, margin + 10, dashboard_w - 170, margin + 42)
        btn_report_rect = (dashboard_w - 164, margin + 10, dashboard_w - 28, margin + 42)

        timer_font_scale = 0.58
        timer_size, _ = cv2.getTextSize(elapsed_txt, self.font, timer_font_scale, 1)
        timer_x = btn_settings_rect[0] - timer_size[0] - 20
        cv2.putText(canvas, elapsed_txt, (timer_x, margin + 30), self.font, timer_font_scale, (210, 220, 235), 1)

        email_status_color = (130, 190, 240)
        if emergency_email_status == 'SENT':
            email_status_color = (90, 235, 170)
        elif emergency_email_status in ('FAILED', 'NOT CONFIGURED'):
            email_status_color = (0, 90, 255)
        elif emergency_email_status == 'COOLDOWN':
            email_status_color = (0, 180, 235)
        cv2.putText(canvas, f"Email: {emergency_email_status}", (margin + 592, margin + 30), self.font, 0.52, email_status_color, 1)

        # Live feed card
        live_x1, live_y1 = margin, content_top
        live_x2, live_y2 = live_x1 + left_w, live_y1 + live_h
        cv2.rectangle(canvas, (live_x1, live_y1), (live_x2, live_y2), (12, 16, 28), -1)
        cv2.rectangle(canvas, (live_x1, live_y1), (live_x2, live_y2), (32, 40, 58), 1)
        cv2.putText(canvas, "LIVE FEED", (live_x1 + 18, live_y1 + 28), self.font, 0.62, (215, 225, 240), 2)
        cv2.putText(canvas, f"FPS: {fps:.0f}", (live_x2 - 140, live_y1 + 28), self.font, 0.58, (160, 220, 255), 1)

        feed_x1, feed_y1 = live_x1 + 18, live_y1 + 44
        feed_x2, feed_y2 = live_x2 - 18, live_y2 - 92
        feed_w, feed_h = max(1, feed_x2 - feed_x1), max(1, feed_y2 - feed_y1)
        feed_img = cv2.resize(frame, (feed_w, feed_h))
        canvas[feed_y1:feed_y2, feed_x1:feed_x2] = feed_img
        cv2.rectangle(canvas, (feed_x1, feed_y1), (feed_x2, feed_y2), (50, 60, 84), 1)

        if face_bbox is not None:
            bx, by, bw, bh = face_bbox
            sx = feed_w / max(1, src_w)
            sy = feed_h / max(1, src_h)
            rx1 = feed_x1 + int(bx * sx)
            ry1 = feed_y1 + int(by * sy)
            rx2 = feed_x1 + int((bx + bw) * sx)
            ry2 = feed_y1 + int((by + bh) * sy)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 130, 40), 2)

        if state_name == 'CALIBRATING':
            overlay_y1 = feed_y1 + 12
            overlay_y2 = feed_y1 + 54
            cv2.rectangle(canvas, (feed_x1 + 12, overlay_y1), (feed_x2 - 12, overlay_y2), (0, 95, 175), -1)
            cv2.putText(
                canvas,
                f"CALIBRATING: {int(round(calibration_progress * 100))}%  ({max(0, int(round(calibration_remaining)))}s left)",
                (feed_x1 + 26, overlay_y1 + 28),
                self.font,
                0.62,
                (240, 248, 255),
                2,
            )

        ribbon_color = (20, 120, 70)
        ribbon_text = "System monitoring normally"
        if state_name == 'CRITICAL' or state_dict.get('emergency_flag', False):
            ribbon_color = (0, 0, 190)
            ribbon_text = "CRITICAL: Emergency protocol active"
        elif state_dict.get('should_alert', False):
            ribbon_color = (0, 70, 200)
            ribbon_text = str(state_dict.get('alert_text', 'Warning active'))
        cv2.rectangle(canvas, (live_x1 + 30, live_y2 - 74), (live_x2 - 30, live_y2 - 22), ribbon_color, -1)
        cv2.putText(canvas, f"ALERT: {ribbon_text[:56]}", (live_x1 + 52, live_y2 - 40), self.font, 0.76, (245, 250, 255), 2)

        # Right cards
        card_gap = 14
        card_y = content_top

        def draw_right_card(height, title):
            nonlocal card_y
            x1, y1 = right_x1, card_y
            x2, y2 = right_x2, card_y + height
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (12, 16, 28), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (32, 40, 58), 1)
            cv2.putText(canvas, title, (x1 + 18, y1 + 30), self.font, 0.62, (215, 225, 240), 2)
            card_y = y2 + card_gap
            return x1, y1, x2, y2

        if self_check_status is not None:
            x1, y1, x2, y2 = draw_right_card(52, "SYSTEM CHECK")
            check_status_text = str(self_check_status.get('status', 'READY')).upper()
            check_details = (
                f"Cam {self_check_status.get('camera', '-')}, "
                f"Models {self_check_status.get('models', '-')}, "
                f"Light {self_check_status.get('lighting', '-')}"
            )
            cv2.putText(canvas, check_status_text, (x2 - 132, y1 + 31), self.font, 0.55, (90, 235, 170), 1)
            cv2.putText(canvas, check_details[:36], (x1 + 20, y1 + 48), self.font, 0.42, (180, 198, 220), 1)

        x1, y1, x2, y2 = draw_right_card(196, "DRIVER STATUS")
        if state_name == 'CRITICAL' or state_dict.get('emergency_flag', False):
            risk_tag = "HIGH RISK"
        else:
            risk_tag = "HIGH RISK" if risk_index >= 70 else ("MEDIUM RISK" if risk_index >= 40 else "LOW RISK")
        cv2.rectangle(canvas, (x2 - 122, y1 + 12), (x2 - 18, y1 + 38), (0, 0, 140) if "HIGH" in risk_tag else (0, 110, 170) if "MEDIUM" in risk_tag else (20, 120, 70), -1)
        cv2.putText(canvas, risk_tag, (x2 - 114, y1 + 30), self.font, 0.44, (240, 248, 255), 1)
        cv2.putText(canvas, effective_state_name, (x1 + 28, y1 + 88), self.font, 1.0, state_color, 2)
        cv2.putText(canvas, str(state_dict.get('state_reason', 'Monitoring active'))[:34], (x1 + 28, y1 + 118), self.font, 0.57, (200, 212, 232), 1)
        cv2.putText(canvas, "Fatigue Score", (x1 + 18, y1 + 146), self.font, 0.54, (190, 205, 225), 1)
        cv2.putText(canvas, f"{int(round(fatigue_score))} / 100", (x2 - 132, y1 + 146), self.font, 0.6, (230, 236, 245), 1)
        bx1, by1, bx2, by2 = x1 + 18, y1 + 156, x2 - 18, y1 + 168
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (36, 42, 58), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx1 + int((bx2 - bx1) * np.clip(fatigue_score / 100.0, 0, 1)), by2), state_color, -1)
        cv2.putText(canvas, f"Confidence: {confidence:.0f}%", (x1 + 18, y1 + 188), self.font, 0.5, (190, 205, 225), 1)

        x1, y1, x2, y2 = draw_right_card(168, "ALERT LEVEL")
        level = "NORMAL"
        if fatigue_score >= 76:
            level = "CRITICAL"
        elif fatigue_score >= 51:
            level = "FATIGUED"
        elif fatigue_score >= 26:
            level = "WARNING"
        cv2.putText(canvas, level, (x1 + 26, y1 + 88), self.font, 0.94, state_color, 2)
        cv2.putText(canvas, "NORMAL > WARNING > FATIGUE > CRITICAL", (x1 + 18, y1 + 128), self.font, 0.43, (145, 160, 184), 1)

        x1, y1, x2, y2 = draw_right_card(186, "SAFETY FEATURES")
        safety_lines = [
            f"Sudden Event: {'DETECTED' if state_dict.get('sudden_event', False) else 'NOT DETECTED'}",
            f"Emergency Flag: {'ACTIVE' if state_dict.get('emergency_flag', False) else 'READY'}",
            f"Re-attention: {'ACTIVE' if state_dict.get('should_alert', False) else 'STANDBY'}",
            f"Failsafe Mode: {'ON' if state_dict.get('failsafe_mode', False) else 'OFF'}",
        ]
        yy = y1 + 58
        for line in safety_lines:
            cv2.circle(canvas, (x1 + 24, yy - 5), 6, (35, 170, 115), -1)
            cv2.putText(canvas, line[:44], (x1 + 40, yy), self.font, 0.5, (200, 212, 232), 1)
            yy += 32

        x1, y1, x2, y2 = draw_right_card(max(120, dashboard_h - card_y - margin), "SESSION INFO")
        session_lines = [
            ("Session Time", elapsed_txt),
            ("Frames Analyzed", str(int(getattr(self.config, 'frame_index', 0)))),
            ("Avg. FPS", f"{fps:.0f}"),
            ("Status", effective_state_name),
            ("Recommendation", recommendation),
            ("Email Status", emergency_email_status),
            ("Screenshot", screenshot_status),
        ]

        detail_lines = []
        if emergency_email_message:
            email_time_text = str(emergency_email_time)[11:19] if emergency_email_time else '--:--:--'
            email_detail = f"Last Email [{email_time_text}]: {emergency_email_status}"
            email_detail_color = (160, 178, 204)
            if emergency_email_status == 'SENT':
                email_detail_color = (90, 235, 170)
            elif emergency_email_status in ('FAILED', 'NOT CONFIGURED'):
                email_detail_color = (0, 90, 255)
            elif emergency_email_status in ('COOLDOWN', 'SKIPPED'):
                email_detail_color = (0, 180, 235)
            detail_lines.append((email_detail, email_detail_color))

        if screenshot_status != 'NONE' or screenshot_message:
            shot_time_text = str(screenshot_time)[11:19] if screenshot_time else '--:--:--'
            shot_name = Path(str(screenshot_path)).name if screenshot_path else screenshot_message
            screenshot_detail = f"Screenshot [{shot_time_text}]: {screenshot_status} {shot_name}"
            screenshot_detail_color = (160, 178, 204)
            if screenshot_status == 'CAPTURED':
                screenshot_detail_color = (90, 235, 170)
            elif screenshot_status == 'FAILED':
                screenshot_detail_color = (0, 90, 255)
            elif screenshot_status in ('PENDING', 'NONE'):
                screenshot_detail_color = (0, 180, 235)
            detail_lines.append((screenshot_detail, screenshot_detail_color))

        top_y = y1 + 58
        bottom_y = y2 - 14
        detail_line_step = 14
        details_reserved = (len(detail_lines) * detail_line_step + 6) if detail_lines else 0
        rows_available = max(40, (bottom_y - top_y) - details_reserved)
        row_step = max(16, min(30, int(rows_available / max(len(session_lines), 1))))

        if row_step <= 18:
            label_font = 0.40
            value_font = 0.42
        elif row_step <= 22:
            label_font = 0.44
            value_font = 0.46
        elif row_step <= 26:
            label_font = 0.48
            value_font = 0.50
        else:
            label_font = 0.52
            value_font = 0.54

        yy = top_y
        for key, value in session_lines:
            cv2.putText(canvas, key, (x1 + 22, yy), self.font, label_font, (175, 192, 214), 1)
            cv2.putText(canvas, str(value)[:22], (x2 - 198, yy), self.font, value_font, (235, 242, 252), 1)
            yy += row_step

        detail_font = max(0.34, label_font - 0.08)
        details_y = min(bottom_y - (len(detail_lines) - 1) * detail_line_step, yy + 2)
        for detail_text, detail_color in detail_lines:
            cv2.putText(canvas, detail_text[:58], (x1 + 22, details_y), self.font, detail_font, detail_color, 1)
            details_y += detail_line_step

        # Bottom cards
        bottom_y1 = content_top + live_h + 12
        bottom_y2 = dashboard_h - margin
        card_w = (left_w - 12) // 2

        ax1, ay1, ax2, ay2 = margin, bottom_y1, margin + card_w, bottom_y2
        cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), (12, 16, 28), -1)
        cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), (32, 40, 58), 1)
        cv2.putText(canvas, "FATIGUE ANALYTICS", (ax1 + 16, ay1 + 30), self.font, 0.62, (215, 225, 240), 2)

        gx1, gy1 = ax1 + 18, ay1 + 46
        gx2, gy2 = ax2 - 18, ay2 - 26
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (10, 12, 20), -1)
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (30, 38, 56), 1)
        for val, col in [(25, (0, 180, 210)), (50, (0, 160, 230)), (75, (0, 140, 255))]:
            y = gy2 - int((val / 100.0) * (gy2 - gy1))
            cv2.line(canvas, (gx1, y), (gx2, y), col, 1)
            cv2.putText(canvas, str(val), (gx1 - 22, y + 4), self.font, 0.4, (170, 185, 210), 1)
        if len(self.fatigue_history) > 1:
            vals = list(self.fatigue_history)[-120:]
            pts = []
            for i, v in enumerate(vals):
                px = gx1 + int(i * (gx2 - gx1) / max(len(vals) - 1, 1))
                py = gy2 - int(np.clip(v, 0, 1) * (gy2 - gy1))
                pts.append((px, py))
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 80, 230), 3)

        wx1, wy1, wx2, wy2 = ax2 + 12, bottom_y1, margin + left_w, bottom_y2
        cv2.rectangle(canvas, (wx1, wy1), (wx2, wy2), (12, 16, 28), -1)
        cv2.rectangle(canvas, (wx1, wy1), (wx2, wy2), (32, 40, 58), 1)
        cv2.putText(canvas, "WHY THIS ALERT?", (wx1 + 16, wy1 + 30), self.font, 0.62, (215, 225, 240), 2)

        reason_rows = [
            ("Eye State", eye_text, (0, 0, 255) if eye_state is False else (160, 220, 255)),
            ("Yawning", yawn_text, (0, 0, 255) if state_dict.get('is_yawning', False) else (160, 220, 255)),
            ("Head Pose", f"{pitch_delta:+.1f}°", (0, 0, 255) if abs(pitch_delta) > 12 else (160, 220, 255)),
            ("Distraction", "NONE" if distraction_class in (None, 'safe') else str(distraction_class).upper(), (0, 0, 255) if distraction_class not in (None, 'safe') else (160, 220, 255)),
            ("Sudden Event", "DETECTED" if sudden_event else "NOT DETECTED", (0, 0, 255) if sudden_event else (160, 220, 255)),
        ]
        yy = wy1 + 62
        row_h = 28
        for key, val, col in reason_rows:
            cv2.putText(canvas, key, (wx1 + 22, yy), self.font, 0.56, (190, 205, 225), 1)
            cv2.putText(canvas, str(val)[:14], (wx2 - 190, yy), self.font, 0.58, col, 2 if col == (0, 0, 255) else 1)
            yy += row_h

        report_overlay = state_dict.get('report_overlay')
        if report_overlay is not None:
            ox1, oy1 = margin + 140, margin + 92
            ox2, oy2 = dashboard_w - margin - 140, dashboard_h - margin - 120
            cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), (8, 12, 20), -1)
            cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), (80, 96, 124), 2)
            cv2.putText(canvas, "SESSION REPORT", (ox1 + 24, oy1 + 38), self.font, 0.86, (230, 238, 248), 2)
            overlay_lines = [
                f"Generated: {str(report_overlay.get('generated_at', '-'))[:19]}",
                f"Duration: {float(report_overlay.get('session_duration_seconds', 0.0)):.1f} sec",
                f"Average FPS: {float(report_overlay.get('avg_fps', 0.0)):.2f}",
                f"Alerts: {int(report_overlay.get('alerts_count', 0))}",
                f"Frames: {int(report_overlay.get('frames_processed_total', 0))} (skipped {int(report_overlay.get('frames_skipped', 0))})",
                f"CPU: {float(report_overlay.get('avg_cpu_percent', 0.0)):.1f}%",
                f"RAM: {float(report_overlay.get('avg_ram_mb', 0.0)):.0f} MB",
            ]
            ly = oy1 + 84
            for line in overlay_lines:
                cv2.putText(canvas, line, (ox1 + 28, ly), self.font, 0.62, (206, 220, 240), 1)
                ly += 34
            cv2.putText(canvas, "Report saved to results/. Press ESC, ENTER, SPACE, Q or E to close.", (ox1 + 28, oy2 - 24), self.font, 0.54, (180, 196, 220), 1)

        emergency_overlay = state_dict.get('emergency_settings_overlay')
        if emergency_overlay is not None:
            ex1, ey1 = margin + 120, margin + 86
            ex2, ey2 = dashboard_w - margin - 120, dashboard_h - margin - 84
            cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (8, 12, 20), -1)
            cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (80, 96, 124), 2)
            cv2.putText(canvas, "EMERGENCY SETTINGS", (ex1 + 24, ey1 + 38), self.font, 0.84, (230, 238, 248), 2)
            smtp_label = "SMTP READY" if emergency_overlay.get('smtp_ready', False) else "SMTP NOT CONFIGURED"
            smtp_color = (90, 235, 170) if emergency_overlay.get('smtp_ready', False) else (0, 80, 255)
            cv2.putText(canvas, smtp_label, (ex2 - 258, ey1 + 38), self.font, 0.58, smtp_color, 1)

            missing_required = emergency_overlay.get('missing_required', [])
            if missing_required:
                missing_text = "Missing: " + ", ".join([str(item) for item in missing_required])
                cv2.putText(canvas, missing_text[:118], (ex1 + 24, ey1 + 64), self.font, 0.46, (0, 140, 255), 1)

            field_rows = emergency_overlay.get('fields', [])
            selected_index = int(emergency_overlay.get('selected_index', 0) or 0)
            row_y_start = ey1 + 92
            row_h = 40
            row_step = 44
            footer_reserved = 78
            visible_rows = max(4, int((ey2 - row_y_start - footer_reserved) / row_step))
            window_start = 0
            if selected_index >= visible_rows:
                window_start = selected_index - visible_rows + 1
            window_end = min(len(field_rows), window_start + visible_rows)

            row_y = row_y_start
            for idx in range(window_start, window_end):
                field = field_rows[idx]
                row_h = 40
                row_x1 = ex1 + 24
                row_x2 = ex2 - 24
                row_y1 = row_y - 24
                row_y2 = row_y1 + row_h
                is_selected = idx == selected_index
                cv2.rectangle(canvas, (row_x1, row_y1), (row_x2, row_y2), (28, 38, 58) if is_selected else (18, 24, 38), -1)
                cv2.rectangle(canvas, (row_x1, row_y1), (row_x2, row_y2), (110, 140, 190) if is_selected else (42, 56, 82), 1)

                label = str(field.get('label', 'Field'))
                raw_value = field.get('value', '')
                if field.get('type') == 'bool':
                    value = 'ON' if bool(raw_value) else 'OFF'
                elif field.get('type') == 'password':
                    value = '*' * len(str(raw_value or ''))
                else:
                    value = str(raw_value)
                if is_selected:
                    value = f"> {value}_"

                cv2.putText(canvas, label[:34], (row_x1 + 10, row_y), self.font, 0.56, (186, 202, 228), 1)
                cv2.putText(canvas, value[:56], (row_x1 + 340, row_y), self.font, 0.56, (235, 244, 255), 1)
                row_y += row_step

            if len(field_rows) > visible_rows:
                progress = f"Fields {window_start + 1}-{window_end} of {len(field_rows)}"
                cv2.putText(canvas, progress, (ex2 - 236, ey2 - 50), self.font, 0.46, (160, 178, 204), 1)

            footer = "TAB/Arrows: Move  SPACE: Toggle Bool  Type to Edit  BACKSPACE: Delete  ENTER: Save  ESC: Close"
            cv2.putText(canvas, footer[:120], (ex1 + 24, ey2 - 48), self.font, 0.5, (186, 202, 228), 1)
            settings_file = str(emergency_overlay.get('settings_file', ''))
            cv2.putText(canvas, f"Saved in: {settings_file[-72:]}", (ex1 + 24, ey2 - 22), self.font, 0.46, (160, 178, 204), 1)

        theme_label = "Theme: Dark" if self.theme_mode == 'dark' else "Theme: Light"
        self._draw_button(canvas, 'emergency_settings', 'Emergency Settings', btn_settings_rect, primary=False)
        self._draw_button(canvas, 'toggle_theme', theme_label, btn_theme_rect, primary=False)
        self._draw_button(canvas, 'exit', 'Exit', btn_exit_rect, primary=False)
        self._draw_button(canvas, 'exit_report', 'Exit + Report', btn_report_rect, primary=True)

        return canvas

    def draw_fps(self, frame, fps):
        if not SHOW_FPS:
            return
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), self.font, VISUALIZATION_FONT_SCALE, self.COLORS['text'], VISUALIZATION_THICKNESS)

    def draw_face_bbox(self, frame, bbox):
        if bbox is None or not SHOW_DETECTIONS:
            return
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.COLORS['face'], 2)

    def draw_eye_state(self, frame, eye_state, eye_aspect_ratio, position=(10, 80)):
        if not SHOW_DETECTIONS:
            return
        state_text = f"Eyes: {'OPEN' if eye_state else 'CLOSED'} ({eye_aspect_ratio:.2f})"
        color = self.COLORS['alert'] if eye_state else self.COLORS['fatigued']
        cv2.putText(frame, state_text, position, self.font, VISUALIZATION_FONT_SCALE, color, VISUALIZATION_THICKNESS)

    def draw_yawn_status(self, frame, is_yawning, position=(10, 110)):
        if not SHOW_DETECTIONS:
            return
        yawn_text = "YAWNING" if is_yawning else "Not yawning"
        color = self.COLORS['fatigued'] if is_yawning else self.COLORS['alert']
        cv2.putText(frame, yawn_text, position, self.font, VISUALIZATION_FONT_SCALE, color, VISUALIZATION_THICKNESS)

    def draw_head_pose(self, frame, pitch, yaw, roll, position=(10, 140)):
        if not SHOW_HEAD_POSE:
            return
        pose_text = f"Pitch: {pitch:.1f}° Yaw: {yaw:.1f}° Roll: {roll:.1f}°"
        cv2.putText(frame, pose_text, position, self.font, VISUALIZATION_FONT_SCALE, self.COLORS['info'], VISUALIZATION_THICKNESS)

    def draw_fatigue_score(self, frame, fatigue_score, position=(10, 170)):
        if not SHOW_FATIGUE_SCORE:
            return
        score_text = f"Fatigue Score: {fatigue_score:.1f}/100"
        cv2.putText(frame, score_text, position, self.font, VISUALIZATION_FONT_SCALE, self.COLORS['text'], VISUALIZATION_THICKNESS)

    def draw_driver_state(self, frame, state, position=(10, 230)):
        if not SHOW_STATE:
            return
        cv2.putText(frame, f"State: {state}", position, self.font, 0.8, self._state_color(state), VISUALIZATION_THICKNESS)

    def draw_alert_message(self, frame, alert_text, is_alert=True):
        if not is_alert:
            return
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "ALERT!", (20, 40), self.font, 1.5, self.COLORS['fatigued'], 3)
        cv2.putText(frame, alert_text, (20, 70), self.font, 0.7, self.COLORS['text'], 2)


class AlertLogger:
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.alerts = []

    def log_alert(self, alert_type, reasons, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        alert_record = {
            'timestamp': timestamp.isoformat(),
            'type': alert_type,
            'reasons': reasons,
        }
        self.alerts.append(alert_record)
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert_record) + '\n')

    def log_event(self, event_type, details):
        event_record = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details,
        }
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_record) + '\n')

    def get_recent_alerts(self, minutes=5):
        threshold = datetime.now() - timedelta(minutes=minutes)
        recent = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time > threshold:
                recent.append(alert)
        return recent


class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.inference_times = []
        self.preprocessing_times = []

    def record_frame_time(self, time_ms):
        self.frame_times.append(time_ms)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def record_inference_time(self, time_ms):
        self.inference_times.append(time_ms)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)

    def record_preprocessing_time(self, time_ms):
        self.preprocessing_times.append(time_ms)
        if len(self.preprocessing_times) > self.window_size:
            self.preprocessing_times.pop(0)

    def get_avg_fps(self):
        if not self.frame_times:
            return 0
        avg_time = np.mean(self.frame_times) / 1000
        return 1 / avg_time if avg_time > 0 else 0

    def get_avg_inference_time(self):
        return np.mean(self.inference_times) if self.inference_times else 0

    def get_stats(self):
        return {
            'avg_fps': self.get_avg_fps(),
            'avg_frame_time_ms': np.mean(self.frame_times) if self.frame_times else 0,
            'avg_inference_time_ms': self.get_avg_inference_time(),
            'avg_preprocessing_time_ms': np.mean(self.preprocessing_times) if self.preprocessing_times else 0,
        }


class EventTimelineLogger:
    def __init__(self, timeline_file_path):
        self.timeline_file_path = Path(timeline_file_path)
        self.timeline_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = deque(maxlen=5000)

    def log(self, event_type, details=None):
        if details is None:
            details = {}
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': str(event_type),
            'details': details,
        }
        self.events.append(event)
        with open(self.timeline_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
        return event

    def get_recent_alert_count(self, minutes=5):
        threshold = datetime.now() - timedelta(minutes=minutes)
        count = 0
        for event in self.events:
            if event.get('event_type') not in ('alert_start', 'critical_start'):
                continue
            try:
                ts = datetime.fromisoformat(event['timestamp'])
            except Exception:
                continue
            if ts >= threshold:
                count += 1
        return count


class ResourceUsageMonitor:
    def __init__(self):
        self.last_wall_time = time.time()
        self.last_cpu_time = time.process_time()
        self.cpu_percent = 0.0
        self.ram_mb = 0.0
        self.cpu_history = deque(maxlen=120)
        self.ram_history_mb = deque(maxlen=120)

    def _get_ram_mb(self):
        try:
            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ('cb', ctypes.c_ulong),
                    ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t),
                ]

            process = ctypes.windll.kernel32.GetCurrentProcess()
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            psapi = ctypes.WinDLL('Psapi.dll')
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            ok = get_process_memory_info(process, ctypes.byref(counters), counters.cb)
            if not ok:
                return 0.0
            return float(counters.WorkingSetSize) / (1024.0 ** 2)
        except Exception:
            return 0.0

    def update(self):
        now = time.time()
        cpu_now = time.process_time()
        dt = max(now - self.last_wall_time, 1e-3)
        dcpu = max(cpu_now - self.last_cpu_time, 0.0)
        cores = max(os.cpu_count() or 1, 1)
        self.cpu_percent = float(np.clip((dcpu / dt) * 100.0 / cores, 0.0, 100.0))
        self.ram_mb = self._get_ram_mb()
        self.last_wall_time = now
        self.last_cpu_time = cpu_now
        self.cpu_history.append(self.cpu_percent)
        self.ram_history_mb.append(self.ram_mb)
        return self.cpu_percent, self.ram_mb
