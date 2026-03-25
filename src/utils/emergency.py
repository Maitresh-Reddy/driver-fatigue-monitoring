import json
import mimetypes
import os
import smtplib
import ssl
import time
import asyncio
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from urllib.request import Request, urlopen


class EmergencyAlertNotifier:
    def __init__(self, config, timeline_logger=None):
        self.config = dict(config or {})
        self.timeline_logger = timeline_logger
        self.last_sent_time = None

    def is_enabled(self):
        return bool(self.config.get('enabled', False) and len(self.get_missing_required_fields()) == 0)

    def get_missing_required_fields(self):
        required_pairs = [
            ('smtp_server', 'SMTP Server'),
            ('smtp_username', 'SMTP Username'),
            ('smtp_password', 'SMTP Password'),
            ('from_email', 'From Email'),
            ('to_email', 'Emergency Contact Email'),
        ]
        missing = []
        for key, label in required_pairs:
            value = self.config.get(key)
            if value is None or str(value).strip() == '':
                missing.append(label)
        return missing

    def send_critical_alert(self, state_dict, screenshot_path):
        if not self.is_enabled():
            return False, 'Emergency email disabled or missing SMTP configuration'

        now_ts = time.time()
        cooldown_seconds = float(self.config.get('cooldown_seconds', 120) or 120)
        if self.last_sent_time is not None and (now_ts - self.last_sent_time) < cooldown_seconds:
            return False, 'Emergency email cooldown active'

        image_path = Path(str(screenshot_path))
        if not image_path.exists():
            return False, 'Critical screenshot not found'

        location = self._resolve_location()
        subject = self._build_subject(state_dict)
        body = self._build_body(state_dict, location, image_path)

        try:
            self._send_email(subject, body, image_path)
            self.last_sent_time = now_ts
            if self.timeline_logger is not None:
                self.timeline_logger.log('emergency_email_sent', {
                    'to': self.config.get('to_email'),
                    'screenshot': str(image_path),
                    'latitude': location.get('latitude'),
                    'longitude': location.get('longitude'),
                    'location_source': location.get('source'),
                })
            return True, 'Emergency email sent'
        except Exception as ex:
            if self.timeline_logger is not None:
                self.timeline_logger.log('emergency_email_failed', {
                    'error': str(ex)[:220],
                    'screenshot': str(image_path),
                })
            return False, f'Emergency email failed: {ex}'

    def _build_subject(self, state_dict):
        prefix = str(self.config.get('subject_prefix', '[Driver Monitor Emergency]')).strip()
        state = str((state_dict or {}).get('state', 'CRITICAL'))
        driver_id = str(self.config.get('driver_id', 'UNKNOWN_DRIVER'))
        vehicle_id = str(self.config.get('vehicle_id', 'UNKNOWN_VEHICLE'))
        return f"{prefix} {state} | Driver {driver_id} | Vehicle {vehicle_id}"

    def _build_body(self, state_dict, location, screenshot_path):
        state_dict = state_dict or {}
        now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        state = str(state_dict.get('state', 'CRITICAL'))
        emergency_flag = bool(state_dict.get('emergency_flag', False))
        alert_text = str(state_dict.get('alert_text', 'CRITICAL: DRIVER INCAPACITATED'))
        fatigue_score = float(state_dict.get('fatigue_score', 0.0) or 0.0)
        risk_index = float(state_dict.get('driver_risk_index', 0.0) or 0.0)
        risk_level = str(state_dict.get('driver_risk_level', 'HIGH RISK'))
        reasons = state_dict.get('alert_reasons', [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        lat = location.get('latitude')
        lon = location.get('longitude')
        loc_text = location.get('location_text', 'Unknown')
        source = location.get('source', 'unknown')

        lines = [
            'Emergency Alert Triggered by Driver Fatigue Monitoring System',
            '',
            f'Timestamp: {now_utc}',
            f'Driver ID: {self.config.get("driver_id", "UNKNOWN_DRIVER")}',
            f'Vehicle ID: {self.config.get("vehicle_id", "UNKNOWN_VEHICLE")}',
            f'Driver Condition: {alert_text}',
            f'State: {state}',
            f'Emergency Flag: {emergency_flag}',
            f'Fatigue Score: {fatigue_score:.1f}/100',
            f'Risk Index: {risk_index:.1f}/100 ({risk_level})',
            '',
            'Location Details:',
            f'Location: {loc_text}',
            f'Latitude: {lat if lat is not None else "N/A"}',
            f'Longitude: {lon if lon is not None else "N/A"}',
            f'Coordinate Source: {source}',
            '',
            f'Critical Screenshot: {screenshot_path.name}',
            '',
            'Alert Reasons:',
        ]

        if reasons:
            lines.extend([f'- {str(reason)}' for reason in reasons])
        else:
            lines.append('- Critical condition detected by system logic')

        lines.extend([
            '',
            'Map Link:',
            self._build_map_link(lat, lon),
        ])

        return '\n'.join(lines)

    @staticmethod
    def _build_map_link(latitude, longitude):
        if latitude is None or longitude is None:
            return 'N/A'
        return f'https://maps.google.com/?q={latitude},{longitude}'

    def _resolve_location(self):
        manual_lat = self._to_float(self.config.get('manual_latitude'))
        manual_lon = self._to_float(self.config.get('manual_longitude'))
        manual_text = str(self.config.get('manual_location_text', '')).strip()

        if manual_lat is not None and manual_lon is not None:
            return {
                'latitude': manual_lat,
                'longitude': manual_lon,
                'location_text': manual_text or 'Configured GPS location',
                'source': 'manual_exact',
            }

        if bool(self.config.get('allow_device_geolocation', True)):
            device_location = self._resolve_via_device_location(manual_text)
            if device_location is not None:
                return device_location

        if not bool(self.config.get('allow_ip_geolocation', True)):
            return {
                'latitude': None,
                'longitude': None,
                'location_text': manual_text or 'Location unavailable',
                'source': 'disabled',
            }

        return self._resolve_via_ip_geolocation(manual_text)

    def _resolve_via_device_location(self, fallback_text):
        if os.name != 'nt':
            return None

        try:
            from winsdk.windows.devices.geolocation import Geolocator, PositionAccuracy
        except Exception:
            return None

        async def _read_position():
            geolocator = Geolocator()
            geolocator.desired_accuracy = PositionAccuracy.HIGH
            geolocator.desired_accuracy_in_meters = 80
            position = await geolocator.get_geoposition_async()
            coords = position.coordinate.point.position
            return float(coords.latitude), float(coords.longitude)

        try:
            try:
                lat, lon = asyncio.run(_read_position())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    lat, lon = loop.run_until_complete(_read_position())
                finally:
                    loop.close()

            return {
                'latitude': lat,
                'longitude': lon,
                'location_text': fallback_text or 'Device location services',
                'source': 'device_location',
            }
        except Exception:
            return None

    def _resolve_via_ip_geolocation(self, fallback_text):
        url = str(self.config.get('ip_geolocation_url', 'https://ipapi.co/json/')).strip()
        timeout = float(self.config.get('http_timeout_seconds', 4.0) or 4.0)

        try:
            req = Request(url, headers={'User-Agent': 'driver-fatigue-monitor/1.0'})
            with urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))

            lat = self._to_float(payload.get('latitude') or payload.get('lat'))
            lon = self._to_float(payload.get('longitude') or payload.get('lon') or payload.get('lng'))

            city = payload.get('city')
            region = payload.get('region')
            country = payload.get('country_name') or payload.get('country')
            pieces = [piece for piece in (city, region, country) if piece]
            location_text = ', '.join(pieces) if pieces else (fallback_text or 'IP geolocation')

            return {
                'latitude': lat,
                'longitude': lon,
                'location_text': location_text,
                'source': 'ip_geolocation',
            }
        except Exception:
            return {
                'latitude': None,
                'longitude': None,
                'location_text': fallback_text or 'Location unavailable',
                'source': 'unavailable',
            }

    def _send_email(self, subject, body, screenshot_path):
        smtp_server = str(self.config.get('smtp_server'))
        smtp_port = int(self.config.get('smtp_port', 587) or 587)
        use_tls = bool(self.config.get('smtp_use_tls', True))
        username = str(self.config.get('smtp_username'))
        password = str(self.config.get('smtp_password'))
        from_email = str(self.config.get('from_email'))
        to_email = str(self.config.get('to_email'))

        message = EmailMessage()
        message['Subject'] = subject
        message['From'] = from_email
        message['To'] = to_email
        message.set_content(body)

        mime_type, _ = mimetypes.guess_type(str(screenshot_path))
        if mime_type:
            maintype, subtype = mime_type.split('/', 1)
        else:
            maintype, subtype = ('application', 'octet-stream')

        with open(screenshot_path, 'rb') as image_file:
            message.add_attachment(
                image_file.read(),
                maintype=maintype,
                subtype=subtype,
                filename=screenshot_path.name,
            )

        with smtplib.SMTP(smtp_server, smtp_port, timeout=12) as smtp:
            if use_tls:
                context = ssl.create_default_context()
                smtp.starttls(context=context)
            smtp.login(username, password)
            smtp.send_message(message)

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        raw = str(value).strip()
        if raw == '':
            return None
        try:
            return float(raw)
        except Exception:
            return None
