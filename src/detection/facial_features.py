import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance


def _resolve_face_mesh_class():
    """Resolve FaceMesh class across MediaPipe package layouts."""
    try:
        return mp.solutions.face_mesh.FaceMesh
    except Exception:
        pass

    try:
        from mediapipe.python.solutions import face_mesh as face_mesh_module
        return face_mesh_module.FaceMesh
    except Exception:
        pass

    return None


class FaceDetector:
    """Detects faces in video frames using MediaPipe."""

    def __init__(self):
        self.face_mesh = None
        face_mesh_class = _resolve_face_mesh_class()
        if face_mesh_class is None:
            mp_version = getattr(mp, '__version__', 'unknown')
            mp_path = getattr(mp, '__file__', 'unknown')
            raise RuntimeError(
                f"MediaPipe FaceMesh API not found. mediapipe version={mp_version}, module={mp_path}. "
                "Install a compatible package, e.g. `pip install mediapipe==0.10.14`, and run with the project venv."
            )

        self.face_mesh = face_mesh_class(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._cached_landmarks = None
        self._cached_bbox = None

    def _process_frame(self, frame):
        """Run a single FaceMesh pass and cache both landmarks and bbox."""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        self._cached_landmarks = None
        self._cached_bbox = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            lm_list = []
            for lm in landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                lm_list.append([x, y])

            lm_array = np.array(lm_list)
            x_min, y_min = np.min(lm_array[:, 0]), np.min(lm_array[:, 1])
            x_max, y_max = np.max(lm_array[:, 0]), np.max(lm_array[:, 1])

            pad_x = int((x_max - x_min) * 0.12)
            pad_y = int((y_max - y_min) * 0.15)

            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)

            self._cached_landmarks = lm_array
            self._cached_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        return self._cached_bbox, self._cached_landmarks

    @staticmethod
    def get_face_bbox_from_landmarks(landmarks, frame_shape):
        """Compute a stable face bounding box from landmarks."""
        if landmarks is None:
            return None
        h, w = frame_shape[:2]
        x_min, y_min = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
        x_max, y_max = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])

        pad_x = int((x_max - x_min) * 0.12)
        pad_y = int((y_max - y_min) * 0.15)

        x_min = max(0, int(x_min - pad_x))
        y_min = max(0, int(y_min - pad_y))
        x_max = min(w, int(x_max + pad_x))
        y_max = min(h, int(y_max + pad_y))
        return (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

    def detect_face(self, frame):
        """
        Detect face in frame.
        Returns: bbox (x, y, w, h) or None if no face detected
        """
        bbox, _ = self._process_frame(frame)
        return bbox

    def get_face_landmarks(self, frame):
        """
        Get facial landmarks from MediaPipe.
        Returns: 468-point landmark array or None
        """
        _, landmarks = self._process_frame(frame)
        return landmarks

    def close(self):
        """Release MediaPipe resources."""
        face_mesh = getattr(self, 'face_mesh', None)
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EyeExtractor:
    """Extracts eye regions from face for eye state classification."""

    # MediaPipe face mesh indices for eyes
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(self, eye_size=224):
        self.eye_size = eye_size

    def extract_eyes(self, frame, landmarks):
        """
        Extract eye regions from frame using landmarks.
        Returns: (left_eye, right_eye, left_bbox, right_bbox) or (None, None, None, None)
        """
        if landmarks is None:
            return None, None, None, None

        try:
            left_eye = self._extract_single_eye(frame, landmarks, self.LEFT_EYE_INDICES)
            right_eye = self._extract_single_eye(frame, landmarks, self.RIGHT_EYE_INDICES)

            left_bbox = self._get_eye_bbox(landmarks, self.LEFT_EYE_INDICES)
            right_bbox = self._get_eye_bbox(landmarks, self.RIGHT_EYE_INDICES)

            return left_eye, right_eye, left_bbox, right_bbox
        except:
            return None, None, None, None

    def _extract_single_eye(self, frame, landmarks, indices):
        """Extract and normalize a single eye region."""
        eye_points = landmarks[indices]

        x_min, x_max = np.min(eye_points[:, 0]), np.max(eye_points[:, 0])
        y_min, y_max = np.min(eye_points[:, 1]), np.max(eye_points[:, 1])

        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        eye_region = frame[y_min:y_max, x_min:x_max]

        if eye_region.size == 0:
            return None

        # Resize to standard size
        eye_resized = cv2.resize(eye_region, (self.eye_size, self.eye_size))
        return cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)

    def _get_eye_bbox(self, landmarks, indices):
        """Get bounding box for an eye."""
        eye_points = landmarks[indices]
        x_min, x_max = np.min(eye_points[:, 0]), np.max(eye_points[:, 0])
        y_min, y_max = np.min(eye_points[:, 1]), np.max(eye_points[:, 1])
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def calculate_eye_aspect_ratio(self, landmarks, indices):
        """
        Calculate Eye Aspect Ratio (EAR).
        Returns: float (higher = more open)
        """
        if landmarks is None:
            return 0

        eye_points = landmarks[indices]

        # Euclidean distances
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])

        ear = (A + B) / (2 * C)
        return ear


class MouthExtractor:
    """Extracts mouth region for yawn detection."""

    # MediaPipe face mesh indices for mouth
    MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

    def __init__(self, mouth_size=224):
        self.mouth_size = mouth_size

    def extract_mouth(self, frame, landmarks):
        """
        Extract mouth region from frame using landmarks.
        Returns: (mouth_region, bbox) or (None, None)
        """
        if landmarks is None:
            return None, None

        try:
            mouth_points = landmarks[self.MOUTH_INDICES]

            x_min, x_max = np.min(mouth_points[:, 0]), np.max(mouth_points[:, 0])
            y_min, y_max = np.min(mouth_points[:, 1]), np.max(mouth_points[:, 1])

            # Add padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)

            mouth_region = frame[y_min:y_max, x_min:x_max]

            if mouth_region.size == 0:
                return None, None

            # Resize to standard size
            mouth_resized = cv2.resize(mouth_region, (self.mouth_size, self.mouth_size))
            mouth_rgb = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2RGB)

            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            return mouth_rgb, bbox
        except:
            return None, None

    def calculate_mouth_aspect_ratio(self, landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR).
        Returns: float (higher = more open)
        """
        if landmarks is None:
            return 0
        try:
            # Use direct MediaPipe lip landmarks for a stable MAR
            # Left/Right corners
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]

            # Outer and inner upper/lower lip centers
            upper_outer = landmarks[13]
            lower_outer = landmarks[14]
            upper_inner = landmarks[0]
            lower_inner = landmarks[17]

            vertical_outer = distance.euclidean(upper_outer, lower_outer)
            vertical_inner = distance.euclidean(upper_inner, lower_inner)
            horizontal = distance.euclidean(mouth_left, mouth_right)

            if horizontal <= 1e-6:
                return 0.0

            mar = (vertical_outer + vertical_inner) / (2.0 * horizontal)
            return float(np.clip(mar, 0.0, 2.0))
        except Exception:
            return 0.0


class HeadPoseEstimator:
    """Estimates head pose (pitch, yaw, roll) using MediaPipe Face Landmarks."""

    # Face landmarks for pose estimation
    POSE_LANDMARKS = {
        'nose': 1,
        'chin': 152,
        'left_eye': 33,
        'right_eye': 263,
        'mouth_left': 61,
        'mouth_right': 291,
    }

    def calculate_head_tilt(self, landmarks):
        """
        Calculate simple head tilt angle from nose and chin.
        Returns: float angle in degrees where ~0 is upright and negative means droop.
        """
        if landmarks is None:
            return 0.0

        try:
            nose = landmarks[self.POSE_LANDMARKS['nose']]
            chin = landmarks[self.POSE_LANDMARKS['chin']]

            dx = float(chin[0] - nose[0])
            dy = float(chin[1] - nose[1])

            angle = np.degrees(np.arctan2(dy, dx))
            # Convert from x-axis angle to tilt relative to vertical axis.
            # Upright face (chin below nose) ~ +90° -> 0° tilt.
            tilt = angle - 90.0
            return float(np.clip(tilt, -90.0, 90.0))
        except Exception:
            return 0.0

    def estimate_head_pose(self, landmarks, frame_shape=None):
        """
        Estimate head pose angles (pitch, yaw, roll) in degrees.
        Returns: (pitch, yaw, roll) tuple in degrees
        """
        if landmarks is None:
            return 0.0, 0.0, 0.0

        try:
            # 3D model points (generic face model)
            model_points = np.array([
                (0, 0, 0),           # Nose tip
                (0, -330, -65),      # Chin
                (-225, 170, -135),   # Left eye
                (225, 170, -135),    # Right eye
                (-150, -150, -125),  # Left mouth corner
                (150, -150, -125),   # Right mouth corner
            ])

            # Get 2D image points from landmarks
            image_points = np.array([
                landmarks[self.POSE_LANDMARKS['nose']],
                landmarks[8],  # Chin
                landmarks[self.POSE_LANDMARKS['left_eye']],
                landmarks[self.POSE_LANDMARKS['right_eye']],
                landmarks[self.POSE_LANDMARKS['mouth_left']],
                landmarks[self.POSE_LANDMARKS['mouth_right']],
            ], dtype=np.float32)

            # Camera matrix (assuming standard webcam intrinsics)
            # Previous implementation used focal_length=image_points.shape[0] (=6),
            # which produced unstable/near-constant pose values.
            if frame_shape is not None:
                frame_h, frame_w = frame_shape[:2]
            else:
                # Fallback derived from landmark spread if frame shape is unavailable
                frame_w = max(int(np.max(landmarks[:, 0]) + 1), 1)
                frame_h = max(int(np.max(landmarks[:, 1]) + 1), 1)

            focal_length = float(max(frame_w, frame_h))
            center_x = frame_w / 2.0
            center_y = frame_h / 2.0
            camera_matrix = np.array([
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1],
            ], dtype=np.float32)

            # Solve PnP
            dist_coeffs = np.zeros(4)
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            if not success:
                return self._estimate_pose_fallback(landmarks)

            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)

            # Fallback when PnP returns unrealistically static near-zero angles
            if abs(pitch) < 0.2 and abs(roll) < 0.2:
                fb_pitch, fb_yaw, fb_roll = self._estimate_pose_fallback(landmarks)
                pitch = fb_pitch if abs(fb_pitch) > abs(pitch) else pitch
                roll = fb_roll if abs(fb_roll) > abs(roll) else roll

            return float(pitch), float(yaw), float(roll)
        except Exception:
            return self._estimate_pose_fallback(landmarks)

    def _estimate_pose_fallback(self, landmarks):
        """Lightweight geometric fallback for pitch/roll/yaw when solvePnP is unstable."""
        try:
            left_eye = landmarks[self.POSE_LANDMARKS['left_eye']].astype(np.float32)
            right_eye = landmarks[self.POSE_LANDMARKS['right_eye']].astype(np.float32)
            nose = landmarks[self.POSE_LANDMARKS['nose']].astype(np.float32)
            chin = landmarks[self.POSE_LANDMARKS['chin']].astype(np.float32)

            eye_dx = float(right_eye[0] - left_eye[0])
            eye_dy = float(right_eye[1] - left_eye[1])
            eye_dist = max(float(np.hypot(eye_dx, eye_dy)), 1e-6)

            roll = np.degrees(np.arctan2(eye_dy, eye_dx))

            # Nose/chin vertical geometry relative to eye distance as proxy for pitch.
            nose_to_chin_y = float(chin[1] - nose[1])
            pitch = ((nose_to_chin_y / eye_dist) - 1.5) * 35.0

            # Nose horizontal offset from eye midpoint as proxy for yaw.
            eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
            yaw = ((float(nose[0]) - eye_mid_x) / eye_dist) * 45.0

            return (
                float(np.clip(pitch, -60.0, 60.0)),
                float(np.clip(yaw, -60.0, 60.0)),
                float(np.clip(roll, -60.0, 60.0)),
            )
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def _rotation_matrix_to_euler_angles(r):
        """Convert rotation matrix to Euler angles (degrees)."""
        sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(r[2, 1], r[2, 2])
            y = np.arctan2(-r[2, 0], sy)
            z = np.arctan2(r[1, 0], r[0, 0])
        else:
            x = np.arctan2(-r[1, 2], r[1, 1])
            y = np.arctan2(-r[2, 0], sy)
            z = 0

        return np.array([x, y, z]) * 180 / np.pi
