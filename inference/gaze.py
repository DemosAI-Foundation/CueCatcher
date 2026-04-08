"""
CueCatcher Gaze Estimation

Head pose is weighted 70% / eye gaze 30% because:
  1. At age 3-4, children orient their whole head toward targets
  2. 9p deletion causes strabismus and nystagmus → unreliable eye tracking
  3. Head pose is more robust from a phone camera at variable distances

Uses:
  - MediaPipe Face Mesh (468 landmarks) → PnP head pose
  - L2CS-Net (optional) → eye gaze direction
  - 6DRepNet (optional) → 6DOF head pose
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from collections import deque
from loguru import logger


@dataclass
class GazeResult:
    """Combined gaze estimation result."""
    # Head pose (primary signal)
    head_yaw: float = 0.0       # left(-) / right(+) in degrees
    head_pitch: float = 0.0     # up(-) / down(+) in degrees
    head_roll: float = 0.0      # tilt in degrees
    head_confidence: float = 0.0

    # Eye gaze (secondary signal)
    eye_yaw: float = 0.0
    eye_pitch: float = 0.0
    eye_confidence: float = 0.0

    # Fused gaze direction
    fused_yaw: float = 0.0
    fused_pitch: float = 0.0

    # Gaze target classification
    target: str = "unknown"     # 'forward', 'left', 'right', 'up', 'down', 'away'
    looking_at_camera: bool = False

    # Gaze alternation tracking
    recent_targets: list = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop('recent_targets', None)
        return d


class GazeEstimator:
    """
    Multi-source gaze estimation with child-appropriate weighting.
    """

    def __init__(self, model_dir: Path, device: str = "cuda:0"):
        self.model_dir = model_dir
        self.device = device

        # Weights (configurable)
        self.head_weight = 0.7
        self.eye_weight = 0.3

        # Gaze history for alternation detection
        self._gaze_history: deque = deque(maxlen=90)  # 3 seconds at 30fps
        self._target_history: deque = deque(maxlen=60)

        # 3D face model for PnP
        self._model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye corner
            [225.0, 170.0, -135.0],    # Right eye corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0],   # Right mouth corner
        ], dtype=np.float64)

        self._face_mesh = None
        self._l2cs = None

    def load(self):
        """Load gaze models."""
        # Always load MediaPipe for head pose
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("  ✅ Gaze: MediaPipe FaceMesh (head pose)")
        except Exception as e:
            logger.error(f"  ❌ FaceMesh failed: {e}")

        # Optional: L2CS-Net for eye gaze
        l2cs_path = self.model_dir / "gaze" / "l2cs_gaze360.pkl"
        if l2cs_path.exists():
            try:
                self._load_l2cs(l2cs_path)
                logger.info("  ✅ Gaze: L2CS-Net loaded (eye gaze)")
            except Exception as e:
                logger.warning(f"  ⚠️  L2CS-Net failed: {e}")

    def _load_l2cs(self, model_path: Path):
        """Load L2CS-Net for eye gaze estimation."""
        import torch
        # L2CS-Net uses ResNet-50 backbone
        # Loading requires the l2cs package or manual checkpoint loading
        try:
            from l2cs import Pipeline
            self._l2cs = Pipeline(
                weights=str(model_path),
                arch="ResNet50",
                device=torch.device(self.device),
            )
        except ImportError:
            logger.warning("  l2cs package not installed, using head pose only")

    def estimate(self, frame: np.ndarray, face_landmarks=None) -> GazeResult:
        """
        Estimate gaze direction from a BGR video frame.

        Args:
            frame: BGR numpy array
            face_landmarks: optional pre-computed MediaPipe face landmarks

        Returns:
            GazeResult with fused head + eye gaze
        """
        result = GazeResult()
        h, w = frame.shape[:2]

        # ── Head Pose from Face Mesh ──
        if face_landmarks is None and self._face_mesh:
            rgb = frame[:, :, ::-1].copy()
            mp_result = self._face_mesh.process(rgb)
            if mp_result.multi_face_landmarks:
                face_landmarks = mp_result.multi_face_landmarks[0]

        if face_landmarks:
            head_pose = self._compute_head_pose(face_landmarks, w, h)
            if head_pose:
                result.head_yaw = head_pose[1]
                result.head_pitch = head_pose[0]
                result.head_roll = head_pose[2]
                result.head_confidence = 0.8

        # ── Eye Gaze from L2CS-Net ──
        if self._l2cs:
            try:
                eye_result = self._l2cs.step(frame)
                if eye_result and len(eye_result.yaw) > 0:
                    result.eye_yaw = float(np.degrees(eye_result.yaw[0]))
                    result.eye_pitch = float(np.degrees(eye_result.pitch[0]))
                    result.eye_confidence = 0.6
            except Exception:
                pass

        # ── Fuse head + eye gaze ──
        if result.head_confidence > 0 or result.eye_confidence > 0:
            total_weight = (self.head_weight * result.head_confidence +
                          self.eye_weight * result.eye_confidence)
            if total_weight > 0:
                result.fused_yaw = (
                    self.head_weight * result.head_confidence * result.head_yaw +
                    self.eye_weight * result.eye_confidence * result.eye_yaw
                ) / total_weight
                result.fused_pitch = (
                    self.head_weight * result.head_confidence * result.head_pitch +
                    self.eye_weight * result.eye_confidence * result.eye_pitch
                ) / total_weight

        # ── Classify gaze target ──
        result.target = self._classify_target(result.fused_yaw, result.fused_pitch)
        result.looking_at_camera = (
            abs(result.fused_yaw) < 10 and abs(result.fused_pitch) < 10
        )

        # ── Track history ──
        self._gaze_history.append({
            "yaw": result.fused_yaw,
            "pitch": result.fused_pitch,
            "target": result.target,
        })
        self._target_history.append(result.target)

        return result

    def _compute_head_pose(self, face, w: int, h: int) -> Optional[tuple]:
        """Compute head pose from MediaPipe face landmarks using PnP."""
        # Key landmark indices
        nose = face.landmark[1]
        chin = face.landmark[152]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]
        left_mouth = face.landmark[61]
        right_mouth = face.landmark[291]

        image_points = np.array([
            [nose.x * w, nose.y * h],
            [chin.x * w, chin.y * h],
            [left_eye.x * w, left_eye.y * h],
            [right_eye.x * w, right_eye.y * h],
            [left_mouth.x * w, left_mouth.y * h],
            [right_mouth.x * w, right_mouth.y * h],
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                self._model_points, image_points, camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                return self._rotation_to_euler(rmat)
        except Exception:
            pass

        return None

    def _rotation_to_euler(self, rmat: np.ndarray) -> tuple:
        """Convert rotation matrix to (pitch, yaw, roll) in degrees."""
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll = 0.0
        return pitch, yaw, roll

    def _classify_target(self, yaw: float, pitch: float) -> str:
        """Classify gaze direction into a named target."""
        if abs(yaw) < 10 and abs(pitch) < 10:
            return "forward"
        elif yaw < -25:
            return "left"
        elif yaw > 25:
            return "right"
        elif pitch < -15:
            return "up"
        elif pitch > 15:
            return "down"
        elif abs(yaw) > 40 or abs(pitch) > 30:
            return "away"
        else:
            return "slight_" + ("left" if yaw < 0 else "right")

    def detect_gaze_alternation(self, window_seconds: float = 2.0, fps: int = 30) -> dict:
        """
        Detect gaze alternation — the hallmark of intentional communication.

        Returns:
            {
                "detected": bool,
                "shift_count": int,
                "targets_visited": list[str],
                "confidence": float,
            }
        """
        window = int(window_seconds * fps)
        recent = list(self._gaze_history)[-window:]

        if len(recent) < fps:
            return {"detected": False, "shift_count": 0, "targets_visited": [], "confidence": 0}

        yaws = [g["yaw"] for g in recent]
        targets = [g["target"] for g in recent]

        # Count significant direction changes
        diffs = np.abs(np.diff(yaws))
        shifts = int(np.sum(diffs > 15))

        # Unique targets visited
        unique_targets = list(set(targets) - {"unknown"})

        detected = shifts >= 2 and len(unique_targets) >= 2
        confidence = min(1.0, shifts / 4) if detected else 0.0

        return {
            "detected": detected,
            "shift_count": shifts,
            "targets_visited": unique_targets,
            "confidence": confidence,
        }

    def unload(self):
        if self._face_mesh:
            self._face_mesh.close()
        self._l2cs = None
