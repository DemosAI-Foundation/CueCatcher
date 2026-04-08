"""
CueCatcher Action Recognition

Detects communicative gestures and stereotypical movements from skeleton sequences.

Two modes:
  1. Rule-based (MVP): Hand-crafted detectors for common behaviors
  2. PoseConv3D (production): Learned skeleton-based action recognition

Based on the ASDMotion algorithm (JAMA Network Open, 2024) which achieved
92.53% recall for detecting stereotypical motor movements in 241 ASD children.

Target behaviors:
  - Communicative: reaching, arms_up, hand_leading, pointing, giving
  - Stimming: hand_flapping, arm_waving, rocking, spinning, head_banging
  - Self-regulation: covering_ears, covering_eyes, withdrawal
  - Social: approaching, turning_toward, imitation
"""

import numpy as np
from collections import deque
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


@dataclass
class ActionDetection:
    """A detected action/behavior."""
    action: str
    confidence: float
    start_frame: int
    end_frame: int
    duration_frames: int
    category: str           # communicative, stimming, regulation, social
    comm_signal: str        # what it might mean communicatively
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


class ActionRecognizer:
    """
    Recognizes actions from temporal pose sequences.
    """

    def __init__(self, model_dir: Path, device: str = "cuda:0", fps: int = 30):
        self.model_dir = model_dir
        self.device = device
        self.fps = fps

        self._poseconv3d = None
        self._mode = "rule_based"

        # Pose buffer
        self._pose_buffer: deque = deque(maxlen=fps * 5)  # 5 seconds
        self._frame_indices: deque = deque(maxlen=fps * 5)

        # Cooldown per action type (prevent duplicate detections)
        self._last_detection_frame: dict = {}
        self._cooldown_frames = fps * 2  # 2 second cooldown

    def load(self):
        """Load action recognition model."""
        poseconv3d_path = self.model_dir / "action" / "poseconv3d_ntu60.pth"

        if poseconv3d_path.exists():
            try:
                self._load_poseconv3d(poseconv3d_path)
                self._mode = "poseconv3d"
                logger.info("  ✅ Action: PoseConv3D loaded")
            except Exception as e:
                logger.warning(f"  ⚠️  PoseConv3D failed: {e}")

        if self._mode == "rule_based":
            logger.info("  ✅ Action: rule-based detection (MVP)")

    def _load_poseconv3d(self, model_path: Path):
        """Load PoseConv3D from MMAction2."""
        try:
            from mmaction.apis import init_recognizer
            config = "configs/skeleton/posec3d/slowonly_r50_ntu60_xsub_keypoint.py"
            self._poseconv3d = init_recognizer(config, str(model_path), device=self.device)
        except ImportError:
            raise ImportError("MMAction2 not installed")

    def update(self, keypoints: Optional[np.ndarray], frame_idx: int) -> list[ActionDetection]:
        """
        Add a new pose frame and check for actions.

        Args:
            keypoints: (N, 3) array of x, y, confidence
            frame_idx: current frame number

        Returns:
            List of newly detected actions
        """
        if keypoints is not None:
            self._pose_buffer.append(keypoints.copy())
            self._frame_indices.append(frame_idx)

        # Only analyze every 5 frames
        if frame_idx % 5 != 0 or len(self._pose_buffer) < self.fps // 2:
            return []

        if self._mode == "poseconv3d":
            return self._detect_poseconv3d(frame_idx)
        else:
            return self._detect_rule_based(frame_idx)

    def _detect_rule_based(self, current_frame: int) -> list[ActionDetection]:
        """Rule-based action detection from pose sequences."""
        detections = []
        poses = list(self._pose_buffer)

        if len(poses) < self.fps // 2:
            return detections

        kps = np.array(poses)
        n_frames, n_kp, _ = kps.shape

        if n_kp < 17:
            return detections

        # MediaPipe indices:
        # 0=nose, 11/12=shoulders, 13/14=elbows, 15/16=wrists,
        # 23/24=hips, 25/26=knees, 27/28=ankles

        # ── Hand Flapping ──
        det = self._detect_hand_flapping(kps, current_frame)
        if det:
            detections.append(det)

        # ── Reaching ──
        det = self._detect_reaching(kps, current_frame)
        if det:
            detections.append(det)

        # ── Arms Up ──
        det = self._detect_arms_up(kps, current_frame)
        if det:
            detections.append(det)

        # ── Rocking ──
        det = self._detect_rocking(kps, current_frame)
        if det:
            detections.append(det)

        # ── Covering Ears ──
        det = self._detect_covering_ears(kps, current_frame)
        if det:
            detections.append(det)

        # ── Spinning ──
        det = self._detect_spinning(kps, current_frame)
        if det:
            detections.append(det)

        return detections

    def _detect_hand_flapping(self, kps: np.ndarray, frame: int) -> Optional[ActionDetection]:
        """
        Hand flapping: rapid, small-amplitude wrist movements while
        arms are held at side or in front of body.
        Distinct from arm waving (larger amplitude, whole-arm movement).
        """
        if self._in_cooldown("hand_flapping", frame):
            return None

        # Use last 1 second
        recent = kps[-self.fps:]
        wrists = recent[:, [15, 16], :2]

        # High frequency, low amplitude movement
        if len(wrists) < 10:
            return None

        # Velocity of wrist movement
        velocities = np.diff(wrists, axis=0)
        speed = np.sqrt(np.sum(velocities**2, axis=2))  # shape: (T-1, 2)
        avg_speed = np.mean(speed)

        # Direction changes (oscillation)
        sign_changes = np.sum(np.diff(np.sign(velocities[:, :, 1]), axis=0) != 0, axis=0)
        avg_oscillation = np.mean(sign_changes)

        # Flapping = moderate speed + high oscillation
        if avg_speed > 0.005 and avg_oscillation > 5:
            # Confirm arms are not fully extended (that would be arm waving)
            shoulders = recent[:, [11, 12], :2]
            arm_length = np.mean(np.linalg.norm(wrists - shoulders, axis=2))

            if arm_length < 0.3:  # arms close to body = flapping
                self._last_detection_frame["hand_flapping"] = frame
                return ActionDetection(
                    action="hand_flapping",
                    confidence=min(1.0, avg_oscillation / 10),
                    start_frame=frame - self.fps,
                    end_frame=frame,
                    duration_frames=self.fps,
                    category="stimming",
                    comm_signal="excitement_or_anxiety",
                    features={
                        "oscillation_rate": float(avg_oscillation),
                        "speed": float(avg_speed),
                        "arm_extension": float(arm_length),
                    },
                )

        return None

    def _detect_reaching(self, kps: np.ndarray, frame: int) -> Optional[ActionDetection]:
        """Sustained arm extension toward a target."""
        if self._in_cooldown("reaching", frame):
            return None

        recent = kps[-self.fps:]  # last 1 second
        sustained = 0
        max_ext = 0
        side = None

        for kp in recent:
            for s, (sh, wr) in [("left", (11, 15)), ("right", (12, 16))]:
                dist = np.linalg.norm(kp[wr, :2] - kp[sh, :2])
                if dist > 0.25:
                    sustained += 1
                    if dist > max_ext:
                        max_ext = dist
                        side = s
                    break

        if sustained >= self.fps // 2 and side:
            self._last_detection_frame["reaching"] = frame
            return ActionDetection(
                action=f"reaching_{side}",
                confidence=min(1.0, max_ext / 0.4),
                start_frame=frame - sustained,
                end_frame=frame,
                duration_frames=sustained,
                category="communicative",
                comm_signal="request",
                features={"side": side, "extension": float(max_ext)},
            )

        return None

    def _detect_arms_up(self, kps: np.ndarray, frame: int) -> Optional[ActionDetection]:
        """Both arms raised above shoulders — request to be picked up."""
        if self._in_cooldown("arms_up", frame):
            return None

        recent = kps[-self.fps // 2:]  # last 0.5 seconds
        sustained = 0

        for kp in recent:
            l_wr_y = kp[15, 1]
            r_wr_y = kp[16, 1]
            l_sh_y = kp[11, 1]
            r_sh_y = kp[12, 1]
            # y=0 is top of image in MediaPipe
            if l_wr_y < l_sh_y and r_wr_y < r_sh_y:
                sustained += 1

        if sustained >= self.fps // 3:
            self._last_detection_frame["arms_up"] = frame
            return ActionDetection(
                action="arms_up",
                confidence=0.85,
                start_frame=frame - sustained,
                end_frame=frame,
                duration_frames=sustained,
                category="communicative",
                comm_signal="request_pickup",
            )

        return None

    def _detect_rocking(self, kps: np.ndarray, frame: int) -> Optional[ActionDetection]:
        """Periodic body oscillation."""
        if self._in_cooldown("rocking", frame):
            return None

        if len(kps) < self.fps * 2:
            return None

        recent = kps[-self.fps * 2:]
        hip_center = (recent[:, 23, :2] + recent[:, 24, :2]) / 2
        hip_var = np.var(hip_center, axis=0).sum()

        if hip_var > 0.001:
            centered = hip_center[:, 0] - hip_center[:, 0].mean()
            zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)

            if zero_crossings >= 4:
                self._last_detection_frame["rocking"] = frame
                return ActionDetection(
                    action="rocking",
                    confidence=min(1.0, zero_crossings / 8),
                    start_frame=frame - self.fps * 2,
                    end_frame=frame,
                    duration_frames=self.fps * 2,
                    category="stimming",
                    comm_signal="self_regulation",
                    features={
                        "cycles": int(zero_crossings // 2),
                        "amplitude": float(np.std(centered)),
                    },
                )

        return None

    def _detect_covering_ears(self, kps: np.ndarray, frame: int) -> Optional[ActionDetection]:
        """Hands near ears — sensory overload signal."""
        if self._in_cooldown("covering_ears", frame):
            return None

        recent = kps[-self.fps // 2:]
        sustained = 0

        for kp in recent:
            # MediaPipe: 7/8 = ear landmarks, 15/16 = wrists
            l_ear = kp[7, :2] if kp.shape[0] > 8 else kp[0, :2]
            r_ear = kp[8, :2] if kp.shape[0] > 8 else kp[0, :2]
            l_wrist = kp[15, :2]
            r_wrist = kp[16, :2]

            l_dist = np.linalg.norm(l_wrist - l_ear)
            r_dist = np.linalg.norm(r_wrist - r_ear)

            if l_dist < 0.08 or r_dist < 0.08:
                sustained += 1

        if sustained >= self.fps // 4:
            self._last_detection_frame["covering_ears"] = frame
            return ActionDetection(
                action="covering_ears",
                confidence=0.75,
                start_frame=frame - sustained,
                end_frame=frame,
                duration_frames=sustained,
                category="regulation",
                comm_signal="sensory_overload",
            )

        return None

    def _detect_spinning(self, kps: np.ndarray, frame: int) -> Optional[ActionDetection]:
        """Full-body rotation / spinning."""
        if self._in_cooldown("spinning", frame):
            return None

        if len(kps) < self.fps:
            return None

        recent = kps[-self.fps:]

        # Track shoulder orientation change over time
        l_shoulders = recent[:, 11, :2]
        r_shoulders = recent[:, 12, :2]
        shoulder_vectors = r_shoulders - l_shoulders

        # Angular change
        angles = np.arctan2(shoulder_vectors[:, 1], shoulder_vectors[:, 0])
        total_rotation = np.abs(np.sum(np.diff(angles)))

        if total_rotation > np.pi:  # >180 degrees of rotation
            self._last_detection_frame["spinning"] = frame
            return ActionDetection(
                action="spinning",
                confidence=min(1.0, total_rotation / (2 * np.pi)),
                start_frame=frame - self.fps,
                end_frame=frame,
                duration_frames=self.fps,
                category="stimming",
                comm_signal="sensory_seeking",
                features={"rotation_degrees": float(np.degrees(total_rotation))},
            )

        return None

    def _in_cooldown(self, action: str, current_frame: int) -> bool:
        """Check if an action type is in cooldown."""
        last = self._last_detection_frame.get(action, -999)
        return (current_frame - last) < self._cooldown_frames

    def _detect_poseconv3d(self, frame: int) -> list[ActionDetection]:
        """Production action recognition using PoseConv3D."""
        # TODO: Implement when PoseConv3D is loaded
        return self._detect_rule_based(frame)

    def unload(self):
        self._poseconv3d = None
        self._pose_buffer.clear()
