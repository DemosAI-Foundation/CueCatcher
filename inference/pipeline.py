"""
CueCatcher Inference Pipeline — Production
Orchestrates all perception modules on the RTX 3090.

VRAM Budget (24 GB):
  YOLO11 + RTMPose:  ~5 GB   (pose.py)
  L2CS-Net:          ~1 GB   (gaze.py)
  LibreFace:         ~1 GB   (face.py)
  PANNs CNN14:       ~1 GB   (audio.py)
  PoseConv3D:        ~3 GB   (action.py)
  Voxtral Q4:       ~3-4 GB  (voice/tts.py)
  ─────────────────────────
  Total:            ~14-15 GB → 9-10 GB headroom

Each module auto-falls back to lightweight/CPU alternatives if production
models aren't downloaded yet. MediaPipe MVP works out of the box.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from config.settings import settings
from inference.pose import PoseEstimator
from inference.gaze import GazeEstimator
from inference.face import FaceAnalyzer
from inference.audio import AudioAnalyzer
from inference.action import ActionRecognizer
from inference.temporal import TemporalEngine


@dataclass
class FrameDetections:
    """All detections for a single video frame."""
    frame_idx: int = 0
    timestamp: float = 0.0

    # Pose
    pose_keypoints: Optional[np.ndarray] = None
    person_bbox: Optional[np.ndarray] = None
    person_confidence: float = 0.0
    num_keypoints: int = 0

    # Gaze / Head Pose
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    eye_gaze_yaw: float = 0.0
    eye_gaze_pitch: float = 0.0
    fused_gaze_yaw: float = 0.0
    fused_gaze_pitch: float = 0.0
    gaze_target: Optional[str] = None
    looking_at_camera: bool = False
    gaze_alternation: Optional[dict] = None

    # Face
    face_detected: bool = False
    expression: Optional[str] = None
    expression_confidence: float = 0.0
    calibrated_expression: Optional[str] = None
    calibrated_expression_confidence: float = 0.0
    action_units: Optional[dict] = None
    mouth_openness: float = 0.0
    smile_score: float = 0.0

    # Audio
    vocalization_class: Optional[str] = None
    vocalization_confidence: float = 0.0
    pitch_hz: float = 0.0
    pitch_trend: str = "flat"
    energy_db: float = -60.0
    is_vocalization: bool = False
    vocalization_onset: bool = False

    # Actions
    actions_detected: list = field(default_factory=list)

    # Temporal
    new_episodes: list = field(default_factory=list)
    child_state: str = "idle"
    child_state_confidence: float = 0.5
    child_state_duration_s: float = 0.0
    state_changed: bool = False

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


class InferencePipeline:
    """
    Production pipeline wiring all modules together.
    Each module loads its best available model with MVP fallback.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.loaded = False
        model_dir = settings.model_dir

        self._pose = PoseEstimator(model_dir, device)
        self._gaze = GazeEstimator(model_dir, device)
        self._face = FaceAnalyzer(model_dir, settings.face_calibration_dir, device)
        self._audio = AudioAnalyzer(model_dir, device, settings.audio_sample_rate)
        self._action = ActionRecognizer(model_dir, device, settings.video_fps)
        self._temporal = TemporalEngine(fps=settings.video_fps)

        self._last_audio = None
        self._frame_times: deque = deque(maxlen=100)

        logger.info(f"Pipeline targeting: {device}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {name} ({vram:.1f} GB)")

    def load_models(self):
        """Load all modules. Each handles its own fallback."""
        logger.info("═══ Loading inference modules ═══")
        t0 = time.time()
        self._pose.load()
        self._gaze.load()
        self._face.load()
        self._audio.load()
        self._action.load()
        self.loaded = True
        logger.info(f"═══ All loaded in {time.time() - t0:.1f}s ═══")
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / (1024**3)
            r = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"VRAM: {a:.1f} GB alloc, {r:.1f} GB reserved")

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """
        Full perception pipeline on one BGR frame.
        Flow: frame → pose → gaze → face → action → temporal → dict
        """
        t0 = time.time()
        det = FrameDetections(frame_idx=frame_idx, timestamp=time.time())

        # 1. Pose
        pr = self._pose.estimate(frame)
        det.pose_keypoints = pr.get("keypoints")
        det.person_bbox = pr.get("bbox")
        det.person_confidence = pr.get("person_score", 0.0)
        det.num_keypoints = pr.get("num_keypoints", 0)

        # 2. Gaze
        gr = self._gaze.estimate(frame)
        det.head_yaw = gr.head_yaw
        det.head_pitch = gr.head_pitch
        det.head_roll = gr.head_roll
        det.eye_gaze_yaw = gr.eye_yaw
        det.eye_gaze_pitch = gr.eye_pitch
        det.fused_gaze_yaw = gr.fused_yaw
        det.fused_gaze_pitch = gr.fused_pitch
        det.gaze_target = gr.target
        det.looking_at_camera = gr.looking_at_camera
        if frame_idx % 15 == 0:
            det.gaze_alternation = self._gaze.detect_gaze_alternation()

        # 3. Face
        fr = self._face.analyze(frame)
        det.face_detected = fr.detected
        det.expression = fr.expression
        det.expression_confidence = fr.expression_confidence
        det.calibrated_expression = fr.calibrated_expression
        det.calibrated_expression_confidence = fr.calibrated_confidence
        det.action_units = fr.action_units
        det.mouth_openness = fr.mouth_openness
        det.smile_score = fr.smile_score

        # 4. Actions
        actions = self._action.update(det.pose_keypoints, frame_idx)
        det.actions_detected = [a.to_dict() for a in actions]

        # 5. Audio (injected from last async chunk)
        if self._last_audio:
            det.vocalization_class = self._last_audio.vocalization_class
            det.vocalization_confidence = self._last_audio.class_confidence
            det.pitch_hz = self._last_audio.pitch_hz
            det.pitch_trend = self._last_audio.pitch_trend
            det.energy_db = self._last_audio.energy_db
            det.is_vocalization = self._last_audio.is_vocalization
            det.vocalization_onset = self._last_audio.onset

        # 6. Temporal
        tr = self._temporal.update(det.to_dict(), frame_idx)
        det.new_episodes = tr.get("new_episodes", [])
        det.child_state = tr.get("state", "idle")
        det.child_state_confidence = tr.get("state_confidence", 0.5)
        det.child_state_duration_s = tr.get("state_duration_s", 0.0)
        det.state_changed = tr.get("state_changed", False)

        self._frame_times.append(time.time() - t0)
        return det.to_dict()

    def process_audio(self, pcm_bytes: bytes) -> dict:
        """Process audio chunk through analyzer."""
        result = self._audio.analyze(pcm_bytes)
        self._last_audio = result
        return result.to_dict()

    def get_communication_summary(self, seconds: int = 300) -> dict:
        return self._temporal.get_communication_summary(seconds)

    @property
    def avg_frame_time_ms(self) -> float:
        return float(np.mean(list(self._frame_times)) * 1000) if self._frame_times else 0.0

    @property
    def current_child_state(self) -> str:
        return self._temporal.current_state.state.value

    def unload(self):
        self._pose.unload()
        self._gaze.unload()
        self._face.unload()
        self._audio.unload()
        self._action.unload()
        torch.cuda.empty_cache()
        self.loaded = False
        logger.info("Pipeline unloaded")
