"""
CueCatcher Face Analysis — Expression Detection with Calibration

Uses LibreFace (or MediaPipe fallback) for facial Action Unit detection,
overlaid with child-specific calibration data to compensate for:
  - Craniofacial features of 9p deletion (hypertelorism, midface hypoplasia)
  - Facial hypotonia (reduced expression range)
  - Different baseline geometry than adult training data

The calibration model (from scripts/calibrate.py) defines the child's
neutral face geometry, and expressions are detected as DELTAS from
that personal baseline rather than absolute AU values.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class FaceAnalysis:
    """Facial expression analysis result."""
    detected: bool = False

    # Action Units (AU intensities, 0.0-5.0 scale)
    action_units: dict = field(default_factory=dict)

    # Expression classification
    expression: str = "neutral"
    expression_confidence: float = 0.0

    # Calibrated expression (relative to THIS child's neutral)
    calibrated_expression: str = "neutral"
    calibrated_confidence: float = 0.0

    # Key metrics
    mouth_openness: float = 0.0       # 0 = closed, 1 = wide open
    brow_raise: float = 0.0           # 0 = neutral, 1 = fully raised
    eye_squeeze: float = 0.0          # 0 = open, 1 = squeezed shut
    smile_score: float = 0.0          # 0 = no smile, 1 = big smile

    def to_dict(self) -> dict:
        return self.__dict__


class FaceAnalyzer:
    """
    Facial expression analysis with child-specific calibration.
    """

    def __init__(self, model_dir: Path, calibration_dir: Path, device: str = "cuda:0"):
        self.model_dir = model_dir
        self.calibration_dir = calibration_dir
        self.device = device

        self._libreface = None
        self._mode = "none"

        # Calibration data
        self._calibration: Optional[dict] = None
        self._neutral_baseline: Optional[dict] = None
        self._expression_deltas: Optional[dict] = None
        self._face_metrics: Optional[dict] = None

    def load(self):
        """Load face analysis model and calibration data."""
        # Try LibreFace
        au_path = self.model_dir / "face" / "libreface_au.pth"
        fer_path = self.model_dir / "face" / "libreface_fer.pth"

        if au_path.exists() and fer_path.exists():
            try:
                self._load_libreface(au_path, fer_path)
                self._mode = "libreface"
                logger.info("  ✅ Face: LibreFace loaded (12 AUs + 7 expressions)")
            except Exception as e:
                logger.warning(f"  ⚠️  LibreFace failed: {e}")

        if self._mode == "none":
            self._mode = "geometric"
            logger.info("  ✅ Face: geometric analysis from landmarks (MVP)")

        # Load calibration if available
        self._load_calibration()

    def _load_libreface(self, au_path: Path, fer_path: Path):
        """Load LibreFace models."""
        import torch

        # LibreFace uses Swin Transformer base
        # For production: use the official libreface package
        try:
            import libreface
            self._libreface = libreface.get_facial_expression_model(
                au_model_path=str(au_path),
                fer_model_path=str(fer_path),
                device=self.device,
            )
        except ImportError:
            # Manual loading fallback
            logger.info("  Using manual LibreFace checkpoint loading")
            self._libreface_au = torch.load(str(au_path), map_location=self.device)
            self._libreface_fer = torch.load(str(fer_path), map_location=self.device)

    def _load_calibration(self):
        """Load child-specific face calibration data."""
        model_path = self.calibration_dir / "model.json"
        if not model_path.exists():
            logger.info("  ℹ️  No face calibration found. Run: python scripts/calibrate.py")
            return

        try:
            with open(model_path) as f:
                self._calibration = json.load(f)

            self._neutral_baseline = self._calibration.get("neutral_baseline", {})
            self._expression_deltas = self._calibration.get("expression_deltas", {})
            self._face_metrics = self._calibration.get("face_metrics", {})

            logger.info(f"  ✅ Face calibration loaded")
            if self._face_metrics.get("eye_distance_ratio", 0) > 0.35:
                logger.info("     Hypertelorism detected — calibrated expression ranges active")
        except Exception as e:
            logger.warning(f"  ⚠️  Calibration load failed: {e}")

    def analyze(self, frame: np.ndarray, face_landmarks=None) -> FaceAnalysis:
        """
        Analyze facial expression from a frame or pre-computed landmarks.
        """
        result = FaceAnalysis()

        if face_landmarks is None:
            result.detected = False
            return result

        result.detected = True

        if self._mode == "libreface":
            result = self._analyze_libreface(frame, result)
        else:
            result = self._analyze_geometric(face_landmarks, frame.shape[1], frame.shape[0], result)

        # Apply calibration overlay
        if self._calibration:
            result = self._apply_calibration(result)

        return result

    def _analyze_libreface(self, frame: np.ndarray, result: FaceAnalysis) -> FaceAnalysis:
        """Run LibreFace for AU and expression detection."""
        if self._libreface is None:
            return result

        try:
            # LibreFace expects a cropped face image
            prediction = self._libreface.predict(frame)
            if prediction:
                result.action_units = prediction.get("action_units", {})
                result.expression = prediction.get("expression", "neutral")
                result.expression_confidence = prediction.get("confidence", 0.0)

                # Extract key metrics from AUs
                result.mouth_openness = result.action_units.get("AU25", 0) / 5.0  # jaw drop
                result.brow_raise = max(
                    result.action_units.get("AU1", 0),
                    result.action_units.get("AU2", 0),
                ) / 5.0
                result.eye_squeeze = result.action_units.get("AU43", 0) / 5.0
                result.smile_score = (
                    result.action_units.get("AU6", 0) +   # cheek raise
                    result.action_units.get("AU12", 0)     # lip corner pull
                ) / 10.0
        except Exception as e:
            logger.debug(f"LibreFace inference error: {e}")

        return result

    def _analyze_geometric(self, face_landmarks, w: int, h: int, result: FaceAnalysis) -> FaceAnalysis:
        """
        Basic expression analysis from landmark geometry.
        Used as MVP when LibreFace isn't available.
        """
        face = face_landmarks

        # Mouth openness
        upper_lip = face.landmark[13]
        lower_lip = face.landmark[14]
        left_mouth = face.landmark[61]
        right_mouth = face.landmark[291]

        mouth_open_px = abs(upper_lip.y - lower_lip.y) * h
        mouth_width_px = abs(right_mouth.x - left_mouth.x) * w
        result.mouth_openness = min(1.0, (mouth_open_px / (mouth_width_px + 1)) / 0.5)

        # Eye openness
        l_eye_top = face.landmark[159]
        l_eye_bot = face.landmark[145]
        r_eye_top = face.landmark[386]
        r_eye_bot = face.landmark[374]

        l_eye_open = abs(l_eye_top.y - l_eye_bot.y) * h
        r_eye_open = abs(r_eye_top.y - r_eye_bot.y) * h
        avg_eye_open = (l_eye_open + r_eye_open) / 2
        result.eye_squeeze = max(0, 1.0 - avg_eye_open / (h * 0.03))

        # Brow raise
        l_brow = face.landmark[105]
        r_brow = face.landmark[334]
        l_eye = face.landmark[33]
        r_eye = face.landmark[263]

        l_brow_dist = abs(l_brow.y - l_eye.y) * h
        r_brow_dist = abs(r_brow.y - r_eye.y) * h
        avg_brow = (l_brow_dist + r_brow_dist) / 2
        result.brow_raise = min(1.0, avg_brow / (h * 0.06))

        # Smile (mouth width relative to face + cheek raise)
        face_width = abs(face.landmark[454].x - face.landmark[234].x) * w
        mouth_ratio = mouth_width_px / (face_width + 1)
        # Mouth corner elevation
        mouth_center_y = (left_mouth.y + right_mouth.y) / 2
        nose_y = face.landmark[1].y
        mouth_elevation = nose_y - mouth_center_y  # positive = corners up
        result.smile_score = min(1.0, max(0, mouth_ratio - 0.3) * 3 + max(0, mouth_elevation * 10))

        # Classify
        mouth_ratio_val = mouth_open_px / (mouth_width_px + 1)

        if mouth_ratio_val > 0.4 and result.brow_raise > 0.5:
            result.expression = "distress"
            result.expression_confidence = 0.5
        elif result.smile_score > 0.4:
            result.expression = "happy"
            result.expression_confidence = 0.4 + result.smile_score * 0.3
        elif result.eye_squeeze > 0.7:
            result.expression = "eyes_closed"
            result.expression_confidence = 0.6
        elif result.brow_raise > 0.6:
            result.expression = "surprise"
            result.expression_confidence = 0.4
        else:
            result.expression = "neutral"
            result.expression_confidence = 0.5

        return result

    def _apply_calibration(self, result: FaceAnalysis) -> FaceAnalysis:
        """
        Adjust expression detection using child-specific calibration.

        The key insight: a child with facial hypotonia has a smaller
        dynamic range of expressions. What looks like "neutral" on
        a standard model might be "happy" for THIS child.
        """
        if not self._expression_deltas:
            result.calibrated_expression = result.expression
            result.calibrated_confidence = result.expression_confidence
            return result

        # Compare current metrics against calibrated baselines
        best_match = "neutral"
        best_score = 0.0

        for state_name, deltas in self._expression_deltas.items():
            # Score how well current metrics match this calibrated state
            score = 0.0
            count = 0

            # Mouth match
            if "upper_lip" in deltas and "lower_lip" in deltas:
                expected_mouth = abs(deltas.get("lower_lip", {}).get("dy", 0))
                if expected_mouth > 0:
                    actual = result.mouth_openness
                    match = 1.0 - min(1.0, abs(actual - expected_mouth / 20) / 0.3)
                    score += match
                    count += 1

            # Use overall geometric similarity
            if count > 0:
                score /= count
                if score > best_score:
                    best_score = score
                    best_match = state_name

        result.calibrated_expression = best_match
        result.calibrated_confidence = best_score * 0.7  # conservative

        # Boost confidence if calibrated and standard models agree
        if result.calibrated_expression == result.expression:
            result.calibrated_confidence = min(1.0, result.calibrated_confidence + 0.15)

        return result

    def unload(self):
        self._libreface = None
