"""CueCatcher configuration — all tunable parameters in one place."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # ── Server ──
    host: str = "127.0.0.1"
    port: int = 8084
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # ── Video Capture ──
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 30
    video_codec: str = "h264"

    # ── GPU / Inference ──
    device: str = "cuda:0"
    model_dir: Path = Path("/models")
    inference_fp16: bool = True
    max_batch_size: int = 1  # single child, single camera

    # ── Model VRAM Budget (approximate, in GB) ──
    # YOLO11-Pose:    ~2.0 GB
    # RTMPose-l:      ~3.0 GB
    # L2CS-Net:       ~1.0 GB
    # LibreFace:      ~1.0 GB
    # PANNs CNN14:    ~1.0 GB
    # PoseConv3D:     ~3.0 GB
    # ─────────────────────────
    # Total:         ~11.0 GB  (leaves ~13 GB for Voxtral TTS)
    # Voxtral Q4:    ~3-4 GB
    # ─────────────────────────
    # Grand total:   ~14-15 GB of 24 GB

    # ── Pose Estimation ──
    pose_model: str = "rtmpose-l"
    pose_det_model: str = "yolo11n-pose"
    pose_score_threshold: float = 0.3

    # ── Gaze / Head Pose ──
    gaze_model: str = "l2cs-net"
    head_pose_weight: float = 0.7   # head pose weighted more than eye gaze
    eye_gaze_weight: float = 0.3    # for age 3-4, head orientation is primary

    # ── Face ──
    face_model: str = "libreface"
    face_det_model: str = "retinaface"
    face_calibration_dir: Path = Path("/data/calibration")

    # ── Audio ──
    audio_model: str = "panns-cnn14"
    audio_sample_rate: int = 16000
    audio_chunk_ms: int = 500       # process 500ms audio chunks
    vocalization_categories: list[str] = [
        "distress_cry", "pleasure", "attention_seeking",
        "rhythmic_repetitive", "babble", "silence",
        "environmental",  # non-child sounds
    ]

    # ── Action Recognition ──
    action_model: str = "poseconv3d"
    action_window_frames: int = 90   # 3 seconds at 30fps
    action_stride_frames: int = 15   # 0.5 second stride

    # ── Temporal Analysis ──
    tier1_buffer_seconds: float = 10.0     # per-frame ring buffer
    tier2_window_seconds: float = 5.0      # behavioral episode window
    tier2_stride_seconds: float = 0.5
    tier3_state_window_seconds: float = 60.0  # state machine context

    # ── Interpreter ──
    confidence_high: float = 0.80
    confidence_medium: float = 0.50
    confidence_low: float = 0.30
    always_show_null_hypothesis: bool = True  # "might not be communicative"

    # ── Communication Matrix Levels ──
    # Level I:   Pre-Intentional (reflexive)
    # Level II:  Intentional but not directed
    # Level III: Unconventional directed communication
    # Level IV:  Conventional gestures
    comm_matrix_primary_levels: list[int] = [1, 2, 3]

    # ── Voxtral TTS ──
    tts_enabled: bool = True
    tts_model: str = "Mistral-AI/Voxtral-4B-TTS-2603"
    tts_quantization: str = "int4"      # keep VRAM low
    tts_voice_reference: Optional[str] = None  # path to parent voice clip
    tts_voice_reference_min_seconds: float = 5.0
    tts_voice_reference_max_seconds: float = 25.0
    tts_speak_confidence_threshold: float = 0.70  # only speak high-confidence
    tts_cooldown_seconds: float = 3.0   # minimum gap between utterances
    tts_max_utterance_length: int = 100  # chars

    # ── Storage ──
    redis_url: str = "redis://localhost:6379/0"
    db_url: str = "postgresql+asyncpg://CueCatcher:CueCatcher@localhost:5432/CueCatcher"

    # ── Privacy ──
    store_raw_video: bool = False      # default: only store metadata
    session_retention_days: int = 90
    require_consent_on_start: bool = True

    model_config = {"env_prefix": "CueCatcher_"}

    def model_post_init(self, __context):
        import sys
        if sys.platform == "win32":
            base = Path(__file__).parent.parent
            if self.model_dir == Path("/models"):
                self.model_dir = base / "models"
            if self.face_calibration_dir == Path("/data/calibration"):
                self.face_calibration_dir = base / "data" / "calibration"


settings = Settings()
