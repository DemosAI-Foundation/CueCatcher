"""
CueCatcher Session Recording & Replay
"""
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from collections import Counter
import numpy as np
from loguru import logger

try:
    import asyncpg
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class SessionSummary:
    """Summary statistics for a completed session."""
    session_id: str
    started_at: str
    ended_at: str
    duration_minutes: float
    total_frames: int = 0
    avg_fps: float = 0.0
    avg_frame_time_ms: float = 0.0
    total_episodes: int = 0
    total_interpretations: int = 0
    confirmed_interpretations: int = 0
    rejected_interpretations: int = 0
    episodes_by_type: dict = field(default_factory=dict)
    episodes_by_function: dict = field(default_factory=dict)
    state_durations: dict = field(default_factory=dict)
    state_transitions: int = 0
    most_common_state: str = "idle"
    highest_comm_level_observed: int = 1
    gaze_alternation_count: int = 0
    coordinated_signals_count: int = 0
    button_presses: int = 0
    button_breakdown: dict = field(default_factory=dict)
    vocalization_seconds: float = 0.0
    vocalization_types: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class SessionRecorder:
    """Records session data to SQL database and optionally to video."""

    def __init__(self, db_url: str, session_dir: Path = Path("./data/sessions")):
        self.db_url = db_url
        self.session_dir = session_dir
        self._pool = None
        self._current_session: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._detection_batch: list = []
        self._episode_batch: list = []
        self._interpretation_batch: list = []
        self._button_presses: list = []
        self._batch_size = 30
        self._video_writer = None
        self._record_video = False
        self._frame_count = 0
        self._episode_count = 0
        self._state_time: dict = {}
        self._current_state = "idle"
        self._state_start = 0.0

    async def connect(self):
        """Connect to SQL database, or fall back to in-memory."""
        if not self.db_url or not self.db_url.startswith(("postgresql://", "postgres://")):
            if self.db_url and "sqlite" in self.db_url:
                logger.warning("⚠️ SQLite DSN detected — asyncpg requires PostgreSQL. Using in-memory buffer.")
            elif not self.db_url:
                logger.info("ℹ️ No database URL configured — using in-memory buffer only.")
            return

        if not PG_AVAILABLE:
            logger.warning("asyncpg not installed — session recording to memory only")
            return

        try:
            clean_url = self.db_url.replace("+asyncpg", "")
            self._pool = await asyncpg.create_pool(
                clean_url,
                min_size=2,
                max_size=5,
                command_timeout=60,
            )
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("✅ Session recorder connected to SQL database")
        except Exception as e:
            logger.warning(f"⚠️ DB connection failed: {e} — recording to memory only")
            self._pool = None

    async def start_session(self, record_video: bool = False) -> str:
        """Start recording a new session."""
        session_id = str(uuid.uuid4())
        self._current_session = session_id
        self._started_at = datetime.now(timezone.utc)
        self._frame_count = 0
        self._episode_count = 0
        self._state_time = {}
        self._current_state = "idle"
        self._state_start = time.time()
        self._button_presses = []
        self._detection_batch = []
        self._episode_batch = []
        self._interpretation_batch = []

        sess_dir = self.session_dir / session_id
        sess_dir.mkdir(parents=True, exist_ok=True)

        if self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO sessions (id, started_at) VALUES ($1, $2)",
                    uuid.UUID(session_id), self._started_at,
                )

        self._record_video = record_video
        if record_video and CV2_AVAILABLE:
            video_path = sess_dir / "session.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (1280, 720))
            logger.info(f"📹 Video recording to {video_path}")

        logger.info(f"📹 Session started: {session_id}")
        return session_id

    async def record_frame(self, detections: dict, frame: Optional[np.ndarray] = None):
        """Record one frame of detections (and optionally video)."""
        if not self._current_session:
            return

        self._frame_count += 1

        self._detection_batch.append({
            "time": datetime.now(timezone.utc),
            "session_id": self._current_session,
            "frame_idx": detections.get("frame_idx", 0),
            "pose": json.dumps(detections.get("pose_keypoints")) if detections.get("pose_keypoints") else None,
            "gaze": json.dumps({
                "head_yaw": detections.get("head_yaw", 0),
                "head_pitch": detections.get("head_pitch", 0),
                "fused_yaw": detections.get("fused_gaze_yaw", 0),
                "target": detections.get("gaze_target"),
            }),
            "face": json.dumps({
                "expression": detections.get("expression"),
                "calibrated": detections.get("calibrated_expression"),
                "confidence": detections.get("expression_confidence", 0),
                "mouth": detections.get("mouth_openness", 0),
                "smile": detections.get("smile_score", 0),
            }),
            "audio": json.dumps({
                "class": detections.get("vocalization_class"),
                "confidence": detections.get("vocalization_confidence", 0),
                "pitch": detections.get("pitch_hz", 0),
                "energy": detections.get("energy_db", 0),
                "is_vocal": detections.get("is_vocalization", False),
            }),
        })

        for ep in detections.get("new_episodes", []):
            self._episode_count += 1
            self._episode_batch.append({
                "time": datetime.now(timezone.utc),
                "session_id": self._current_session,
                "episode_type": ep.get("type", "unknown"),
                "duration_ms": ep.get("duration_ms", 0),
                "confidence": ep.get("confidence", 0),
                "features": json.dumps(ep.get("features", {})),
            })

        new_state = detections.get("child_state", "idle")
        if new_state != self._current_state:
            duration = time.time() - self._state_start
            self._state_time[self._current_state] = self._state_time.get(self._current_state, 0) + duration
            self._current_state = new_state
            self._state_start = time.time()

        if self._video_writer and frame is not None:
            self._video_writer.write(frame)

        if len(self._detection_batch) >= self._batch_size:
            await self._flush_batch()

    async def record_interpretation(self, interpretation: dict):
        """Record an interpretation."""
        if not self._current_session:
            return

        self._interpretation_batch.append({
            "time": datetime.now(timezone.utc),
            "session_id": self._current_session,
            "intent": interpretation.get("intent", ""),
            "target": interpretation.get("target"),
            "description": interpretation.get("description", ""),
            "confidence": interpretation.get("confidence", 0),
            "comm_level": interpretation.get("comm_level"),
            "evidence": json.dumps(interpretation.get("evidence", [])),
            "spoken": interpretation.get("should_speak", False),
        })

    def record_button_press(self, button_id: str, phrase: str):
        """Record a child-initiated button press."""
        self._button_presses.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "button_id": button_id,
            "phrase": phrase,
        })

    async def stop_session(self) -> SessionSummary:
        """Stop recording and generate session summary."""
        if not self._current_session:
            return SessionSummary(session_id="", started_at="", ended_at="", duration_minutes=0)

        await self._flush_batch()
        ended_at = datetime.now(timezone.utc)
        duration = time.time() - self._state_start
        self._state_time[self._current_state] = self._state_time.get(self._current_state, 0) + duration

        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None

        total_minutes = (ended_at - self._started_at).total_seconds() / 60
        btn_counts = Counter(bp["button_id"] for bp in self._button_presses)

        summary = SessionSummary(
            session_id=self._current_session,
            started_at=self._started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            duration_minutes=round(total_minutes, 1),
            total_frames=self._frame_count,
            avg_fps=round(self._frame_count / max(1, total_minutes * 60), 1),
            total_episodes=self._episode_count,
            state_durations={k: round(v, 1) for k, v in self._state_time.items()},
            most_common_state=max(self._state_time, key=self._state_time.get) if self._state_time else "idle",
            button_presses=len(self._button_presses),
            button_breakdown=dict(btn_counts),
        )

        sess_dir = self.session_dir / self._current_session
        sess_dir.mkdir(parents=True, exist_ok=True)
        summary_path = sess_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)

        if self._button_presses:
            bp_path = sess_dir / "button_presses.json"
            with open(bp_path, "w") as f:
                json.dump(self._button_presses, f, indent=2)

        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """UPDATE sessions SET ended_at=$1, total_frames=$2,
                           total_episodes=$3, total_interpretations=$4, notes=$5
                           WHERE id=$6""",
                        ended_at, self._frame_count, self._episode_count,
                        len(self._interpretation_batch),
                        json.dumps(summary.to_dict()),
                        uuid.UUID(self._current_session),
                    )
            except Exception as e:
                logger.error(f"DB session update failed: {e}")

        logger.info(f"📊 Session {self._current_session[:8]}… ended: "
                    f"{total_minutes:.1f} min, {self._frame_count} frames, "
                    f"{self._episode_count} episodes, {len(self._button_presses)} button presses")

        self._current_session = None
        return summary

    async def _flush_batch(self):
        """Flush accumulated detection batches to SQL database."""
        if not self._pool:
            self._detection_batch.clear()
            self._episode_batch.clear()
            self._interpretation_batch.clear()
            return

        try:
            async with self._pool.acquire() as conn:
                if self._detection_batch:
                    await conn.executemany(
                        """INSERT INTO detections (time, session_id, frame_idx, pose, gaze, face, audio)
                           VALUES ($1, $2::uuid, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb)""",
                        [(d["time"], d["session_id"], d["frame_idx"],
                          d["pose"], d["gaze"], d["face"], d["audio"])
                         for d in self._detection_batch]
                    )
                    self._detection_batch.clear()

                if self._episode_batch:
                    await conn.executemany(
                        """INSERT INTO episodes (time, session_id, episode_type, duration_ms, confidence, features)
                           VALUES ($1, $2::uuid, $3, $4, $5, $6::jsonb)""",
                        [(e["time"], e["session_id"], e["episode_type"],
                          e["duration_ms"], e["confidence"], e["features"])
                         for e in self._episode_batch]
                    )
                    self._episode_batch.clear()

                if self._interpretation_batch:
                    await conn.executemany(
                        """INSERT INTO interpretations
                           (time, session_id, intent, target, description, confidence, comm_level, evidence, spoken)
                           VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8::jsonb, $9)""",
                        [(i["time"], i["session_id"], i["intent"], i["target"],
                          i["description"], i["confidence"], i["comm_level"],
                          i["evidence"], i["spoken"])
                         for i in self._interpretation_batch]
                    )
                    self._interpretation_batch.clear()

        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            self._detection_batch.clear()
            self._episode_batch.clear()
            self._interpretation_batch.clear()


class SessionReplayEngine:
    """Replays a recorded session for therapist review."""

    def __init__(self, db_url: str, session_dir: Path = Path("/data/sessions")):
        self.db_url = db_url
        self.session_dir = session_dir

    async def get_session_list(self, limit: int = 50) -> list[dict]:
        """Get list of recorded sessions."""
        sessions = []
        if self.session_dir.exists():
            for sess_path in sorted(self.session_dir.iterdir(), reverse=True):
                summary_path = sess_path / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    sessions.append(summary)
                    if len(sessions) >= limit:
                        break
        return sessions

    async def get_session_detail(self, session_id: str) -> dict:
        """Get full session data for replay."""
        sess_dir = self.session_dir / session_id
        result = {
            "session_id": session_id,
            "summary": None,
            "has_video": False,
            "button_presses": [],
        }
        summary_path = sess_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                result["summary"] = json.load(f)
        video_path = sess_dir / "session.mp4"
        result["has_video"] = video_path.exists()
        bp_path = sess_dir / "button_presses.json"
        if bp_path.exists():
            with open(bp_path) as f:
                result["button_presses"] = json.load(f)
        return result

    async def get_session_episodes(self, session_id: str, limit: int = 500) -> list[dict]:
        """Get all episodes for a session from DB."""
        if not PG_AVAILABLE:
            return []
        try:
            pool = await asyncpg.create_pool(self.db_url.replace("+asyncpg", ""), min_size=1, max_size=2)
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT time, episode_type, duration_ms, confidence, features
                       FROM episodes WHERE session_id = $1::uuid
                       ORDER BY time ASC LIMIT $2""",
                    session_id, limit,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Episode query failed: {e}")
            return []

    async def get_session_interpretations(self, session_id: str, limit: int = 200) -> list[dict]:
        """Get all interpretations for a session."""
        if not PG_AVAILABLE:
            return []
        try:
            pool = await asyncpg.create_pool(self.db_url.replace("+asyncpg", ""), min_size=1, max_size=2)
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT time, intent, target, description, confidence,
                              comm_level, spoken, caregiver_feedback
                       FROM interpretations WHERE session_id = $1::uuid
                       ORDER BY time ASC LIMIT $2""",
                    session_id, limit,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Interpretation query failed: {e}")
            return []

    async def export_csv(self, session_id: str, output_path: Path) -> Path:
        """Export session data as CSV for therapist review."""
        import csv
        episodes = await self.get_session_episodes(session_id)
        interpretations = await self.get_session_interpretations(session_id)
        detail = await self.get_session_detail(session_id)

        csv_path = output_path / f"session_{session_id[:8]}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CueCatcher Session Report"])
            writer.writerow(["Session ID", session_id])
            if detail["summary"]:
                writer.writerow(["Duration", f"{detail['summary'].get('duration_minutes', 0)} minutes"])
                writer.writerow(["Total Episodes", detail["summary"].get("total_episodes", 0)])
                writer.writerow(["Button Presses", detail["summary"].get("button_presses", 0)])
            writer.writerow([])
            writer.writerow(["=== Episodes ==="])
            writer.writerow(["Time", "Type", "Duration (ms)", "Confidence", "Features"])
            for ep in episodes:
                writer.writerow([
                    ep.get("time", ""),
                    ep.get("episode_type", ""),
                    ep.get("duration_ms", 0),
                    round(ep.get("confidence", 0), 2),
                    ep.get("features", ""),
                ])
            writer.writerow([])
            writer.writerow(["=== Interpretations ==="])
            writer.writerow(["Time", "Intent", "Target", "Description", "Confidence", "Level", "Spoken", "Feedback"])
            for interp in interpretations:
                writer.writerow([
                    interp.get("time", ""),
                    interp.get("intent", ""),
                    interp.get("target", ""),
                    interp.get("description", ""),
                    round(interp.get("confidence", 0), 2),
                    interp.get("comm_level", ""),
                    interp.get("spoken", False),
                    interp.get("caregiver_feedback", ""),
                ])
            writer.writerow([])
            if detail["button_presses"]:
                writer.writerow(["=== Button Presses (Child-Initiated) ==="])
                writer.writerow(["Time", "Button", "Phrase"])
                for bp in detail["button_presses"]:
                    writer.writerow([bp.get("time", ""), bp.get("button_id", ""), bp.get("phrase", "")])
        logger.info(f"📄 CSV export: {csv_path}")
        return csv_path