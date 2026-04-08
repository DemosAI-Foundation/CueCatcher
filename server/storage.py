"""
COMPASS SQLite Storage

Saves all session data to a local SQLite database file.
Zero setup — just works on Windows. The .db file can later be:
  - Queried with any SQL tool (DB Browser for SQLite, DBeaver, etc.)
  - Loaded into pandas for analysis
  - Fed to an LLM for natural language queries
  - Used to fine-tune or evaluate models

Database: data/compass.db (~50-100MB per hour of recording at 10fps)
"""

import sqlite3
import json
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Optional
from dataclasses import dataclass, field, asdict

from loguru import logger


DB_PATH = Path(__file__).parent.parent / "data" / "compass.db"


def _init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_frames INTEGER DEFAULT 0,
            total_episodes INTEGER DEFAULT 0,
            total_interpretations INTEGER DEFAULT 0,
            total_button_presses INTEGER DEFAULT 0,
            summary_json TEXT
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            -- Pose
            person_confidence REAL,
            pose_keypoints_json TEXT,
            num_keypoints INTEGER,
            -- Gaze
            head_yaw REAL,
            head_pitch REAL,
            head_roll REAL,
            gaze_target TEXT,
            looking_at_camera INTEGER,
            -- Face
            face_detected INTEGER,
            expression TEXT,
            expression_confidence REAL,
            mouth_openness REAL,
            smile_score REAL,
            -- Audio
            is_vocalization INTEGER,
            vocalization_class TEXT,
            vocalization_confidence REAL,
            pitch_hz REAL,
            energy_db REAL,
            -- State
            child_state TEXT,
            child_state_confidence REAL,
            -- Actions (JSON array)
            actions_json TEXT,
            -- Episodes (JSON array)
            episodes_json TEXT
        );

        CREATE TABLE IF NOT EXISTS interpretations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            frame_idx INTEGER,
            intent TEXT,
            target TEXT,
            description TEXT,
            spoken_text TEXT,
            confidence REAL,
            comm_level INTEGER,
            evidence_json TEXT,
            alternatives_json TEXT,
            should_speak INTEGER,
            caregiver_feedback TEXT
        );

        CREATE TABLE IF NOT EXISTS button_presses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            button_id TEXT,
            phrase TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            interpretation_id TEXT,
            timestamp REAL NOT NULL,
            action TEXT,
            correct_meaning TEXT
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id, frame_idx);
        CREATE INDEX IF NOT EXISTS idx_det_state ON detections(child_state, timestamp);
        CREATE INDEX IF NOT EXISTS idx_interp_session ON interpretations(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_interp_intent ON interpretations(intent, timestamp);
        CREATE INDEX IF NOT EXISTS idx_btn_session ON button_presses(session_id, timestamp);
    """)
    conn.commit()


class SQLiteStorage:
    """
    Thread-safe SQLite storage with batched writes.
    
    Detections are batched (default: every 30 frames) to avoid
    hammering the disk on every frame. Interpretations and button
    presses are written immediately since they're infrequent.
    """

    def __init__(self, db_path: Path = DB_PATH, batch_size: int = 30):
        self.db_path = db_path
        self.batch_size = batch_size
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._det_batch: list = []
        self._current_session: Optional[str] = None
        self._frame_count = 0
        self._episode_count = 0
        self._interp_count = 0
        self._btn_count = 0

    def connect(self):
        """Open database connection and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")      # faster concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")     # good balance of speed/safety
        self._conn.execute("PRAGMA cache_size=-64000")      # 64MB cache
        _init_db(self._conn)
        logger.info(f"✅ SQLite database: {self.db_path}")

    def start_session(self) -> str:
        """Start a new session. Returns session_id."""
        session_id = str(uuid.uuid4())
        self._current_session = session_id
        self._frame_count = 0
        self._episode_count = 0
        self._interp_count = 0
        self._btn_count = 0
        self._det_batch = []

        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                (session_id, datetime.now(timezone.utc).isoformat())
            )
            self._conn.commit()

        return session_id

    def stop_session(self) -> dict:
        """Stop current session, flush remaining data, return summary."""
        if not self._current_session:
            return {}

        self._flush_detections()

        summary = {
            "session_id": self._current_session,
            "total_frames": self._frame_count,
            "total_episodes": self._episode_count,
            "total_interpretations": self._interp_count,
            "total_button_presses": self._btn_count,
        }

        with self._lock:
            self._conn.execute(
                """UPDATE sessions SET ended_at=?, total_frames=?, total_episodes=?,
                   total_interpretations=?, total_button_presses=?, summary_json=?
                   WHERE id=?""",
                (datetime.now(timezone.utc).isoformat(),
                 self._frame_count, self._episode_count,
                 self._interp_count, self._btn_count,
                 json.dumps(summary),
                 self._current_session)
            )
            self._conn.commit()

        sid = self._current_session
        self._current_session = None
        logger.info(f"📊 Session {sid[:8]}… saved: {self._frame_count} frames, "
                    f"{self._episode_count} episodes, {self._interp_count} interpretations")
        return summary

    def save_detection(self, detections: dict):
        """Queue a frame's detections for batch insert."""
        if not self._current_session:
            return

        self._frame_count += 1

        # Count episodes
        episodes = detections.get("new_episodes", [])
        self._episode_count += len(episodes)

        self._det_batch.append({
            "session_id": self._current_session,
            "frame_idx": detections.get("frame_idx", self._frame_count),
            "timestamp": detections.get("timestamp", time.time()),
            "person_confidence": detections.get("person_confidence", 0),
            "pose_keypoints_json": json.dumps(detections.get("pose_keypoints")) if detections.get("pose_keypoints") else None,
            "num_keypoints": detections.get("num_keypoints", 0),
            "head_yaw": detections.get("head_yaw", 0),
            "head_pitch": detections.get("head_pitch", 0),
            "head_roll": detections.get("head_roll", 0),
            "gaze_target": detections.get("gaze_target"),
            "looking_at_camera": 1 if detections.get("looking_at_camera") else 0,
            "face_detected": 1 if detections.get("face_detected") else 0,
            "expression": detections.get("calibrated_expression") or detections.get("expression"),
            "expression_confidence": detections.get("expression_confidence", 0),
            "mouth_openness": detections.get("mouth_openness", 0),
            "smile_score": detections.get("smile_score", 0),
            "is_vocalization": 1 if detections.get("is_vocalization") else 0,
            "vocalization_class": detections.get("vocalization_class"),
            "vocalization_confidence": detections.get("vocalization_confidence", 0),
            "pitch_hz": detections.get("pitch_hz", 0),
            "energy_db": detections.get("energy_db", -60),
            "child_state": detections.get("child_state"),
            "child_state_confidence": detections.get("child_state_confidence", 0),
            "actions_json": json.dumps(detections.get("actions_detected", [])) if detections.get("actions_detected") else None,
            "episodes_json": json.dumps(episodes) if episodes else None,
        })

        if len(self._det_batch) >= self.batch_size:
            self._flush_detections()

    def save_interpretation(self, interpretation: dict):
        """Save an interpretation immediately."""
        if not self._current_session:
            return

        self._interp_count += 1

        with self._lock:
            self._conn.execute(
                """INSERT INTO interpretations
                   (id, session_id, timestamp, intent, target, description,
                    spoken_text, confidence, comm_level, evidence_json,
                    alternatives_json, should_speak)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (interpretation.get("id", str(uuid.uuid4())),
                 self._current_session,
                 interpretation.get("timestamp", time.time()),
                 interpretation.get("intent"),
                 interpretation.get("target"),
                 interpretation.get("description"),
                 interpretation.get("spoken_text"),
                 interpretation.get("confidence", 0),
                 interpretation.get("comm_level"),
                 json.dumps(interpretation.get("evidence", [])),
                 json.dumps(interpretation.get("alternatives", [])),
                 1 if interpretation.get("should_speak") else 0)
            )
            self._conn.commit()

    def save_button_press(self, button_id: str, phrase: str):
        """Save a button press immediately."""
        if not self._current_session:
            return

        self._btn_count += 1

        with self._lock:
            self._conn.execute(
                "INSERT INTO button_presses (session_id, timestamp, button_id, phrase) VALUES (?, ?, ?, ?)",
                (self._current_session, time.time(), button_id, phrase)
            )
            self._conn.commit()

    def save_feedback(self, interpretation_id: str, action: str, correct_meaning: str = None):
        """Save caregiver feedback."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO feedback (session_id, interpretation_id, timestamp, action, correct_meaning)
                   VALUES (?, ?, ?, ?, ?)""",
                (self._current_session, interpretation_id, time.time(), action, correct_meaning)
            )
            self._conn.execute(
                "UPDATE interpretations SET caregiver_feedback=? WHERE id=?",
                (action, interpretation_id)
            )
            self._conn.commit()

    def _flush_detections(self):
        """Batch-insert queued detections."""
        if not self._det_batch:
            return

        batch = self._det_batch
        self._det_batch = []

        cols = list(batch[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        with self._lock:
            self._conn.executemany(
                f"INSERT INTO detections ({col_names}) VALUES ({placeholders})",
                [tuple(row[c] for c in cols) for row in batch]
            )
            self._conn.commit()

    # ── Query methods (for dashboard and analysis) ──

    def get_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_session_episodes(self, session_id: str) -> list[dict]:
        """Get all episodes for a session (from detections with non-null episodes)."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT frame_idx, timestamp, episodes_json 
                   FROM detections 
                   WHERE session_id=? AND episodes_json IS NOT NULL
                   ORDER BY frame_idx""",
                (session_id,)
            )
            results = []
            for row in cur.fetchall():
                episodes = json.loads(row[2]) if row[2] else []
                for ep in episodes:
                    ep["frame_idx"] = row[0]
                    ep["timestamp"] = row[1]
                    results.append(ep)
            return results

    def get_session_interpretations(self, session_id: str) -> list[dict]:
        """Get all interpretations for a session."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM interpretations WHERE session_id=? ORDER BY timestamp",
                (session_id,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_session_detections(self, session_id: str, limit: int = 10000) -> list[dict]:
        """Get raw detections for analysis. Returns lightweight rows (no keypoints)."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT frame_idx, timestamp, person_confidence,
                   head_yaw, head_pitch, gaze_target, looking_at_camera,
                   face_detected, expression, expression_confidence,
                   mouth_openness, smile_score,
                   is_vocalization, vocalization_class, pitch_hz, energy_db,
                   child_state, child_state_confidence, actions_json
                   FROM detections WHERE session_id=?
                   ORDER BY frame_idx LIMIT ?""",
                (session_id, limit)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_summary_stats(self, days: int = 30) -> dict:
        """Dashboard summary across recent sessions."""
        import time as _time
        cutoff = _time.time() - days * 86400

        with self._lock:
            # Total sessions
            sessions = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE started_at > datetime(?, 'unixepoch')",
                (cutoff,)
            ).fetchone()[0]

            # Total interpretations by intent
            intents = self._conn.execute(
                """SELECT intent, COUNT(*) as cnt, AVG(confidence) as avg_conf,
                   SUM(CASE WHEN caregiver_feedback='confirmed' THEN 1 ELSE 0 END) as confirmed,
                   SUM(CASE WHEN caregiver_feedback='rejected' THEN 1 ELSE 0 END) as rejected
                   FROM interpretations WHERE timestamp > ?
                   GROUP BY intent""",
                (cutoff,)
            ).fetchall()

            # State distribution
            states = self._conn.execute(
                """SELECT child_state, COUNT(*) FROM detections 
                   WHERE timestamp > ? AND child_state IS NOT NULL
                   GROUP BY child_state""",
                (cutoff,)
            ).fetchall()

            # Button usage
            buttons = self._conn.execute(
                """SELECT button_id, COUNT(*), phrase FROM button_presses
                   WHERE timestamp > ? GROUP BY button_id ORDER BY COUNT(*) DESC""",
                (cutoff,)
            ).fetchall()

        return {
            "period_days": days,
            "total_sessions": sessions,
            "interpretations_by_intent": {
                row[0]: {"count": row[1], "avg_confidence": round(row[2] or 0, 2),
                         "confirmed": row[3], "rejected": row[4]}
                for row in intents
            },
            "state_distribution": {row[0]: row[1] for row in states},
            "button_usage": [
                {"button_id": row[0], "count": row[1], "phrase": row[2]}
                for row in buttons
            ],
        }

    def export_session_csv(self, session_id: str, output_path: Path) -> Path:
        """Export a session to CSV for therapist review or LLM analysis."""
        import csv

        csv_path = output_path / f"session_{session_id[:8]}.csv"

        detections = self.get_session_detections(session_id)
        interpretations = self.get_session_interpretations(session_id)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["=== COMPASS Session Export ==="])
            writer.writerow(["Session", session_id])
            writer.writerow(["Frames", len(detections)])
            writer.writerow(["Interpretations", len(interpretations)])
            writer.writerow([])

            if detections:
                writer.writerow(["=== Detections ==="])
                writer.writerow(list(detections[0].keys()))
                for d in detections:
                    writer.writerow(list(d.values()))
                writer.writerow([])

            if interpretations:
                writer.writerow(["=== Interpretations ==="])
                writer.writerow(list(interpretations[0].keys()))
                for i in interpretations:
                    writer.writerow(list(i.values()))

        return csv_path

    def close(self):
        """Flush and close."""
        self._flush_detections()
        if self._conn:
            self._conn.close()
