"""
CueCatcher REST API — Dashboard, History, and Therapist Export

Endpoints for the longitudinal communication dashboard:
  /api/dashboard/summary     — overall communication trends
  /api/dashboard/daily       — day-by-day breakdown
  /api/dashboard/weekly      — weekly trends
  /api/dashboard/patterns    — learned behavior patterns
  /api/sessions              — session list
  /api/sessions/{id}         — session detail
  /api/sessions/{id}/export  — CSV export for therapists
  /api/sessions/{id}/replay  — replay data stream
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

try:
    import asyncpg
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

from config.settings import settings


router = APIRouter(prefix="/api", tags=["dashboard"])

DB_URL = settings.db_url.replace("+asyncpg", "")
import sys as _sys
SESSION_DIR = Path("/data/sessions") if _sys.platform != "win32" else Path(__file__).parent.parent / "data" / "sessions"


# ── Helpers ────────────────────────────────────────────────────

async def _get_pool():
    if not PG_AVAILABLE:
        return None
    try:
        return await asyncpg.create_pool(DB_URL, min_size=1, max_size=3)
    except Exception:
        return None


# ── Dashboard ──────────────────────────────────────────────────

@router.get("/dashboard/summary")
async def dashboard_summary(days: int = Query(30, ge=1, le=365)):
    """
    Overall communication summary over the last N days.
    This is the core longitudinal view — shows trends invisible to real-time observation.
    """
    pool = await _get_pool()
    if not pool:
        return _fallback_summary()

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with pool.acquire() as conn:
            # Total sessions
            sessions = await conn.fetchval(
                "SELECT count(*) FROM sessions WHERE started_at > $1", cutoff
            )

            # Total episodes by type
            episode_rows = await conn.fetch(
                """SELECT episode_type, count(*) as cnt, avg(confidence) as avg_conf
                   FROM episodes WHERE time > $1
                   GROUP BY episode_type ORDER BY cnt DESC""",
                cutoff,
            )

            # Total interpretations by intent
            interp_rows = await conn.fetch(
                """SELECT intent, count(*) as cnt, avg(confidence) as avg_conf,
                          sum(case when caregiver_feedback='confirmed' then 1 else 0 end) as confirmed,
                          sum(case when caregiver_feedback='rejected' then 1 else 0 end) as rejected
                   FROM interpretations WHERE time > $1
                   GROUP BY intent ORDER BY cnt DESC""",
                cutoff,
            )

            # Highest comm level observed
            max_level = await conn.fetchval(
                "SELECT max(comm_level) FROM interpretations WHERE time > $1",
                cutoff,
            )

            # Gaze alternation episodes (highest communicative signal)
            gaze_alt = await conn.fetchval(
                """SELECT count(*) FROM episodes
                   WHERE time > $1 AND episode_type = 'gaze_alternation'""",
                cutoff,
            )

            # Communication trend (episodes per day, last 7 days vs previous 7)
            recent_eps = await conn.fetchval(
                "SELECT count(*) FROM episodes WHERE time > $1",
                datetime.now(timezone.utc) - timedelta(days=7),
            )
            prev_eps = await conn.fetchval(
                """SELECT count(*) FROM episodes
                   WHERE time > $1 AND time < $2""",
                datetime.now(timezone.utc) - timedelta(days=14),
                datetime.now(timezone.utc) - timedelta(days=7),
            )

        trend = "stable"
        if prev_eps and prev_eps > 0:
            ratio = (recent_eps or 0) / prev_eps
            if ratio > 1.2:
                trend = "increasing"
            elif ratio < 0.8:
                trend = "decreasing"

        return {
            "period_days": days,
            "total_sessions": sessions,
            "episodes_by_type": {r["episode_type"]: {
                "count": r["cnt"],
                "avg_confidence": round(float(r["avg_conf"] or 0), 2),
            } for r in episode_rows},
            "interpretations_by_intent": {r["intent"]: {
                "count": r["cnt"],
                "avg_confidence": round(float(r["avg_conf"] or 0), 2),
                "confirmed": r["confirmed"],
                "rejected": r["rejected"],
            } for r in interp_rows},
            "highest_comm_level": max_level or 1,
            "gaze_alternation_count": gaze_alt or 0,
            "communication_trend": trend,
            "recent_7d_episodes": recent_eps or 0,
            "previous_7d_episodes": prev_eps or 0,
        }
    except Exception as e:
        logger.error(f"Dashboard summary error: {e}")
        return _fallback_summary()
    finally:
        await pool.close()


@router.get("/dashboard/daily")
async def dashboard_daily(days: int = Query(14, ge=1, le=90)):
    """Day-by-day communication breakdown."""
    pool = await _get_pool()
    if not pool:
        return {"days": []}

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT date_trunc('day', time) as day,
                          count(*) as total_episodes,
                          count(distinct episode_type) as unique_types,
                          avg(confidence) as avg_confidence,
                          sum(case when episode_type = 'gaze_alternation' then 1 else 0 end) as gaze_alt,
                          sum(case when episode_type in ('reach', 'arms_up', 'hand_leading') then 1 else 0 end) as requests
                   FROM episodes WHERE time > $1
                   GROUP BY day ORDER BY day ASC""",
                cutoff,
            )

        return {
            "days": [{
                "date": r["day"].isoformat(),
                "total_episodes": r["total_episodes"],
                "unique_types": r["unique_types"],
                "avg_confidence": round(float(r["avg_confidence"] or 0), 2),
                "gaze_alternation": r["gaze_alt"],
                "request_episodes": r["requests"],
            } for r in rows]
        }
    except Exception as e:
        logger.error(f"Daily dashboard error: {e}")
        return {"days": []}
    finally:
        await pool.close()


@router.get("/dashboard/weekly")
async def dashboard_weekly(weeks: int = Query(12, ge=1, le=52)):
    """Weekly communication trends — best for tracking developmental progress."""
    pool = await _get_pool()
    if not pool:
        return {"weeks": []}

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT date_trunc('week', time) as week,
                          count(*) as total_episodes,
                          max(e.confidence) as peak_confidence,
                          count(distinct episode_type) as behavior_variety
                   FROM episodes e WHERE time > $1
                   GROUP BY week ORDER BY week ASC""",
                cutoff,
            )

            # Also get interpretation accuracy per week
            interp_rows = await conn.fetch(
                """SELECT date_trunc('week', time) as week,
                          count(*) as total,
                          sum(case when caregiver_feedback='confirmed' then 1 else 0 end) as confirmed,
                          avg(confidence) as avg_conf
                   FROM interpretations WHERE time > $1
                   GROUP BY week ORDER BY week ASC""",
                cutoff,
            )

        interp_map = {r["week"]: r for r in interp_rows}

        return {
            "weeks": [{
                "week": r["week"].isoformat(),
                "total_episodes": r["total_episodes"],
                "peak_confidence": round(float(r["peak_confidence"] or 0), 2),
                "behavior_variety": r["behavior_variety"],
                "interpretations": interp_map.get(r["week"], {}).get("total", 0),
                "confirmed": interp_map.get(r["week"], {}).get("confirmed", 0),
                "accuracy": round(
                    (interp_map.get(r["week"], {}).get("confirmed", 0) /
                     max(1, interp_map.get(r["week"], {}).get("total", 1))) * 100, 1
                ),
            } for r in rows]
        }
    except Exception as e:
        logger.error(f"Weekly dashboard error: {e}")
        return {"weeks": []}
    finally:
        await pool.close()


@router.get("/dashboard/patterns")
async def dashboard_patterns():
    """Most frequent communication patterns (from behavior dictionary)."""
    pool = await _get_pool()
    if not pool:
        return {"patterns": []}

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT pattern_name, description, confidence,
                          times_confirmed, times_rejected, first_seen, last_seen
                   FROM behavior_dictionary WHERE active = true
                   ORDER BY times_confirmed DESC LIMIT 20"""
            )

        return {
            "patterns": [{
                "name": r["pattern_name"],
                "description": r["description"],
                "confidence": round(float(r["confidence"]), 2),
                "confirmed": r["times_confirmed"],
                "rejected": r["times_rejected"],
                "accuracy": round(r["times_confirmed"] / max(1, r["times_confirmed"] + r["times_rejected"]) * 100, 1),
                "first_seen": r["first_seen"].isoformat() if r["first_seen"] else None,
                "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None,
            } for r in rows]
        }
    except Exception as e:
        logger.error(f"Patterns error: {e}")
        return {"patterns": []}
    finally:
        await pool.close()


@router.get("/dashboard/hourly")
async def dashboard_hourly(days: int = Query(7, ge=1, le=30)):
    """When does communication happen? Hourly heatmap data."""
    pool = await _get_pool()
    if not pool:
        return {"hours": {}}

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT extract(hour from time) as hour, count(*) as cnt
                   FROM episodes WHERE time > $1
                   GROUP BY hour ORDER BY hour ASC""",
                cutoff,
            )

        return {"hours": {int(r["hour"]): r["cnt"] for r in rows}}
    except Exception as e:
        return {"hours": {}}
    finally:
        await pool.close()


# ── Sessions ──────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(limit: int = Query(20, ge=1, le=100)):
    """List recorded sessions."""
    sessions = []

    if SESSION_DIR.exists():
        for sess_path in sorted(SESSION_DIR.iterdir(), reverse=True):
            summary_file = sess_path / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    sessions.append(json.load(f))
                if len(sessions) >= limit:
                    break

    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session detail with full summary."""
    sess_dir = SESSION_DIR / session_id
    if not sess_dir.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)

    result = {"session_id": session_id}

    summary_path = sess_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            result["summary"] = json.load(f)

    result["has_video"] = (sess_dir / "session.mp4").exists()

    bp_path = sess_dir / "button_presses.json"
    if bp_path.exists():
        with open(bp_path) as f:
            result["button_presses"] = json.load(f)

    return result


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = Query("csv")):
    """Export session data for therapist review."""
    from server.recorder import SessionReplayEngine

    replay = SessionReplayEngine(settings.db_url, SESSION_DIR)
    export_dir = SESSION_DIR / session_id / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        csv_path = await replay.export_csv(session_id, export_dir)
        return FileResponse(csv_path, filename=csv_path.name, media_type="text/csv")

    return JSONResponse({"error": f"Unsupported format: {format}"}, status_code=400)


@router.get("/sessions/{session_id}/episodes")
async def session_episodes(session_id: str, limit: int = Query(200)):
    """Get all episodes for a session (for replay timeline)."""
    from server.recorder import SessionReplayEngine

    replay = SessionReplayEngine(settings.db_url, SESSION_DIR)
    episodes = await replay.get_session_episodes(session_id, limit)
    return {"episodes": episodes}


@router.get("/sessions/{session_id}/interpretations")
async def session_interpretations(session_id: str, limit: int = Query(100)):
    """Get all interpretations for a session."""
    from server.recorder import SessionReplayEngine

    replay = SessionReplayEngine(settings.db_url, SESSION_DIR)
    interps = await replay.get_session_interpretations(session_id, limit)
    return {"interpretations": interps}


@router.get("/sessions/{session_id}/video")
async def session_video(session_id: str):
    """Stream recorded session video (if available)."""
    video_path = SESSION_DIR / session_id / "session.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "No video for this session"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4")


# ── Fallback ──────────────────────────────────────────────────

def _fallback_summary():
    """Build summary from on-disk session JSON files when DB is unavailable."""
    from datetime import datetime, timezone, timedelta

    summaries = []
    if SESSION_DIR.exists():
        for sess_path in SESSION_DIR.iterdir():
            sf = sess_path / "summary.json"
            if sf.exists():
                try:
                    with open(sf) as f:
                        summaries.append(json.load(f))
                except Exception:
                    pass

    if not summaries:
        return {
            "period_days": 30,
            "total_sessions": 0,
            "episodes_by_type": {},
            "interpretations_by_intent": {},
            "highest_comm_level": 1,
            "gaze_alternation_count": 0,
            "communication_trend": "unknown",
            "note": "No session data yet",
        }

    total_episodes = sum(s.get("total_episodes", 0) for s in summaries)
    total_buttons = sum(s.get("button_presses", 0) for s in summaries)

    # Aggregate episode types across sessions
    ep_types = {}
    for s in summaries:
        for etype, info in s.get("episodes_by_type", {}).items():
            cnt = info.get("count", 0) if isinstance(info, dict) else info
            ep_types[etype] = ep_types.get(etype, 0) + cnt

    return {
        "period_days": 30,
        "total_sessions": len(summaries),
        "total_episodes": total_episodes,
        "total_button_presses": total_buttons,
        "episodes_by_type": {k: {"count": v} for k, v in ep_types.items()},
        "interpretations_by_intent": {},
        "highest_comm_level": max((s.get("highest_comm_level_observed", 1) for s in summaries), default=1),
        "gaze_alternation_count": 0,
        "communication_trend": "unknown",
        "source": "disk",
    }
