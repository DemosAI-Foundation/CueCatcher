"""
CueCatcher LLM Session Analyzer
Generates natural language insights from session and longitudinal data.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    import asyncpg
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False


class LLMSessionAnalyzer:
    """
    Analyzes session data and generates LLM-powered insights.
    
    This module:
    1. Extracts short-horizon data (per-frame detections, episodes, interpretations)
    2. Aggregates long-horizon data (session summaries, trends, patterns)
    3. Constructs structured prompts for LLM analysis
    4. Generates natural language reports for caregivers and therapists
    """
    
    def __init__(self, db_url: str, session_dir: Path = Path("./data/sessions")):
        self.db_url = db_url.replace("+asyncpg", "") if db_url else ""
        self.session_dir = session_dir
        self._pool = None
        
    async def connect(self):
        """Connect to PostgreSQL database if available."""
        if not self.db_url or not PG_AVAILABLE:
            logger.info("ℹ️ No database — LLM analysis will use JSON files only")
            return
            
        try:
            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=3,
                command_timeout=60,
            )
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("✅ LLM analyzer connected to database")
        except Exception as e:
            logger.warning(f"⚠️ DB connection failed: {e} — using JSON files only")
            self._pool = None
    
    async def analyze_session(self, session_id: str, include_video_timestamps: bool = False) -> dict:
        """
        Analyze a single session and generate LLM-ready insights.
        
        Returns:
            dict with:
            - session_summary: basic stats
            - key_moments: high-confidence interpretations and episodes
            - communication_highlights: gaze alternations, coordinated signals
            - behavioral_patterns: repeated behaviors
            - caregiver_feedback_summary: confirmed/rejected interpretations
            - llm_prompt: pre-formatted prompt for LLM analysis
        """
        sess_dir = self.session_dir / session_id
        if not sess_dir.exists():
            return {"error": "Session not found"}
        
        # Load session summary
        summary_path = sess_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        
        # Load button presses
        button_presses = []
        bp_path = sess_dir / "button_presses.json"
        if bp_path.exists():
            with open(bp_path) as f:
                button_presses = json.load(f)
        
        # Get episodes and interpretations from DB or fallback
        episodes = await self._get_session_episodes(session_id)
        interpretations = await self._get_session_interpretations(session_id)
        
        # Identify key moments (high confidence, gaze alternation, coordinated signals)
        key_moments = []
        for ep in episodes[:50]:  # Limit to first 50 for LLM context
            if ep.get("confidence", 0) >= 0.7 or ep.get("episode_type") == "gaze_alternation":
                key_moments.append({
                    "time_offset": ep.get("time"),
                    "type": "episode",
                    "episode_type": ep.get("episode_type"),
                    "confidence": ep.get("confidence"),
                    "features": ep.get("features", {}),
                })
        
        for interp in interpretations[:30]:  # Limit interpretations
            if interp.get("confidence", 0) >= 0.7 or interp.get("comm_level", 0) >= 4:
                key_moments.append({
                    "time_offset": interp.get("time"),
                    "type": "interpretation",
                    "intent": interp.get("intent"),
                    "target": interp.get("target"),
                    "description": interp.get("description"),
                    "confidence": interp.get("confidence"),
                    "comm_level": interp.get("comm_level"),
                    "evidence": interp.get("evidence", []),
                })
        
        # Sort by time
        key_moments.sort(key=lambda x: x.get("time_offset", ""))
        
        # Calculate communication highlights
        gaze_alt_count = sum(1 for ep in episodes if ep.get("episode_type") == "gaze_alternation")
        coordinated_signals = sum(1 for ep in episodes 
                                  if ep.get("features", {}).get("coordination_score", 0) >= 0.6)
        
        # Build LLM prompt
        llm_prompt = self._build_session_prompt(
            summary=summary,
            button_presses=button_presses,
            key_moments=key_moments[:20],  # Top 20 moments
            gaze_alternation_count=gaze_alt_count,
            coordinated_signals_count=coordinated_signals,
        )
        
        return {
            "session_id": session_id,
            "session_summary": summary,
            "key_moments": key_moments[:20],
            "communication_highlights": {
                "gaze_alternation_count": gaze_alt_count,
                "coordinated_signals_count": coordinated_signals,
                "total_episodes": len(episodes),
                "total_interpretations": len(interpretations),
            },
            "behavioral_patterns": self._extract_patterns(episodes),
            "caregiver_feedback_summary": self._summarize_feedback(interpretations),
            "llm_prompt": llm_prompt,
            "video_timestamps": self._get_video_timestamps(key_moments) if include_video_timestamps else [],
        }
    
    async def analyze_longitudinal(self, days: int = 30) -> dict:
        """
        Analyze multiple sessions over time for longitudinal insights.
        
        Returns:
            dict with:
            - period_summary: overall stats for the period
            - trends: communication trends (increasing/decreasing/stable)
            - developmental_milestones: highest comm levels achieved
            - recurring_patterns: behaviors seen across multiple sessions
            - llm_prompt: pre-formatted prompt for longitudinal analysis
        """
        from datetime import timedelta
        
        # Get all sessions in period
        sessions = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        if self.session_dir.exists():
            for sess_path in sorted(self.session_dir.iterdir(), reverse=True):
                summary_path = sess_path / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        sess_summary = json.load(f)
                    try:
                        started_at = datetime.fromisoformat(sess_summary.get("started_at", ""))
                        if started_at.tzinfo is None:
                            started_at = started_at.replace(tzinfo=timezone.utc)
                        if started_at >= cutoff:
                            sessions.append(sess_summary)
                    except Exception:
                        pass
        
        if not sessions:
            return {
                "period_days": days,
                "total_sessions": 0,
                "note": "No sessions found in this period",
                "llm_prompt": "No data available for analysis.",
            }
        
        # Aggregate statistics
        total_episodes = sum(s.get("total_episodes", 0) for s in sessions)
        total_button_presses = sum(s.get("button_presses", 0) for s in sessions)
        total_gaze_alt = sum(s.get("gaze_alternation_count", 0) for s in sessions)
        total_coordinated = sum(s.get("coordinated_signals_count", 0) for s in sessions)
        
        # Episode type breakdown across all sessions
        episode_types = {}
        for s in sessions:
            for etype, info in s.get("episodes_by_type", {}).items():
                cnt = info.get("count", 0) if isinstance(info, dict) else info
                episode_types[etype] = episode_types.get(etype, 0) + cnt
        
        # State durations
        state_totals = {}
        for s in sessions:
            for state, duration in s.get("state_durations", {}).items():
                state_totals[state] = state_totals.get(state, 0) + duration
        
        # Highest comm level observed
        max_comm_level = max((s.get("highest_comm_level_observed", 1) for s in sessions), default=1)
        
        # Trend analysis (compare first half vs second half of period)
        mid_point = len(sessions) // 2
        first_half_eps = sum(s.get("total_episodes", 0) for s in sessions[mid_point:])
        second_half_eps = sum(s.get("total_episodes", 0) for s in sessions[:mid_point])
        
        trend = "stable"
        if first_half_eps > 0:
            ratio = second_half_eps / first_half_eps
            if ratio > 1.2:
                trend = "increasing"
            elif ratio < 0.8:
                trend = "decreasing"
        
        # Build LLM prompt
        llm_prompt = self._build_longitudinal_prompt(
            period_days=days,
            total_sessions=len(sessions),
            total_episodes=total_episodes,
            episode_types=episode_types,
            state_durations=state_totals,
            gaze_alternation_total=total_gaze_alt,
            coordinated_signals_total=total_coordinated,
            max_comm_level=max_comm_level,
            trend=trend,
            button_presses=total_button_presses,
        )
        
        return {
            "period_days": days,
            "total_sessions": len(sessions),
            "total_episodes": total_episodes,
            "total_button_presses": total_button_presses,
            "episode_types": episode_types,
            "state_durations": state_totals,
            "gaze_alternation_total": total_gaze_alt,
            "coordinated_signals_total": total_coordinated,
            "max_comm_level_observed": max_comm_level,
            "trend": trend,
            "recurring_patterns": self._extract_longitudinal_patterns(sessions),
            "llm_prompt": llm_prompt,
        }
    
    def _build_session_prompt(self, summary: dict, button_presses: list, 
                             key_moments: list, gaze_alternation_count: int,
                             coordinated_signals_count: int) -> str:
        """Build a structured prompt for LLM session analysis."""
        
        duration = summary.get("duration_minutes", 0)
        total_episodes = summary.get("total_episodes", 0)
        total_interps = summary.get("total_interpretations", 0)
        
        prompt = f"""You are analyzing a CueCatcher session with a non-verbal or minimally verbal child.

SESSION OVERVIEW:
- Duration: {duration:.1f} minutes
- Total behavioral episodes detected: {total_episodes}
- Total AI interpretations generated: {total_interps}
- Child-initiated button presses: {len(button_presses)}

KEY COMMUNICATIVE SIGNALS:
- Gaze alternation events (looking between object and person): {gaze_alternation_count}
- Coordinated multi-signal behaviors (reach + gaze + vocal): {coordinated_signals_count}

CHILD'S BUTTON PRESSES:
{json.dumps(button_presses[:10], indent=2) if button_presses else "None recorded"}

HIGH-CONFIDENCE MOMENTS (top examples):
{json.dumps(key_moments[:10], indent=2)}

TASK:
Generate a compassionate, strengths-based summary for the caregiver that:
1. Highlights the child's communicative attempts and successes
2. Explains what gaze alternation means (it's a sophisticated social signal!)
3. Notes any patterns in behavior or communication
4. Suggests 1-2 specific strategies to encourage more communication
5. Uses accessible language (avoid jargon like "episode" or "interpretation")

Keep the tone warm, encouraging, and focused on the child's agency and competence."""
        
        return prompt
    
    def _build_longitudinal_prompt(self, period_days: int, total_sessions: int,
                                   total_episodes: int, episode_types: dict,
                                   state_durations: dict, gaze_alternation_total: int,
                                   coordinated_signals_total: int, max_comm_level: int,
                                   trend: str, button_presses: int) -> str:
        """Build a structured prompt for LLM longitudinal analysis."""
        
        prompt = f"""You are analyzing {total_sessions} CueCatcher sessions over {period_days} days for a non-verbal or minimally verbal child.

LONGITUDINAL OVERVIEW:
- Total sessions: {total_sessions}
- Total behavioral episodes: {total_episodes}
- Total child-initiated button presses: {button_presses}
- Overall trend: {trend} (compared to earlier period)

COMMUNICATIVE BEHAVIORS ACROSS ALL SESSIONS:
{json.dumps(episode_types, indent=2)}

TIME SPENT IN EACH STATE:
{json.dumps(state_durations, indent=2)}

ADVANCED SOCIAL SIGNALS:
- Total gaze alternation events: {gaze_alternation_total}
- Total coordinated multi-signal behaviors: {coordinated_signals_total}

HIGHEST COMMUNICATION LEVEL OBSERVED: Level {max_comm_level}
(Level 1 = single signal, Level 5 = coordinated request with gaze+reach+vocal, Level 7 = symbolic/ AAC use)

TASK:
Generate an insightful longitudinal report for parents and therapists that:
1. Celebrates growth and progress (even small gains!)
2. Explains the significance of gaze alternation and coordination
3. Identifies patterns: When does communication happen most? What types are increasing?
4. Notes the child's preferred communication methods
5. Suggests 2-3 evidence-based strategies for the next month
6. Flags any concerns if communication is decreasing or limited

Use warm, hopeful language. Emphasize that every communicative attempt matters."""
        
        return prompt
    
    def _extract_patterns(self, episodes: list) -> dict:
        """Extract recurring behavioral patterns from episodes."""
        from collections import Counter
        
        type_counts = Counter(ep.get("episode_type") for ep in episodes if ep.get("episode_type"))
        
        # Find feature patterns
        feature_patterns = []
        for ep in episodes:
            features = ep.get("features", {})
            if features.get("coordination_score", 0) >= 0.7:
                signals = []
                if features.get("has_reach"):
                    signals.append("reach")
                if features.get("has_gaze_alternation"):
                    signals.append("gaze_alt")
                if features.get("has_vocalization"):
                    signals.append("vocal")
                if len(signals) >= 2:
                    feature_patterns.append("+".join(signals))
        
        pattern_counts = Counter(feature_patterns)
        
        return {
            "most_common_behaviors": dict(type_counts.most_common(5)),
            "coordinated_signal_combos": dict(pattern_counts.most_common(5)),
        }
    
    def _extract_longitudinal_patterns(self, sessions: list) -> dict:
        """Extract patterns across multiple sessions."""
        # Track which behaviors appear in multiple sessions
        behavior_sessions = {}
        for sess in sessions:
            for etype in sess.get("episodes_by_type", {}).keys():
                if etype not in behavior_sessions:
                    behavior_sessions[etype] = 0
                behavior_sessions[etype] += 1
        
        recurring = {k: v for k, v in behavior_sessions.items() if v >= 2}
        
        return {
            "behaviors_seen_in_multiple_sessions": recurring,
            "sessions_with_gaze_alternation": sum(1 for s in sessions if s.get("gaze_alternation_count", 0) > 0),
            "average_episodes_per_session": round(sum(s.get("total_episodes", 0) for s in sessions) / max(1, len(sessions)), 1),
        }
    
    def _summarize_feedback(self, interpretations: list) -> dict:
        """Summarize caregiver feedback on interpretations."""
        confirmed = sum(1 for i in interpretations if i.get("spoken") and i.get("confidence", 0) >= 0.7)
        # In future: track explicit caregiver confirm/reject buttons
        
        return {
            "high_confidence_spoken": confirmed,
            "total_interpretations": len(interpretations),
            "estimated_accuracy": round(confirmed / max(1, len(interpretations)) * 100, 1),
        }
    
    def _get_video_timestamps(self, key_moments: list) -> list:
        """Extract video timestamps for key moments."""
        timestamps = []
        for moment in key_moments:
            time_str = moment.get("time_offset", "")
            if time_str:
                try:
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    timestamps.append({
                        "type": moment.get("type"),
                        "description": moment.get("episode_type") or moment.get("intent"),
                        "timestamp": dt.strftime("%H:%M:%S"),
                    })
                except Exception:
                    pass
        return timestamps[:10]  # Top 10 for video scrubbing
    
    async def _get_session_episodes(self, session_id: str, limit: int = 500) -> list:
        """Get episodes from DB or fallback to JSON."""
        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """SELECT time, episode_type, duration_ms, confidence, features
                           FROM episodes WHERE session_id = $1::uuid
                           ORDER BY time ASC LIMIT $2""",
                        session_id, limit,
                    )
                    return [dict(row) for row in rows]
            except Exception as e:
                logger.warning(f"DB episode fetch failed: {e}")
        
        # Fallback: read from session summary (aggregated only)
        sess_dir = self.session_dir / session_id
        summary_path = sess_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            # Return aggregated counts since we don't have per-episode data in JSON
            return [{"episode_type": k, "count": v.get("count", v) if isinstance(v, dict) else v}
                    for k, v in summary.get("episodes_by_type", {}).items()]
        return []
    
    async def _get_session_interpretations(self, session_id: str, limit: int = 200) -> list:
        """Get interpretations from DB or fallback."""
        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """SELECT time, intent, target, description, confidence, 
                                  comm_level, evidence, spoken
                           FROM interpretations WHERE session_id = $1::uuid
                           ORDER BY time ASC LIMIT $2""",
                        session_id, limit,
                    )
                    return [dict(row) for row in rows]
            except Exception as e:
                logger.warning(f"DB interpretation fetch failed: {e}")
        
        # Fallback: no detailed interpretation data in JSON
        return []
    
    async def close(self):
        """Close database connections."""
        if self._pool:
            await self._pool.close()


# Example usage for integration with LLM API
async def generate_llm_report(analyzer: LLMSessionAnalyzer, session_id: str, llm_client=None):
    """
    Generate a full LLM-powered session report.
    
    Args:
        analyzer: LLMSessionAnalyzer instance
        session_id: ID of session to analyze
        llm_client: Async LLM client (e.g., OpenAI, Anthropic, Ollama)
    
    Returns:
        dict with analysis and LLM-generated narrative
    """
    # Get structured analysis
    analysis = await analyzer.analyze_session(session_id, include_video_timestamps=True)
    
    if "error" in analysis:
        return analysis
    
    # Call LLM if client provided
    llm_narrative = None
    if llm_client:
        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are a compassionate child development specialist."},
                    {"role": "user", "content": analysis["llm_prompt"]},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            llm_narrative = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
    
    return {
        **analysis,
        "llm_narrative": llm_narrative,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
