"""
CueCatcher LLM Chat Interface
Real-time chat with local LLM (llama.cpp) about session data.
"""
import json
import sqlite3
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, AsyncGenerator, Dict, Any
from loguru import logger

# Database path matches server/recorder.py
DB_PATH = "data/compass.db"

def get_db_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

async def fetch_session_data(session_id: str) -> Dict[str, Any]:
    """Fetch session data directly from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    session_data = {
        "session_id": session_id,
        "frames": [],
        "episodes": [],
        "states": [],
        "summary": {}
    }
    
    try:
        # 1. Fetch Frames (Tier 1 Data)
        cursor.execute("""
            SELECT timestamp, pose_data, gaze_data, face_data, audio_data, action_data
            FROM frames 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        
        frames = []
        for row in cursor.fetchall():
            frames.append({
                "timestamp": row["timestamp"],
                "pose": json.loads(row["pose_data"]) if row["pose_data"] else {},
                "gaze": json.loads(row["gaze_data"]) if row["gaze_data"] else {},
                "face": json.loads(row["face_data"]) if row["face_data"] else {},
                "audio": json.loads(row["audio_data"]) if row["audio_data"] else {},
                "action": json.loads(row["action_data"]) if row["action_data"] else {},
            })
        session_data["frames"] = frames

        # 2. Fetch Episodes (Tier 2 Data - Behavioral Patterns)
        cursor.execute("""
            SELECT timestamp, episode_type, confidence, details
            FROM episodes 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        
        episodes = []
        for row in cursor.fetchall():
            episodes.append({
                "timestamp": row["timestamp"],
                "type": row["episode_type"],
                "confidence": row["confidence"],
                "details": json.loads(row["details"]) if row["details"] else {}
            })
        session_data["episodes"] = episodes

        # 3. Fetch States (Tier 3 Data - Child State)
        cursor.execute("""
            SELECT timestamp, state, confidence, interpretation
            FROM states 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        
        states = []
        for row in cursor.fetchall():
            states.append({
                "timestamp": row["timestamp"],
                "state": row["state"],
                "confidence": row["confidence"],
                "interpretation": row["interpretation"]
            })
        session_data["states"] = states

        # 4. Fetch Session Summary/Metadata
        cursor.execute("""
            SELECT start_time, end_time, duration, notes
            FROM sessions 
            WHERE id = ?
        """, (session_id,))
        
        summary_row = cursor.fetchone()
        if summary_row:
            session_data["summary"] = {
                "start_time": summary_row["start_time"],
                "end_time": summary_row["end_time"],
                "duration": summary_row["duration"],
                "notes": summary_row["notes"]
            }
            
    except Exception as e:
        logger.error(f"Error fetching session data from DB: {e}")
        raise
    finally:
        conn.close()
        
    return session_data


class LLMChatSession:
    """
    Chat interface for querying session data via LLM.
    
    This module:
    1. Connects to local llama.cpp backend (default: http://127.0.0.1:8080)
    2. Builds context from current/past session data
    3. Streams chat responses in real-time
    4. Maintains conversation history
    """
    
    def __init__(
        self, 
        llama_url: str = "http://127.0.0.1:8083",
        session_dir: Path = Path("./data/sessions")
    ):
        self.llama_url = llama_url.rstrip("/")
        self.session_dir = session_dir
        self.conversation_history = []
        self.current_session_data = None
        self._client = None
        self._accumulated_response = ""
    
    async def connect(self):
        """Test connection to llama.cpp backend."""
        try:
            self._client = httpx.AsyncClient(timeout=60.0)
            # Test endpoint - llama.cpp uses /health or just check root
            resp = await self._client.get(f"{self.llama_url}/")
            if resp.status_code == 200:
                logger.info(f"✅ Connected to llama.cpp at {self.llama_url}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Cannot connect to llama.cpp at {self.llama_url}: {e}")
            self._client = None
            return False
        return False
    
    def set_current_session(self, session_data: dict):
        """Set the current session data for context."""
        self.current_session_data = session_data
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the chat."""
        return """You are CueCatcher AI, a compassionate assistant helping caregivers understand non-verbal children's communication.

ROLE:
- You analyze behavioral data (gaze, reach, vocalizations, facial expressions)
- You explain what the child might be trying to communicate
- You use strengths-based language focusing on the child's competence
- You acknowledge uncertainty - interpretations are hypotheses, not facts
- You encourage caregivers and validate their observations

CONTEXT:
- Gaze alternation (looking between object and person) is a sophisticated social signal
- Coordinated signals (reach + gaze + vocal) indicate intentional communication
- The child may be minimally verbal or non-verbal
- Every communicative attempt matters and should be celebrated

GUIDELINES:
- Use warm, accessible language (avoid jargon)
- Explain technical terms when needed
- Suggest practical strategies to encourage communication
- Acknowledge when data is limited or ambiguous
- Never diagnose - you're observing patterns, not making clinical judgments"""

    def _build_context_message(self) -> str:
        """Build context from current session data."""
        if not self.current_session_data:
            return "No active session data available."
        
        ctx = []
        ctx.append(f"**Current Session**: {self.current_session_data.get('session_id', 'unknown')[:8]}...")
        
        summary = self.current_session_data.get('session_summary', {})
        ctx.append(f"- Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        ctx.append(f"- Episodes detected: {summary.get('total_episodes', 0)}")
        ctx.append(f"- Interpretations generated: {summary.get('total_interpretations', 0)}")
        
        highlights = self.current_session_data.get('communication_highlights', {})
        ctx.append(f"- Gaze alternations: {highlights.get('gaze_alternation_count', 0)}")
        ctx.append(f"- Coordinated signals: {highlights.get('coordinated_signals_count', 0)}")
        
        patterns = self.current_session_data.get('behavioral_patterns', {})
        if patterns:
            ctx.append("\n**Behavioral Patterns**:")
            most_common = patterns.get('most_common_behaviors', {})
            if most_common:
                ctx.append(f"- Most common behaviors: {', '.join(list(most_common.keys())[:5])}")
            combos = patterns.get('coordinated_signal_combos', {})
            if combos:
                ctx.append(f"- Signal combinations: {', '.join(list(combos.keys())[:3])}")
        
        return "\n".join(ctx)
    
    async def chat(self, user_message: str, stream: bool = True) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response.
        
        Args:
            user_message: User's question or comment
            stream: Whether to stream the response token-by-token
        
        Yields:
            Response tokens as they arrive
        """
        if not self._client:
            yield "⚠️ Not connected to LLM. Please check that llama.cpp is running at " + self.llama_url
            return
        
        # Build messages array
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"Context:\n{self._build_context_message()}\n\nUser question: {user_message}"}
        ]
        
        # Add conversation history (last 10 exchanges)
        for msg in self.conversation_history[-10:]:
            messages.insert(-1, msg)
        
        try:
            # llama.cpp completion endpoint (OpenAI compatible)
            payload = {
                "prompt": self._format_messages_for_llama(messages),
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9,
                "stream": stream,
                "stop": ["User:", "\n\n"],
            }
            
            if stream:
                async with self._client.stream("POST", f"{self.llama_url}/completion", json=payload) as resp:
                    if resp.status_code != 200:
                        yield f"Error: HTTP {resp.status_code}"
                        return
                    
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            content = data.get("content", "")
                            if content:
                                yield content
                                # Store in history
                                self._accumulated_response += content
            else:
                resp = await self._client.post(f"{self.llama_url}/completion", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    content = result.get("content", "")
                    yield content
                else:
                    yield f"Error: HTTP {resp.status_code}"
            
            # Save to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": self._accumulated_response})
            self._accumulated_response = ""
            
        except httpx.ConnectError:
            yield "⚠️ Cannot connect to llama.cpp. Is it running at " + self.llama_url + "?"
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _format_messages_for_llama(self, messages: list) -> str:
        """Format messages for llama.cpp prompt format."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n".join(formatted)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self._accumulated_response = ""


# Global instance for API use
_chat_instance: Optional[LLMChatSession] = None

async def get_llm_chat() -> LLMChatSession:
    """Get or create the global LLM chat instance."""
    global _chat_instance
    if _chat_instance is None:
        _chat_instance = LLMChatSession()
        await _chat_instance.connect()
    return _chat_instance
