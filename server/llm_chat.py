"""
LLM Chat Module for CueCatcher
Connects to local llama.cpp server and analyzes session data from SQLite or JSON files
"""
import sqlite3
import requests
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configuration
LLAMA_CPP_URL = "http://127.0.0.1:8083"
DB_PATH = Path(__file__).parent.parent / "data" / "compass.db"
SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"

def get_session_data_from_db(session_id: str) -> Dict[str, Any]:
    """Fetch real session data from SQLite database or JSON files as fallback"""
    
    # Try SQLite first
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT id, started_at, ended_at, duration as duration_seconds, notes
                FROM sessions 
                WHERE id = ?
            """, (session_id,))
            session_row = cursor.fetchone()
            
            if session_row:
                result = _build_session_data_from_cursor(session_row, cursor, session_id)
                conn.close()
                return result
            
            conn.close()
        except Exception as e:
            print(f"SQLite error: {e}")
    
    # Fallback to JSON files in sessions directory
    return _load_session_from_json(session_id)

def _load_session_from_json(session_id: str) -> Dict[str, Any]:
    """Load session data from JSON summary file as fallback"""
    summary_path = SESSIONS_DIR / session_id / "summary.json"
    
    if not summary_path.exists():
        return {"error": f"Session {session_id} not found in database or filesystem"}
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        session_data = {
            "session_id": summary.get("session_id", session_id),
            "child_name": "Unknown",
            "started_at": summary.get("started_at", ""),
            "ended_at": summary.get("ended_at", ""),
            "duration_seconds": summary.get("duration_minutes", 0) * 60,
            "frames": [],
            "episodes": [],
            "states": [],
            "gaze_analysis": [],
            "emotions": [],
            "audio_events": [],
            "summary": {
                "total_frames": summary.get("total_frames", 0),
                "total_episodes": summary.get("total_episodes", 0),
                "gaze_alternations": summary.get("gaze_alternation_count", 0),
                "positive_emotions": 0,
                "vocalizations": int(summary.get("vocalization_seconds", 0)),
                "communicative_episodes": 0,
                "state_durations": summary.get("state_durations", {}),
                "most_common_state": summary.get("most_common_state", "")
            }
        }
        
        # Load episodes if available
        episodes_path = SESSIONS_DIR / session_id / "episodes.json"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                episodes_data = json.load(f)
                if isinstance(episodes_data, list):
                    session_data["episodes"] = episodes_data[:100]
        
        # Load states if available
        states_path = SESSIONS_DIR / session_id / "states.json"
        if states_path.exists():
            with open(states_path, 'r') as f:
                states_data = json.load(f)
                if isinstance(states_data, list):
                    session_data["states"] = states_data[:100]
        
        return session_data
        
    except Exception as e:
        return {"error": f"Error loading session from JSON: {str(e)}"}

def _build_session_data_from_cursor(session_row, cursor, session_id: str) -> Dict[str, Any]:
    """Build session data from SQLite cursor results using detections table"""
    session_data = {
        "session_id": session_row["id"],
        "child_name": session_row["notes"] or "Unknown",
        "started_at": session_row["started_at"],
        "ended_at": session_row["ended_at"],
        "duration_seconds": session_row["duration_seconds"],
        "frames": [],
        "episodes": [],
        "states": [],
        "gaze_analysis": [],
        "emotions": [],
        "audio_events": [],
        "summary": {}
    }
    
    try:
        # Get detections from the detections table
        cursor.execute("""
            SELECT id, frame_idx, timestamp, person_confidence, pose_keypoints_json, num_keypoints,
                   head_yaw, head_pitch, head_roll, gaze_target, looking_at_camera,
                   face_detected, expression, expression_confidence, mouth_openness, smile_score,
                   is_vocalization, vocalization_class, vocalization_confidence,
                   pitch_hz, energy_db, child_state, child_state_confidence,
                   actions_json, episodes_json
            FROM detections 
            WHERE session_id = ?
            ORDER BY timestamp
            LIMIT 500
        """, (session_id,))
        
        frames = []
        gaze_targets = []
        emotions = []
        audio_events = []
        states = []
        episodes = []
        
        for row in cursor.fetchall():
            # Parse JSON fields
            pose_keypoints = json.loads(row["pose_keypoints_json"]) if row["pose_keypoints_json"] else {}
            actions = json.loads(row["actions_json"]) if row["actions_json"] else {}
            eps = json.loads(row["episodes_json"]) if row["episodes_json"] else {}
            
            frame_info = {
                "frame_idx": row["frame_idx"],
                "timestamp": row["timestamp"],
                "person_confidence": row["person_confidence"],
                "head_yaw": row["head_yaw"],
                "head_pitch": row["head_pitch"],
                "head_roll": row["head_roll"],
                "gaze_target": row["gaze_target"],
                "looking_at_camera": row["looking_at_camera"],
                "face_detected": row["face_detected"],
                "expression": row["expression"],
                "expression_confidence": row["expression_confidence"],
                "mouth_openness": row["mouth_openness"],
                "smile_score": row["smile_score"],
                "is_vocalization": row["is_vocalization"],
                "vocalization_class": row["vocalization_class"],
                "vocalization_confidence": row["vocalization_confidence"],
                "pitch_hz": row["pitch_hz"],
                "energy_db": row["energy_db"],
                "child_state": row["child_state"],
                "child_state_confidence": row["child_state_confidence"],
                "pose_keypoints": pose_keypoints,
                "actions": actions,
                "episodes": eps
            }
            frames.append(frame_info)
            
            # Extract gaze target
            if row["gaze_target"]:
                gaze_targets.append({
                    "timestamp": row["timestamp"],
                    "target": row["gaze_target"]
                })
            
            # Extract emotion/expression
            if row["expression"]:
                emotions.append({
                    "timestamp": row["timestamp"],
                    "emotion": row["expression"],
                    "confidence": row["expression_confidence"]
                })
            
            # Extract vocalization/audio events
            if row["is_vocalization"] or row["vocalization_class"]:
                audio_events.append({
                    "timestamp": row["timestamp"],
                    "event": row["vocalization_class"] or "vocalization",
                    "confidence": row["vocalization_confidence"],
                    "pitch_hz": row["pitch_hz"],
                    "energy_db": row["energy_db"]
                })
            
            # Extract child state
            if row["child_state"]:
                states.append({
                    "timestamp": row["timestamp"],
                    "state": row["child_state"],
                    "confidence": row["child_state_confidence"]
                })
            
            # Extract episodes from episodes_json
            if eps and isinstance(eps, list):
                for ep in eps:
                    if isinstance(ep, dict):
                        episodes.append({
                            "timestamp": row["timestamp"],
                            "type": ep.get("type", "unknown"),
                            "confidence": ep.get("confidence", 0),
                            "description": ep.get("description", "")
                        })
        
        session_data["frames"] = frames
        session_data["gaze_analysis"] = gaze_targets[:100]
        session_data["emotions"] = emotions[:100]
        session_data["audio_events"] = audio_events[:100]
        session_data["states"] = states[:100]
        session_data["episodes"] = episodes[:100]
        
        # Calculate summary statistics
        session_data["summary"] = {
            "total_frames": len(frames),
            "total_episodes": len(episodes),
            "gaze_alternations": len([g for g in gaze_targets if g["target"] == "alternating"]),
            "positive_emotions": len([e for e in emotions if e["emotion"].lower() in ["happy", "excited", "engaged", "smile", "joy"]]),
            "vocalizations": len([a for a in audio_events if a["event"]]),
            "communicative_episodes": len([e for e in episodes if e.get("confidence", 0) > 0.7])
        }
        
        return session_data
        
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}


def build_llm_prompt(session_data: Dict[str, Any], user_question: str) -> str:
    """Build a structured prompt for the LLM with session context"""
    
    if "error" in session_data:
        return f"Error loading session data: {session_data['error']}\n\nUser question: {user_question}"
    
    summary = session_data.get("summary", {})
    
    prompt = f"""You are CueCatcher AI, an expert assistant analyzing non-verbal communication patterns in children. You have access to real session data from a recent interaction.

SESSION OVERVIEW:
- Child: {session_data.get('child_name', 'Unknown')}
- Session ID: {session_data.get('session_id', 'N/A')}
- Duration: {session_data.get('duration_seconds', 0):.1f} seconds
- Total frames analyzed: {summary.get('total_frames', 0)}
- Behavioral episodes detected: {summary.get('total_episodes', 0)}

KEY METRICS:
- Gaze alternations (looking between object and person): {summary.get('gaze_alternations', 0)}
- Positive emotional expressions: {summary.get('positive_emotions', 0)}
- Vocalizations detected: {summary.get('vocalizations', 0)}
- High-confidence communicative acts: {summary.get('communicative_episodes', 0)}

RECENT GAZE PATTERNS (sample):
{json.dumps(session_data.get('gaze_analysis', [])[:10], indent=2)}

RECENT EMOTIONS (sample):
{json.dumps(session_data.get('emotions', [])[:10], indent=2)}

BEHAVIORAL EPISODES (highlights):
{json.dumps(session_data.get('episodes', [])[:10], indent=2)}

COMMUNICATIVE STATES (sample):
{json.dumps(session_data.get('states', [])[:10], indent=2)}

Based on this REAL data from the session, please answer the following question thoughtfully and provide actionable insights for parents and therapists:

USER QUESTION: {user_question}

Provide your response in a warm, supportive tone. Focus on the child's strengths and competencies. If you notice patterns suggesting intentional communication, highlight them. Be specific about timestamps or moments when important behaviors occurred."""

    return prompt

def chat_with_llm(session_id: str, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Send user message to llama.cpp with session context
    Returns streaming response or error
    """
    
    # Fetch real session data from SQLite
    session_data = get_session_data_from_db(session_id)
    
    # Build prompt with context
    prompt = build_llm_prompt(session_data, user_message)
    
    # Prepare request for llama.cpp
    # Using the completion endpoint compatible with llama-server
    payload = {
        "prompt": prompt,
        "n_predict": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "stream": False  # Set to True for streaming support
    }
    
    try:
        # Check if llama.cpp is available
        health_check = requests.get(f"{LLAMA_CPP_URL}/health", timeout=5)
        if health_check.status_code != 200:
            return {
                "success": False,
                "error": "llama.cpp server health check failed",
                "response": None
            }
        
        # Send completion request
        response = requests.post(
            f"{LLAMA_CPP_URL}/completion",
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result.get("content", "")
            
            return {
                "success": True,
                "response": llm_response,
                "session_data_loaded": True if "error" not in session_data else False,
                "metrics": session_data.get("summary", {}) if "error" not in session_data else None
            }
        else:
            return {
                "success": False,
                "error": f"llama.cpp returned status {response.status_code}",
                "details": response.text[:500] if response.text else "No details"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Cannot connect to llama.cpp at {LLAMA_CPP_URL}. Make sure it's running.",
            "hint": "Start llama.cpp with: ./server -m your-model.gguf --host 127.0.0.1 --port 8083"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The model may be processing a large context."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_available_sessions() -> List[Dict]:
    """Get list of all sessions from database or filesystem"""
    sessions = []
    
    # Try SQLite first
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, started_at, ended_at, duration as duration_seconds, notes
                FROM sessions
                ORDER BY started_at DESC
                LIMIT 50
            """)
            
            for row in cursor.fetchall():
                sessions.append({
                    "id": row["id"],
                    "child_name": row["notes"] or "Unknown",
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "duration_seconds": row["duration_seconds"]
                })
            
            conn.close()
            
            if sessions:
                return sessions
        except Exception as e:
            print(f"SQLite error: {e}")
    
    # Fallback to filesystem - scan sessions directory
    if SESSIONS_DIR.exists():
        try:
            for session_folder in SESSIONS_DIR.iterdir():
                if session_folder.is_dir():
                    summary_path = session_folder / "summary.json"
                    if summary_path.exists():
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                            sessions.append({
                                "id": session_folder.name,
                                "child_name": "Unknown",
                                "started_at": summary.get("started_at", ""),
                                "ended_at": summary.get("ended_at", ""),
                                "duration_seconds": summary.get("duration_minutes", 0) * 60
                            })
            
            # Sort by started_at descending
            sessions.sort(key=lambda x: x["started_at"], reverse=True)
            sessions = sessions[:50]  # Limit to 50
            
        except Exception as e:
            print(f"Filesystem scan error: {e}")
    
    return sessions

if __name__ == "__main__":
    # Test the module
    print("Testing LLM Chat Module...")
    
    # Check database
    if DB_PATH.exists():
        print(f"✅ Database found at {DB_PATH}")
        
        # Get sessions
        sessions = get_available_sessions()
        print(f"📋 Found {len(sessions)} sessions")
        
        if sessions:
            test_session = sessions[0]
            print(f"\n🧪 Testing with session: {test_session['id']}")
            
            # Fetch data
            data = get_session_data_from_db(test_session['id'])
            if "error" in data:
                print(f"❌ Error: {data['error']}")
            else:
                print(f"✅ Loaded {data['summary']['total_frames']} frames")
                print(f"✅ Found {data['summary']['gaze_alternations']} gaze alternations")
                
                # Test LLM
                print("\n🤖 Sending to LLM...")
                result = chat_with_llm(
                    test_session['id'],
                    "What communication patterns do you see in this session?"
                )
                
                if result["success"]:
                    print(f"\n💬 LLM Response:\n{result['response'][:500]}...")
                else:
                    print(f"\n❌ LLM Error: {result['error']}")
    else:
        print(f"❌ Database not found at {DB_PATH}")

def get_llm_chat():
    """Factory function to provide LLM chat functionality"""
    return {
        "chat_with_llm": chat_with_llm,
        "get_session_data": get_session_data_from_db,
        "get_available_sessions": get_available_sessions
    }
