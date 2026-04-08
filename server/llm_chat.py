"""
LLM Chat Module for CueCatcher
Connects to local llama.cpp server and analyzes session data from SQLite
"""
import sqlite3
import requests
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configuration
LLAMA_CPP_URL = "http://127.0.0.1:8083"
DB_PATH = Path(__file__).parent.parent / "data" / "compass.db"

def get_session_data_from_db(session_id: str) -> Dict[str, Any]:
    """Fetch real session data from SQLite database"""
    if not DB_PATH.exists():
        return {"error": f"Database not found at {DB_PATH}"}
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute("""
            SELECT id, child_name, started_at, ended_at, duration_seconds
            FROM sessions 
            WHERE id = ?
        """, (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            conn.close()
            return {"error": f"Session {session_id} not found"}
        
        session_data = {
            "session_id": session_row["id"],
            "child_name": session_row["child_name"],
            "started_at": session_row["started_at"],
            "ended_at": session_row["ended_at"],
            "duration_seconds": session_row["duration_seconds"],
            "frames": [],
            "episodes": [],
            "states": [],
            "gaze_analysis": [],
            "summary": {}
        }
        
        # Get frames with detections
        cursor.execute("""
            SELECT timestamp, frame_data, detections
            FROM frames 
            WHERE session_id = ?
            ORDER BY timestamp
            LIMIT 500
        """, (session_id,))
        
        frames = []
        gaze_targets = []
        emotions = []
        audio_events = []
        
        for row in cursor.fetchall():
            frame_info = {
                "timestamp": row["timestamp"],
                "detections": json.loads(row["detections"]) if row["detections"] else {}
            }
            frames.append(frame_info)
            
            # Extract specific signals
            detections = frame_info["detections"]
            
            # Gaze target
            if "gaze_target" in detections:
                gaze_targets.append({
                    "timestamp": row["timestamp"],
                    "target": detections["gaze_target"]
                })
            
            # Emotion
            if "emotion" in detections:
                emotions.append({
                    "timestamp": row["timestamp"],
                    "emotion": detections["emotion"]
                })
            
            # Audio events
            if "audio_event" in detections:
                audio_events.append({
                    "timestamp": row["timestamp"],
                    "event": detections["audio_event"]
                })
        
        session_data["frames"] = frames
        session_data["gaze_analysis"] = gaze_targets[:100]  # Limit for context
        session_data["emotions"] = emotions[:100]
        session_data["audio_events"] = audio_events[:100]
        
        # Get episodes
        cursor.execute("""
            SELECT timestamp, episode_type, confidence, description
            FROM episodes 
            WHERE session_id = ?
            ORDER BY timestamp
            LIMIT 100
        """, (session_id,))
        
        episodes = []
        for row in cursor.fetchall():
            episodes.append({
                "timestamp": row["timestamp"],
                "type": row["episode_type"],
                "confidence": row["confidence"],
                "description": row["description"]
            })
        
        session_data["episodes"] = episodes
        
        # Get states
        cursor.execute("""
            SELECT timestamp, state, confidence
            FROM states 
            WHERE session_id = ?
            ORDER BY timestamp
            LIMIT 100
        """, (session_id,))
        
        states = []
        for row in cursor.fetchall():
            states.append({
                "timestamp": row["timestamp"],
                "state": row["state"],
                "confidence": row["confidence"]
            })
        
        session_data["states"] = states
        
        # Calculate summary statistics
        session_data["summary"] = {
            "total_frames": len(frames),
            "total_episodes": len(episodes),
            "gaze_alternations": len([g for g in gaze_targets if g["target"] == "alternating"]),
            "positive_emotions": len([e for e in emotions if e["emotion"] in ["happy", "excited", "engaged"]]),
            "vocalizations": len([a for a in audio_events if a["event"] in ["vocalization", "laugh", "babble"]]),
            "communicative_episodes": len([e for e in episodes if e["confidence"] > 0.7])
        }
        
        conn.close()
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
    """Get list of all sessions from database"""
    if not DB_PATH.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, child_name, started_at, ended_at, duration_seconds
            FROM sessions
            ORDER BY started_at DESC
            LIMIT 50
        """)
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "id": row["id"],
                "child_name": row["child_name"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "duration_seconds": row["duration_seconds"]
            })
        
        conn.close()
        return sessions
        
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return []

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
