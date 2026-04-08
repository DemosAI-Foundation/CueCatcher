"""CueCatcher server — receives video/audio from tablet, runs inference, sends back interpretations."""

import asyncio
import uuid
import time
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None
import orjson

from config.settings import settings


# ── Global State ───────────────────────────────────────────────
class AppState:
    """Mutable application state shared across the server."""
    def __init__(self):
        self.redis: aioredis.Redis | None = None
        self.pipeline = None           # InferencePipeline
        self.interpreter = None        # BehaviorInterpreter
        self.tts = None                # VoxtralTTS
        self.recorder = None           # SessionRecorder
        self.active_session_id: str | None = None
        self.connected_clients: set[WebSocket] = set()
        self.frame_count: int = 0
        self.is_running: bool = False

state = AppState()


# ── Lifecycle ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🧭 CueCatcher starting up...")

    # Connect Redis (optional — works without it)
    try:
        if aioredis:
            state.redis = aioredis.from_url(settings.redis_url, decode_responses=False)
            await state.redis.ping()
            logger.info("✅ Redis connected")
        else:
            state.redis = None
            logger.info("ℹ️  redis package not installed — using in-memory buffer")
    except Exception as e:
        state.redis = None
        logger.warning(f"⚠️  Redis not available ({e}) — using in-memory buffer")

    # Initialize inference pipeline (lazy GPU loading)
    try:
        from inference.pipeline import InferencePipeline
        state.pipeline = InferencePipeline(device=settings.device)
        await asyncio.get_event_loop().run_in_executor(None, state.pipeline.load_models)
        logger.info("✅ Inference pipeline loaded")
    except Exception as e:
        logger.warning(f"⚠️  Pipeline not loaded (run setup first): {e}")

    # Initialize interpreter
    try:
        from inference.interpreter import BehaviorInterpreter
        state.interpreter = BehaviorInterpreter()
        logger.info("✅ Behavior interpreter ready")
    except Exception as e:
        logger.warning(f"⚠️  Interpreter not loaded: {e}")

    # Initialize TTS
    if settings.tts_enabled:
        try:
            from voice.tts import VoxtralTTS
            state.tts = VoxtralTTS()
            await asyncio.get_event_loop().run_in_executor(None, state.tts.load)
            logger.info("✅ Voxtral TTS loaded")
        except Exception as e:
            logger.warning(f"⚠️  TTS not loaded: {e}")

    # ── Initialize session recorder ──
    try:
        from server.recorder import SessionRecorder
        import sys
        sess_dir = Path("/data/sessions") if sys.platform != "win32" else Path(__file__).parent.parent / "data" / "sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        state.recorder = SessionRecorder(settings.db_url, sess_dir)
        
        # Verify connect() exists before calling
        if hasattr(state.recorder, 'connect') and callable(state.recorder.connect):
            await state.recorder.connect()
            logger.info("✅ Session recorder ready")
        else:
            logger.warning("⚠️ SessionRecorder.connect() not found — using in-memory only")
            
    except ImportError as e:
        logger.warning(f"⚠️ Session recorder module not found: {e} — using in-memory only")
    except Exception as e:
        logger.warning(f"⚠️ Session recorder initialization failed: {e} — using in-memory only")
        state.recorder = None

    logger.info("🧭 CueCatcher ready — open tablet UI at http://<server-ip>:8084")
    yield

    # Shutdown
    logger.info("🧭 CueCatcher shutting down...")
    if state.recorder and state.active_session_id:
        await state.recorder.stop_session()
    if state.redis:
        await state.redis.close()
    if state.pipeline:
        state.pipeline.unload()
    if state.tts:
        state.tts.unload()


app = FastAPI(
    title="CueCatcher",
    description="Communication Pattern Analysis & Speech System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dashboard / session API
from server.api import router as api_router
app.include_router(api_router)


# ── Health / Info ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_loaded": state.pipeline is not None and state.pipeline.loaded,
        "tts_loaded": state.tts is not None and state.tts.loaded,
        "active_session": state.active_session_id,
        "connected_clients": len(state.connected_clients),
        "total_frames_processed": state.frame_count,
    }


@app.get("/api/config")
async def get_config():
    """Return current configuration for the UI."""
    return {
        "video": {"width": settings.video_width, "height": settings.video_height, "fps": settings.video_fps},
        "confidence_thresholds": {
            "high": settings.confidence_high,
            "medium": settings.confidence_medium,
            "low": settings.confidence_low,
        },
        "tts_enabled": settings.tts_enabled,
        "tts_voice_loaded": settings.tts_voice_reference is not None,
        "comm_matrix_levels": settings.comm_matrix_primary_levels,
    }


# ── Session Management ────────────────────────────────────────
@app.post("/api/session/start")
async def start_session():
    """Start a new observation session."""
    session_id = str(uuid.uuid4())
    state.active_session_id = session_id
    state.frame_count = 0
    state.is_running = True

    # Start recording
    if state.recorder:
        session_id = await state.recorder.start_session(record_video=settings.store_raw_video)
        state.active_session_id = session_id

    logger.info(f"📹 Session started: {session_id}")
    return {"session_id": session_id, "started_at": datetime.now(timezone.utc).isoformat()}


@app.post("/api/session/stop")
async def stop_session():
    """Stop the current session."""
    sid = state.active_session_id
    summary = None

    # Stop recording and get summary
    if state.recorder:
        summary = await state.recorder.stop_session()

    state.active_session_id = None
    state.is_running = False
    logger.info(f"⏹️  Session stopped: {sid}")
    return {
        "session_id": sid,
        "total_frames": state.frame_count,
        "summary": summary.to_dict() if summary else None,
    }


# ── Voice Reference Upload ────────────────────────────────────
@app.post("/api/voice/upload")
async def upload_voice_reference(file: UploadFile = File(...)):
    """Upload a parent's voice clip for Voxtral TTS cloning."""
    import sys
    voice_dir = Path("/data/voice") if sys.platform != "win32" else Path(__file__).parent.parent / "data" / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voice_dir / f"parent_voice_{int(time.time())}.wav"

    content = await file.read()
    voice_path.write_bytes(content)

    settings.tts_voice_reference = str(voice_path)

    if state.tts:
        await asyncio.get_event_loop().run_in_executor(
            None, state.tts.set_voice_reference, str(voice_path)
        )

    logger.info(f"🎤 Voice reference saved: {voice_path}")
    return {"status": "ok", "path": str(voice_path)}


# ── Caregiver Feedback ────────────────────────────────────────
@app.post("/api/feedback/{interpretation_id}")
async def submit_feedback(interpretation_id: str, feedback: dict):
    """Caregiver confirms or rejects an interpretation."""
    action = feedback.get("action")  # "confirmed" or "rejected"
    correct_meaning = feedback.get("correct_meaning")  # optional

    if state.interpreter:
        state.interpreter.record_feedback(interpretation_id, action, correct_meaning)

    logger.info(f"📝 Feedback for {interpretation_id}: {action}")
    return {"status": "ok"}


# ── WebSocket: Video/Audio Ingest ──────────────────────────────
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    Receive video frames + audio chunks from the tablet.

    Protocol:
    - Binary messages: JPEG-encoded video frames (prefixed with b'V')
                       or PCM audio chunks (prefixed with b'A')
    - Text messages:   JSON control commands
    """
    await ws.accept()
    state.connected_clients.add(ws)
    logger.info(f"📱 Tablet connected ({len(state.connected_clients)} clients)")

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                msg_type = raw[0:1]
                payload = raw[1:]

                if msg_type == b"V":
                    # Video frame
                    asyncio.create_task(_process_video_frame(payload, ws))
                elif msg_type == b"A":
                    # Audio chunk
                    asyncio.create_task(_process_audio_chunk(payload))

            elif "text" in data and data["text"]:
                cmd = orjson.loads(data["text"])
                await _handle_command(cmd, ws)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        state.connected_clients.discard(ws)
        logger.info(f"📱 Tablet disconnected ({len(state.connected_clients)} clients)")


async def _process_video_frame(jpeg_bytes: bytes, sender: WebSocket):
    """Decode frame, run inference, broadcast results."""
    if not state.is_running or not state.pipeline:
        return

    # Frame dropping: skip if already processing
    if not hasattr(state, '_processing'):
        state._processing = False
    if state._processing:
        return  # drop this frame
    
    state._processing = True
    state.frame_count += 1

    try:
        # Decode
        frame = await asyncio.get_event_loop().run_in_executor(None, _decode_jpeg, jpeg_bytes)
        if frame is None:
            return

        # Process
        detections = await asyncio.get_event_loop().run_in_executor(
            None, state.pipeline.process_frame, frame, state.frame_count
        )

        # Store in Redis ring buffer
        ts = time.time()
        if state.redis:
            key = f"frame:{state.active_session_id}:{state.frame_count}"
            await state.redis.setex(
                key,
                int(settings.tier1_buffer_seconds),
                orjson.dumps(detections)
            )

        # Run interpretation (if interpreter available)
        interpretation = None
        if state.interpreter:
            interpretation = await asyncio.get_event_loop().run_in_executor(
                None, state.interpreter.interpret, detections, state.frame_count
            )

        # Build result payload for UI
        result = {
            "frame_idx": state.frame_count,
            "timestamp": ts,
            "detections": _serialize_detections(detections),
        }
        if interpretation:
            result["interpretation"] = interpretation

            # Speak via TTS if confidence is high enough
            if (
                state.tts
                and state.tts.loaded
                and interpretation.get("confidence", 0) >= settings.tts_speak_confidence_threshold
                and interpretation.get("should_speak", False)
            ):
                asyncio.create_task(_speak_interpretation(interpretation))

        # Broadcast to all connected tablets
        msg = orjson.dumps(result)
        disconnected = set()
        for client in state.connected_clients:
            try:
                await client.send_bytes(msg)
            except Exception:
                disconnected.add(client)
        state.connected_clients -= disconnected

        # Record to session (async, non-blocking)
        if state.recorder:
            asyncio.create_task(state.recorder.record_frame(detections, frame if settings.store_raw_video else None))
            if interpretation:
                asyncio.create_task(state.recorder.record_interpretation(interpretation))
                
    finally:
        state._processing = False

async def _process_audio_chunk(pcm_bytes: bytes):
    """Process an audio chunk through the audio classifier."""
    if not state.is_running or not state.pipeline:
        return

    audio_result = await asyncio.get_event_loop().run_in_executor(
        None, state.pipeline.process_audio, pcm_bytes
    )

    if state.redis and audio_result:
        key = f"audio:{state.active_session_id}:{int(time.time() * 1000)}"
        await state.redis.setex(key, int(settings.tier1_buffer_seconds), orjson.dumps(audio_result))


async def _speak_interpretation(interpretation: dict):
    """Generate and send TTS audio for an interpretation."""
    if not state.tts:
        return

    text = interpretation.get("spoken_text", interpretation.get("description", ""))
    if not text:
        return

    try:
        audio_bytes = await asyncio.get_event_loop().run_in_executor(
            None, state.tts.synthesize, text
        )
        if audio_bytes:
            # Send audio to all clients (prefixed with 'S' for speech)
            msg = b"S" + audio_bytes
            for client in state.connected_clients:
                try:
                    await client.send_bytes(msg)
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"TTS error: {e}")


async def _handle_command(cmd: dict, ws: WebSocket):
    """Handle JSON control commands from the tablet."""
    action = cmd.get("action")

    if action == "start":
        result = await start_session()
        await ws.send_text(orjson.dumps(result).decode())

    elif action == "stop":
        result = await stop_session()
        await ws.send_text(orjson.dumps(result).decode())

    elif action == "feedback":
        if state.interpreter:
            state.interpreter.record_feedback(
                cmd.get("interpretation_id"),
                cmd.get("feedback"),
                cmd.get("correct_meaning")
            )

    elif action == "set_tts":
        settings.tts_enabled = cmd.get("enabled", True)

    elif action == "speak":
        # Child (or caregiver) pressed a communication button
        text = cmd.get("text", "")
        button_id = cmd.get("button_id", "")
        if text and state.tts:
            logger.info(f"💬 Button press: [{button_id}] \"{text}\"")
            asyncio.create_task(_speak_button(text, button_id))

    elif action == "log_comm":
        # Log a child-initiated communication event (button press)
        button_id = cmd.get("button_id", "")
        phrase = cmd.get("phrase", "")
        logger.info(f"📝 Child communication: button={button_id} phrase=\"{phrase}\"")
        if state.recorder:
            state.recorder.record_button_press(button_id, phrase)


async def _speak_button(text: str, button_id: str):
    """Speak a communication button phrase via TTS and broadcast to all clients."""
    if not state.tts:
        return
    try:
        audio_bytes = await asyncio.get_event_loop().run_in_executor(
            None, state.tts.synthesize, text
        )
        if audio_bytes:
            msg = b"S" + audio_bytes
            disconnected = set()
            for client in state.connected_clients:
                try:
                    await client.send_bytes(msg)
                except Exception:
                    disconnected.add(client)
            state.connected_clients -= disconnected
    except Exception as e:
        logger.error(f"Button TTS error: {e}")


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _serialize_detections(detections: dict) -> dict:
    """Make detections JSON-serializable (convert all numpy types)."""
    result = {}
    for key, value in detections.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        elif isinstance(value, dict):
            result[key] = _serialize_detections(value)
        elif isinstance(value, list):
            result[key] = [
                _serialize_detections(v) if isinstance(v, dict)
                else v.item() if isinstance(v, (np.floating, np.integer))
                else v.tolist() if isinstance(v, np.ndarray)
                else bool(v) if isinstance(v, np.bool_)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


# ── Serve the PWA UI ───────────────────────────────────────────
ui_path = Path("/app/static")
if not ui_path.exists():
    ui_path = Path(__file__).parent.parent / "ui" / "dist"

if ui_path.exists() and (ui_path / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=ui_path / "assets"), name="assets")

    @app.get("/")
    async def serve_ui():
        return FileResponse(ui_path / "index.html")
else:
    @app.get("/")
    async def serve_placeholder():
        """Landing page with connection status and setup info."""
        import sys
        gpu = "unknown"
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return JSONResponse({
            "name": "CueCatcher",
            "status": "running",
            "platform": sys.platform,
            "gpu": gpu,
            "pipeline_loaded": state.pipeline is not None and state.pipeline.loaded if state.pipeline else False,
            "tts_loaded": state.tts is not None and state.tts.loaded if state.tts else False,
            "tts_mode": state.tts.mode_description if state.tts and hasattr(state.tts, 'mode_description') else "not loaded",
            "recorder_ready": state.recorder is not None,
            "redis_connected": state.redis is not None,
            "docs": "/docs",
            "health": "/health",
            "note": "Open /docs for the API explorer. Connect the tablet UI to ws://<this-ip>:4/ws/stream",
        })
