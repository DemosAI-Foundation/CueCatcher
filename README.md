# CueCatcher — Communication and Interpretation System for non-verbal children

**Real-time AI system that interprets childrens non-verbal communication throguh analyzing macro and micro movement, poses, sounds etc. Runs on local computer with powerful GPU and phone or webcam, no data leaves local server.**


## Quick Start

```bash
# 1. Clone and setup
cd CueCatcher
./scripts/setup.sh

# 2. Download models (first time only)
python scripts/download_models.py

# 3. Start the stack (optional: /DB for advanced features)
docker compose up -d
# OR run standalone without Docker: just start the server directly

# 4. Open the tablet UI
# Navigate to http://<server-ip>:8084 on your tablet
```

**Note:** CueCatcher works **without any database**.  and DB are optional enhancements:
- **No database required**: Core real-time interpretation works standalone with in-memory buffers and JSON session files
- **** (optional): Enhances frame buffering performance
- **DB/PostgreSQL** (optional): Enables long-term analytics, therapist exports, and longitudinal dashboards
- **SQLite**: Supported for basic persistence if PostgreSQL is not available

## ToDo

### Phase 1: Foundation ✅
- [x] FastAPI server with WebSocket streaming
- [x] Video capture and transport pipeline
- [x] Basic pose estimation (MediaPipe MVP)
- [x] Tablet PWA with live feed + overlays
- [x] Voxtral TTS integration
- [x] AAC communication board (3/6/12 buttons)

### Phase 2: Full Perception 
- [x] Production pose module (YOLO11 + RTMPose with MediaPipe fallback)
- [x] Gaze estimation (L2CS-Net + head pose, 70/30 weighting)
- [] Facial expression analysis (LibreFace + child-specific calibration)
- [] Audio classification (PANNs CNN14 for non-speech vocalizations)
- [] Action recognition (hand flapping, reaching, rocking, covering ears, spinning)
- [x] Face calibration tool for 9p deletion craniofacial features

### Phase 3: Intelligence 
- [] Three-tier temporal analysis (frame → episode → state)
- [] Probabilistic intent interpreter
- [] Caregiver feedback loop
- [] Behavior dictionary (pre-loaded + learned patterns)
- [] Pipeline fully wired (all modules connected)

### Phase 4: Recording & Analysis 
- [] Session recording to DB
- [] Session replay and therapist CSV export
- [] Longitudinal dashboard (daily/weekly trends)
- [] Hourly communication heatmap
- [] Button press tracking and analytics
- [x] Voxtral voice cloning end-to-end (3 tiers)

## Voice Cloning Setup

Three options, from easiest to most powerful:

### Option A: Mistral API (easiest)
```bash
export MISTRAL_API_KEY=your_key_here
# Restart CueCatcher — it auto-detects the key
# Upload a 5-25s parent voice clip via Settings → Voice
```

### Option B: Community encoder (fully offline)
```bash
./scripts/setup_voice_clone.sh
# Follow the training instructions
# Then restart CueCatcher
```

### Option C: Preset voices (no cloning)
Works out of the box. Uses a warm female preset voice.
Upload a voice clip later when you're ready for cloning.

## Face Calibration

Standard expression models fail for childrens craniofacial features.
Run the calibration tool to build a personalized expression baseline:

```bash
python scripts/calibrate.py --mode capture   # guided photo session
python scripts/calibrate.py --mode build     # build the model
```

## Therapist Integration

Export any session as CSV for therapist review:
```
GET /api/sessions/{session_id}/export?format=csv
```

Or use the dashboard API for longitudinal data:
```
GET /api/dashboard/summary?days=30    # overall trends
GET /api/dashboard/daily?days=14      # day-by-day
GET /api/dashboard/weekly?weeks=12    # weekly developmental tracking
GET /api/dashboard/patterns           # learned communication patterns
GET /api/dashboard/hourly?days=7      # when does she communicate?
```

## Architecture

```
┌──────────────┐     WebRTC/WS      ┌─────────────────────────────────┐
│  Phone/Tablet │ ──────────────────▶│  FastAPI Server (port 8084)     │
│  (PWA)        │ ◀──────────────────│  - WebSocket video receiver     │
│  - Camera     │   interpretations  │  - WebSocket result sender      │
│  - Mic        │   + audio          │  - REST API for config/history  │
│  - UI         │                    └──────────┬──────────────────────┘
└──────────────┘                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │  Inference Pipeline     │
                                   │  (RTX 3090 — 24GB)     │
                                   │                        │
                                   │  ┌──────────────────┐  │
                                   │  │ YOLO11 (detect)   │  │
                                   │  │ RTMPose (pose)    │  │
                                   │  │ L2CS-Net (gaze)   │  │
                                   │  │ LibreFace (face)  │  │
                                   │  │ PANNs (audio)     │  │
                                   │  │ PoseConv3D (action)│  │
                                   │  └──────────────────┘  │
                                   └──────────┬─────────────┘
                                              │
                                              ▼
                                   ┌────────────────────────┐
                                   │  Temporal Analysis      │
                                   │  Tier 1: per-frame      │
                                   │  Tier 2: 1-5s episodes  │
                                   │  Tier 3: state machine  │
                                   └──────────┬─────────────┘
                                              │
                                   ┌──────────▼─────────────┐
                                   │  Interpreter            │
                                   │  → Probabilistic intent │
                                   │  → Voxtral TTS output   │
                                   │  → WebSocket to UI      │
                                   └────────────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                              ▼               ▼               ▼
                         ┌────────┐    ┌───────────┐   ┌────────────┐
                         │ ? │    │DB│   │ Voxtral TTS│
                         │(opt)   │    │(opt)      │   │(voice out) │
                         └────────┘    └───────────┘   └────────────┘
                                              │
                              (All components work without DBs
                               using in-memory buffers + JSON files)
```


## Project Structure

```
CueCatcher/
├── server/              # FastAPI application
│   ├── main.py          # Entrypoint, WebSocket handlers
│   ├── stream.py        # Video/audio stream processing
│   └── api.py           # REST endpoints for config, history
├── inference/           # ML pipeline
│   ├── pipeline.py      # Orchestrator — runs all models
│   ├── pose.py          # RTMPose + YOLO body tracking
│   ├── gaze.py          # Head pose + gaze estimation
│   ├── face.py          # Facial expression / AU detection
│   ├── audio.py         # Non-speech vocalization analysis
│   ├── action.py        # Stereotypy & gesture recognition
│   ├── temporal.py      # 3-tier temporal analysis engine
│   └── interpreter.py   # Behavioral → communicative intent
├── voice/               # TTS module
│   └── tts.py           # Voxtral TTS with parent voice clone
├── ui/                  # React PWA (tablet interface)
├── config/
│   ├── settings.py      # Central configuration
│   ├── behaviors.yaml   # Learned behavior dictionary
│   └── init.sql         # DB schema
├── scripts/
│   ├── setup.sh         # Environment setup
│   ├── setup_voice_clone.sh  # Voice cloning encoder setup
│   ├── download_models.py    # ML model downloader
│   └── calibrate.py     # Child-specific face calibration
├── docker-compose.yml
└── README.md
```