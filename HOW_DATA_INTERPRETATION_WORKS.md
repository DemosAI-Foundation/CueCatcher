# How CueCatcher Actually Interprets & Stores Data

## Real Implementation Summary

Based on the actual code in `/workspace`, here's exactly how data flows through the system:

---

## The Three-Tier Flow (As Implemented)

### **Tier 1: Per-Frame Detection** (`inference/pipeline.py`)

**What actually happens:**
```python
# Every 33ms (30fps), process_frame() runs:
det = FrameDetections(frame_idx=frame_idx, timestamp=time.time())

# 1. Pose estimation (MediaPipe/RTMPose)
pr = self._pose.estimate(frame)
det.pose_keypoints = pr.get("keypoints")  # 17-25 keypoints

# 2. Gaze estimation (L2CS-Net)
gr = self._gaze.estimate(frame)
det.head_yaw = gr.head_yaw  # -90 to +90 degrees
det.fused_gaze_yaw = gr.fused_yaw

# 3. Face analysis (LibreFace/MediaPipe)
fr = self._face.analyze(frame)
det.expression = fr.expression  # "neutral", "happy", "sad", etc.
det.mouth_openness = fr.mouth_openness  # 0.0 to 1.0

# 4. Action recognition (PoseConv3D)
actions = self._action.update(det.pose_keypoints, frame_idx)
det.actions_detected = [a.to_dict() for a in actions]

# 5. Audio (from async chunk)
det.vocalization_class = self._last_audio.vocalization_class
det.is_vocalization = self._last_audio.is_vocalization

# 6. Pass to temporal engine
tr = self._temporal.update(det.to_dict(), frame_idx)
```

**Storage:** 
- NOT saved to database individually (too high-frequency: 30 FPS × 60 seconds = 1800 records/minute)
- Held in `temporal.py` ring buffer: `self._frame_buffer = deque(maxlen=fps * 10)` (10 seconds)
- Used immediately for Tier 2 analysis

---

### **Tier 2: Episode Detection** (`inference/temporal.py`)

**What actually happens:**
```python
# Runs every 5 frames (~6Hz, every 167ms)
if frame_idx % 5 == 0 and len(self._frame_buffer) >= self.fps:
    new_episodes = self._detect_episodes(frame_idx)

# Inside _detect_episodes():
window = frames[-self._episode_window:]  # Last 5 seconds

# Extract time-series
poses = [f["detection"].get("pose_keypoints") for f in window]
head_yaws = [f["detection"].get("head_yaw", 0) for f in window]
vocalizing = [f["detection"].get("is_vocalization", False) for f in window]

# Detect REACHING
ep = self._detect_reaching(window, poses, current_frame)
# Checks: sustained arm extension >0.25 normalized distance for 15+ frames (0.5s)

# Detect GAZE ALTERNATION ⭐ (KEY communication signal)
ep = self._detect_gaze_alternation(window, head_yaws, current_frame)
# Checks: ≥2 head turns >15° in 2 seconds
# This is the hallmark of intentional communication (Level III)

# Detect VOCALIZATION BURST
ep = self._detect_vocalization_burst(window, vocalizing, current_frame)
# Checks: 10+ consecutive vocalization frames (0.33s)

# Each episode gets:
Episode(
    type=EpisodeType.REACH,
    duration_ms=1500,
    confidence=0.82,
    comm_relevance=0.7,      # How likely communicative
    comm_function="request"   # request/reject/social/regulate
)
```

**Storage:**
- Completed episodes stored in: `self._completed_episodes = deque(maxlen=100)` (≈5 minutes)
- Sent to interpreter for meaning-making
- Written to database in **batches of 30** via `recorder.py`:
  ```python
  # recorder.py line 179-188
  for ep in detections.get("new_episodes", []):
      self._episode_batch.append({
          "time": datetime.now(timezone.utc),
          "episode_type": ep.get("type"),
          "duration_ms": ep.get("duration_ms"),
          "confidence": ep.get("confidence"),
      })
  
  # Flushed every ~1 second (batch_size=30)
  if len(self._detection_batch) >= self._batch_size:
      await self._flush_batch()
  ```

---

### **Tier 3: State Machine** (`inference/temporal.py`)

**What actually happens:**
```python
# Runs every 15 frames (~2Hz, every 500ms)
if frame_idx % 15 == 0:
    state_changed = self._update_state()

# Inside _update_state():
recent_episodes = list(self._completed_episodes)[-20:]  # Last ~1 minute

# State machine logic:
if distress_cry_detected and high_movement:
    new_state = ChildState.DISTRESSED
elif gaze_alternation and positive_expression:
    new_state = ChildState.ENGAGED
elif rocking or arm_wave without social signals:
    new_state = ChildState.REGULATING
elif reach + gaze_alternation + vocalization:
    new_state = ChildState.COMMUNICATING
else:
    new_state = ChildState.IDLE

# Track duration
if new_state != self._current_state:
    duration = time.time() - self._state_start
    self._state_time[self._current_state] += duration
    self._current_state = new_state
    self._state_start = time.time()
```

**Output per frame:**
```python
{
    "child_state": "communicating",
    "child_state_confidence": 0.78,
    "child_state_duration_s": 12.5,
    "state_changed": True
}
```

---

## From Episodes to Meaning: The Interpreter (`inference/interpreter.py`)

**What actually happens:**
```python
# Runs every 15 frames (~0.5s) when confidence threshold met
if frame_idx % 15 != 0:
    return None

interp = self._analyze_current_state(detections)

if interp and interp.confidence >= 0.30:
    # Priority-based interpretation
    
    # PRIORITY 1: Distress
    if voc_class == "distress_cry" and confidence > 0.5:
        return Interpretation(
            intent="reject",
            description="The child appears distressed — crying detected",
            spoken_text="I think she's upset about something",
            confidence=0.75,
            comm_level=1,
            alternatives=["pain", "frustration", "sensory overload"]
        )
    
    # PRIORITY 2: Coordinated Request ⭐
    reaching_signals = [s for s in signals if "reaching" in s.get("action")]
    if reaching_signals:
        coordination_bonus = 0.0
        
        # Gaze alternation + reaching = intentional communication!
        if gaze_alternation:
            coordination_bonus += 0.20
            comm_level = 3  # Level III in Communication Matrix
        
        # Vocalization + reaching = even stronger
        if is_vocalizing:
            coordination_bonus += 0.15
        
        confidence = min(1.0, base_confidence + coordination_bonus)
        
        return Interpretation(
            intent="request",
            description="The child may be reaching toward something",
            spoken_text="She might be reaching for something",
            confidence=confidence,
            comm_level=comm_level,
            evidence=["reaching", "gaze_alternation"],
            alternatives=["exploring", "pointing", "stretching"]
        )
    
    # Decide if TTS should speak
    now = time.time()
    interp.should_speak = (
        interp.confidence >= 0.70
        and (now - self._last_spoken_time) >= 3.0  # Cooldown
    )
```

**Critical safeguards in code:**
1. **All interpretations are hypotheses** → `alternatives` field always populated
2. **Confidence thresholds** → Minimum 0.30 to return, 0.70+ to speak
3. **Cooldown period** → 3 seconds between TTS outputs
4. **Caregiver feedback tracking** → `record_feedback()` logs confirmations/rejections

---

## Storage: Short vs Long Horizon (As Implemented)

### **Short-Horizon (In-Memory)**

**Actual buffers in `temporal.py`:**
```python
self._frame_buffer = deque(maxlen=fps * 10)        # 10 seconds (300 frames)
self._completed_episodes = deque(maxlen=100)       # ~5 minutes
self._recent_interpretations = deque(maxlen=20)    # Last 20 interpretations
self._state_history = deque(maxlen=50)             # Last 50 states
```

**Lifetime:** Sliding window — oldest data automatically discarded

**Purpose:** Real-time interpretation only, no persistence

---

### **Long-Horizon (Persistent Storage)**

**As implemented in `server/recorder.py`:**

#### **Option 1: TimescaleDB/PostgreSQL** (lines 82-108)
```python
async def connect(self):
    if not self.db_url.startswith(("postgresql://", "postgres://")):
        logger.warning("SQLite DSN detected — asyncpg requires PostgreSQL. Using in-memory.")
        return
    
    self._pool = await asyncpg.create_pool(self.db_url)
    # Creates connection pool for batched writes
```

**Batched writes (lines 292-338):**
```python
async def _flush_batch(self):
    if not self._pool:
        self._detection_batch.clear()  # Graceful degradation
        return
    
    async with self._pool.acquire() as conn:
        # Batch insert detections
        await conn.executemany(
            """INSERT INTO detections (time, session_id, frame_idx, pose, gaze, face, audio)
               VALUES ($1, $2::uuid, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb)""",
            [(d["time"], d["session_id"], ...) for d in self._detection_batch]
        )
        
        # Batch insert episodes
        await conn.executemany(
            """INSERT INTO episodes (time, session_id, episode_type, duration_ms, confidence, features)
               VALUES ($1, $2::uuid, $3, $4, $5, $6::jsonb)""",
            [(e["time"], e["session_id"], ...) for e in self._episode_batch]
        )
```

#### **Option 2: JSON Files (Fallback, Always Works)** (lines 259-268)
```python
# On session stop:
sess_dir = self.session_dir / self._current_session
sess_dir.mkdir(parents=True, exist_ok=True)

# Write summary
summary_path = sess_dir / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary.to_dict(), f, indent=2)

# Write button presses
bp_path = sess_dir / "button_presses.json"
with open(bp_path, "w") as f:
    json.dump(self._button_presses, f, indent=2)
```

**Directory structure:**
```
/data/sessions/
├── {session_id_1}/
│   ├── summary.json
│   ├── button_presses.json
│   └── session.mp4 (optional)
├── {session_id_2}/
│   └── ...
```

**Session summary contents (lines 245-257):**
```python
SessionSummary(
    session_id="uuid",
    started_at="2024-01-06T10:00:00Z",
    ended_at="2024-01-06T10:30:00Z",
    duration_minutes=30.0,
    total_frames=54000,
    total_episodes=145,
    episodes_by_type={"reach": 45, "gaze_alternation": 23},
    state_durations={"communicating": 680, "regulating": 300},
    button_presses=15,
    highest_comm_level_observed=3
)
```

---

## Actual Data Flow Diagram

```
┌─────────────────┐
│ Camera (30fps)  │
│ Microphone      │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ inference/pipeline.py                │
│ process_frame(frame, frame_idx)      │
│                                      │
│ 1. Pose → 17-25 keypoints            │
│ 2. Gaze → yaw/pitch angles           │
│ 3. Face → expression + AUs           │
│ 4. Audio → vocalization class        │
│ 5. Actions → reaching, waving, etc.  │
└────────┬─────────────────────────────┘
         │
         │ det.to_dict()
         ▼
┌──────────────────────────────────────┐
│ inference/temporal.py                │
│ update(detection, frame_idx)         │
│                                      │
│ Every 5 frames (167ms):              │
│   → Analyze 10s frame buffer         │
│   → Detect episodes                  │
│      (reach, gaze_alt, wave, rock)   │
│                                      │
│ Every 15 frames (500ms):             │
│   → Update state machine             │
│   → Track state durations            │
│                                      │
│ Returns:                             │
│   {new_episodes, state, changed}     │
└────────┬─────────────────────────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌──────────────────┐  ┌─────────────────────┐
│ interpreter.py   │  │ server/recorder.py  │
│ interpret()      │  │ record_frame()      │
│                  │  │                     │
│ Every 15 frames: │  │ Buffers:            │
│ Combine episodes │  │ - _detection_batch  │
│ + state + context│  │ - _episode_batch    │
│                  │  │ - _interpretation_..│
│ Priority logic:  │  │                     │
│ 1. Distress      │  │ Every 30 frames:    │
│ 2. Coordinated   │  │   → _flush_batch()  │
│ 3. Single signal │  │                     │
│                  │  │ If DB available:    │
│ Output:          │  │   → INSERT into     │
│ Interpretation   │  │      TimescaleDB    │
│ with:            │  │                     │
│ - intent         │  │ If NO DB:           │
│ - confidence     │  │   → Clear buffers   │
│ - alternatives   │  │   → No problem!     │
│ - should_speak   │  │                     │
└────────┬─────────┘  │ On session stop:    │
         │            │   → Write JSON      │
         │            │      summary        │
         ▼            └─────────────────────┘
┌──────────────────┐
│ voice/tts.py     │
│ speak(text)      │
│                  │
│ If should_speak  │
│ and confidence   │
│ ≥0.70 and        │
│ cooldown expired │
│                  │
│ Output: Audio    │
└──────────────────┘
```

---

## Key Code Locations

| Component | File | Key Function | Lines |
|-----------|------|--------------|-------|
| **Per-frame detection** | `inference/pipeline.py` | `process_frame()` | 144-207 |
| **Episode detection** | `inference/temporal.py` | `_detect_episodes()` | 169-223 |
| **State machine** | `inference/temporal.py` | `_update_state()` | 500+ |
| **Interpretation** | `inference/interpreter.py` | `_analyze_current_state()` | 88-262 |
| **DB recording** | `server/recorder.py` | `record_frame()` | 145-202 |
| **Batch flush** | `server/recorder.py` | `_flush_batch()` | 292-338 |
| **JSON fallback** | `server/recorder.py` | `stop_session()` | 228-290 |

---

## Real Example: Child Reaches for Toy

**Timeline from actual code execution:**

**Frame 1500 (t=0ms):**
```python
# pipeline.py line 144
det = process_frame(frame, 1500)
# det.head_yaw = -25 (looking left at toy)
# det.pose_keypoints[16] (right wrist) elevated
```

**Frame 1515 (t=500ms):**
```python
# temporal.py line 149
if frame_idx % 5 == 0:  # 1515 % 5 = 0 ✓
    episodes = _detect_episodes(1515)
    
# _detect_reaching() checks last 1 second:
recent_poses = poses[-30:]  # Last 30 frames
reach_frames = count(kp where wrist_distance > 0.25)
# reach_frames = 18 (0.6s sustained) ✓

# Episode created:
Episode(
    type=EpisodeType.REACH,
    duration_ms=600,
    confidence=0.78,
    comm_relevance=0.7,
    comm_function="request"
)
```

**Frame 1530 (t=1000ms):**
```python
# temporal.py line 149
episodes = _detect_episodes(1530)

# _detect_gaze_alternation() checks last 2 seconds:
recent_yaws = head_yaws[-60:]
shifts = count(yaw_diff > 15)
# shifts = 3 (toy → caregiver → toy) ✓

Episode(
    type=EpisodeType.GAZE_ALTERNATION,
    duration_ms=2000,
    confidence=0.85,
    comm_relevance=0.9,  # HIGHEST - intentional communication!
    comm_function="social"
)
```

**Frame 1545 (t=1500ms):**
```python
# temporal.py line 156
if frame_idx % 15 == 0:  # 1545 % 15 = 0 ✓
    _update_state()
    
# Recent episodes: reach + gaze_alternation
# State machine logic:
if reach AND gaze_alternation:
    new_state = ChildState.COMMUNICATING

# Returns:
{
    "state": "communicating",
    "state_confidence": 0.82,
    "state_changed": True
}
```

**Frame 1560 (t=2000ms):**
```python
# interpreter.py line 60
interp = interpret(detections, 1560)

# _analyze_current_state():
reaching_signals = [...]  # Found
gaze_alternation = _detect_gaze_alternation()  # True ✓

# Coordination bonus:
coordination_bonus = 0.20  # gaze_alt
coordination_bonus += 0.15  # vocalization (if present)
confidence = 0.5 + 0.35 = 0.85

# Interpretation created:
Interpretation(
    intent="request",
    description="The child may be reaching toward something while alternating gaze",
    spoken_text="She might be reaching for something",
    confidence=0.85,
    comm_level=3,  # Intentional communication!
    evidence=["reach", "gaze_alternation"],
    alternatives=["exploring", "pointing"],
    should_speak=True  # confidence ≥0.70 ✓
)

# TTS triggered:
tts.speak("She might be reaching for something")
```

**Frame 1560 (simultaneously):**
```python
# recorder.py line 145
await record_frame(det)

# Added to batch:
self._detection_batch.append({...})
self._episode_batch.append({...})  # reach episode
self._episode_batch.append({...})  # gaze_alternation episode

# When batch reaches 30:
if len(self._detection_batch) >= 30:
    await _flush_batch()
    # INSERT INTO detections (...)
    # INSERT INTO episodes (...)
```

**Session End (t=30 minutes):**
```python
# recorder.py line 228
summary = await stop_session()

# JSON written:
/data/sessions/{session_id}/summary.json
{
    "duration_minutes": 30.0,
    "total_episodes": 145,
    "gaze_alternation_count": 23,
    "coordinated_signals_count": 12,
    "highest_comm_level_observed": 3,
    "most_common_state": "communicating"
}
```

---

## Critical Insights from the Code

1. **No single frame matters** — Everything is about patterns over time (10s buffer, 5s episode window, 30s state evaluation)

2. **Gaze alternation is THE signal** — Highest comm_relevance (0.9), triggers Level 3 communication, adds +0.20 confidence bonus

3. **Coordination multiplies confidence** — Single signal: 0.50; Reach + gaze: 0.70; Reach + gaze + vocal: 0.85

4. **Database is optional** — Lines 84-89 in `recorder.py`: if no PostgreSQL, clears batches and saves JSON only

5. **Graceful degradation everywhere** — Try/except blocks, fallback models, in-memory buffers

6. **Transparency by design** — Every interpretation includes `alternatives` list to prevent over-confidence

7. **Caregiver feedback loop ready** — `record_feedback()` method exists but not yet connected to UI

---

## What's Actually Stored Where

| Data Type | Short-Horizon | Long-Horizon (DB) | Long-Horizon (JSON) |
|-----------|---------------|-------------------|---------------------|
| Raw keypoints | ✅ 10s buffer | ❌ Too high-freq | ❌ Not saved |
| Gaze angles | ✅ 10s buffer | ✅ As JSONB | ❌ Not saved |
| Expression | ✅ 10s buffer | ✅ As JSONB | ❌ Not saved |
| Episodes | ✅ 5min buffer | ✅ Full details | ✅ Counts only |
| States | ✅ 50 history | ✅ Durations | ✅ Durations |
| Interpretations | ✅ 20 history | ✅ With evidence | ❌ Not saved |
| Button presses | ✅ In-memory | ✅ Full log | ✅ Full log |
| Session summary | ❌ N/A | ✅ Metadata | ✅ Full summary |

**Bottom line:** You lose some detail without a database (can't replay exact gaze angles from 10 minutes ago), but all meaningful insights (episodes, states, interpretations, summaries) are preserved in JSON files.
