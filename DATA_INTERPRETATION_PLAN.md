# CueCatcher Data Interpretation & Storage Plan

## Executive Summary

This document outlines how CueCatcher interprets raw sensor data into meaningful insights and stores them across **short-horizon** (real-time, per-frame) and **long-horizon** (session-level, longitudinal) timescales. The system uses a **three-tier temporal aggregation** approach that progressively builds meaning from raw detections to behavioral episodes to communicative states.

---

## 1. Data Hierarchy: From Signals to Meaning

### Tier 1: Per-Frame Detections (0ms horizon)
**What:** Raw sensor outputs from ML models  
**Frequency:** 30 Hz (every 33ms)  
**Storage:** High-frequency time-series data  

| Signal | Source | Data Structure | Example |
|--------|--------|----------------|---------|
| **Pose** | MediaPipe BlazePose | 133 keypoints (x,y,visibility) | `[[x,y,v], [x,y,v], ...]` |
| **Gaze** | Gaze model + head pose | `head_yaw`, `head_pitch`, `fused_gaze_yaw`, `target` | `{head_yaw: -15, head_pitch: 5}` |
| **Face** | MediaPipe Face Mesh | Expression label, AU scores, mouth openness, smile | `{expression: "smile", confidence: 0.85}` |
| **Audio** | Audio classifier | Class, confidence, pitch (Hz), energy (dB), is_vocal | `{class: "babble", pitch: 450}` |

**Interpretation:** None at this level — these are objective measurements only.

---

### Tier 2: Behavioral Episodes (1-5 second horizon)
**What:** Temporally-aggregated patterns that indicate specific behaviors  
**Frequency:** Detected every ~0.5-2 seconds when patterns emerge  
**Storage:** Event-based records with duration and features  

#### Episode Types & Detection Logic:

| Episode Type | Duration | Detection Criteria | Communicative Relevance | Function |
|--------------|----------|-------------------|------------------------|----------|
| **reach** | 0.5-3s | Sustained arm extension >25% body width for 15+ frames | 0.7 (high) | request |
| **gaze_alternation** | 2s | ≥2 head turns >15° in 2 seconds | 0.9 (very high) | social/intentional |
| **arm_wave** | 0.5-2s | High wrist position variance (>0.005) | 0.5 (moderate) | regulate/social |
| **arms_up** | 1-3s | Both arms raised above shoulders | 0.85 (high) | request (pickup) |
| **rocking** | 2-5s | Periodic hip oscillation ≥2 cycles | 0.3 (low-moderate) | regulate |
| **vocalization_burst** | 0.3-3s | Consecutive vocal frames ≥10 | 0.6 (moderate) | varies by class |
| **distress_cry** | 1-5s | Sustained distress vocalization | 0.75 (high) | reject/regulate |
| **withdrawal** | 2-5s | Going still + head turn away | 0.5 (moderate) | regulate/reject |

**Key Insight:** A single frame of reaching means nothing. Two seconds of sustained reaching + gaze alternation + vocalization = probable request.

**Storage Format:**
```json
{
  "type": "reach",
  "start_frame": 1523,
  "end_frame": 1568,
  "start_time": 1704567890.123,
  "duration_ms": 1500,
  "confidence": 0.82,
  "features": {"side": "right", "extension": 0.35},
  "comm_relevance": 0.7,
  "comm_function": "request"
}
```

---

### Tier 3: Child State (30s-5min horizon)
**What:** Overall emotional/communicative state of the child  
**Frequency:** Updated every 0.5 seconds  
**Storage:** State transitions with timestamps and durations  

#### Child States:

| State | Description | Trigger Conditions |
|-------|-------------|-------------------|
| **idle** | No significant activity | No episodes detected in last 5s |
| **attending** | Focused on something | Sustained gaze direction + low movement |
| **communicating** | Active communication attempt | ≥2 coordinated episodes (e.g., reach + gaze_alt) |
| **distressed** | Crying, agitated | Distress cry episode + high movement |
| **engaged** | In social interaction | Gaze alternation + positive expression |
| **regulating** | Stimming, self-soothing | Rocking/arm_wave without social signals |
| **transitioning** | Shifting between states | State change in progress |
| **withdrawn** | Shutdown, disengaged | Withdrawal episode + no response to stimuli |

**State Machine Logic:**
- Transitions tracked with timestamps
- Duration of each state calculated
- Transition patterns analyzed (e.g., how often distressed → regulating)

---

### Tier 4: Interpretations (Semantic Meaning)
**What:** Hypothesized communicative intent based on combined signals  
**Frequency:** Every 0.5 seconds (when confidence ≥30%)  
**Storage:** Interpretation records with evidence and alternatives  

#### Interpretation Framework (Communication Matrix):

| Intent | Target | Description | Confidence Threshold | Spoken? |
|--------|--------|-------------|---------------------|---------|
| **request** | object/person/action | "The child may be reaching for something" | ≥0.70 | Yes if ≥0.70 |
| **reject** | stimulus/situation | "The child appears distressed — crying detected" | ≥0.75 | Yes if urgent |
| **social** | person | "She seems excited about something" | ≥0.45 | Selective |
| **regulate** | self | "She's moving her arms a lot" | ≥0.55 | Rarely |
| **explore** | environment | "He's investigating something" | ≥0.40 | No |

**Critical Safeguards:**
1. **ALL interpretations are hypotheses, never facts**
2. Alternatives always provided (e.g., "could be stimming, not communication")
3. Confidence thresholds prevent over-interpretation
4. Cooldown period (3s) prevents TTS spam

**Storage Format:**
```json
{
  "id": "uuid-1234",
  "timestamp": 1704567890.5,
  "intent": "request",
  "target": "cup",
  "description": "The child may be reaching toward the cup",
  "spoken_text": "She might be reaching for something",
  "confidence": 0.82,
  "comm_level": 3,
  "evidence": ["reach_right", "gaze_alternation"],
  "alternatives": ["exploring/touching", "pointing at something interesting"],
  "should_speak": true
}
```

---

## 2. Storage Architecture

### Short-Horizon Data (Real-Time Buffer)

**Purpose:** Immediate interpretation and TTS output  
**Retention:** 5-10 seconds sliding window  
**Storage Location:** In-memory deques (no database required)

| Component | Data Structure | Max Size | Purpose |
|-----------|---------------|----------|---------|
| `_frame_buffer` | deque of detections | 300 frames (10s @ 30fps) | Episode detection |
| `_history` | deque of detections | 150 frames (5s @ 30fps) | Interpretation context |
| `_recent_interpretations` | deque | 20 items | Avoid repetition |
| `_active_episodes` | dict | Variable | Track ongoing episodes |
| `_completed_episodes` | deque | 100 items | Recent episode history |

**When Database is Unavailable:**
- All data stays in memory during session
- Session summary saved as JSON file on stop
- No historical analysis possible, but real-time works perfectly

---

### Long-Horizon Data (Session & Longitudinal)

**Purpose:** Therapist review, pattern analysis, progress tracking  
**Retention:** Indefinite (user-managed)  
**Storage Options:**

#### Option A: TimescaleDB/PostgreSQL (Recommended for Clinics)
```sql
-- Detections table (compressed after 7 days)
detections (time, session_id, frame_idx, pose, gaze, face, audio)

-- Episodes table (compressed after 30 days)
episodes (time, session_id, episode_type, duration_ms, confidence, features)

-- Interpretations table
interpretations (time, session_id, intent, target, description, confidence, comm_level, evidence, spoken, caregiver_feedback)

-- Sessions table
sessions (id, started_at, ended_at, total_frames, total_episodes, total_interpretations, notes)

-- Behavior dictionary (learned patterns)
behavior_dictionary (pattern_name, description, pattern_data, confidence, times_confirmed, times_rejected)
```

**Batching Strategy:**
- Accumulate 30 frames before flushing to DB
- Reduces write amplification by 30x
- Graceful degradation if DB unavailable

#### Option B: SQLite (Supported but Limited)
- Same schema as PostgreSQL
- No time-series compression
- Suitable for single-user setups

#### Option C: JSON Files (Fallback, Always Available)
```
/data/sessions/
├── {session_id}/
│   ├── summary.json        # Session statistics
│   ├── button_presses.json # Child-initiated AAC presses
│   └── session.mp4         # Optional video recording
```

**Summary.json Contents:**
```json
{
  "session_id": "uuid",
  "started_at": "2024-01-06T10:00:00Z",
  "ended_at": "2024-01-06T10:30:00Z",
  "duration_minutes": 30.0,
  "total_frames": 54000,
  "avg_fps": 30.0,
  "total_episodes": 145,
  "total_interpretations": 42,
  "confirmed_interpretations": 38,
  "rejected_interpretations": 4,
  "episodes_by_type": {
    "reach": 45,
    "gaze_alternation": 23,
    "vocalization_burst": 67,
    "rocking": 10
  },
  "state_durations": {
    "idle": 420,
    "communicating": 680,
    "regulating": 300,
    "distressed": 120
  },
  "button_presses": 15,
  "button_breakdown": {
    "want_drink": 8,
    "all_done": 4,
    "help": 3
  }
}
```

---

## 3. Data Interpretation Pipeline

### Step-by-Step Flow:

```
[Camera/Mic] 
    ↓ (30 fps)
[Tier 1: Raw Detections]
    ├─ Pose keypoints (133 points)
    ├─ Gaze angles (yaw, pitch)
    ├─ Face expression (label + confidence)
    └─ Audio class (pitch, energy)
    ↓ (every 5 frames = 0.17s)
[Tier 2: Episode Detection]
    ├─ Analyze 5-second sliding window
    ├─ Detect patterns (reach, gaze_alt, wave, etc.)
    └─ Output: Completed episodes with duration/confidence
    ↓ (every 15 frames = 0.5s)
[Tier 3: State Update]
    ├─ Aggregate recent episodes
    ├─ Update child state machine
    └─ Output: Current state + duration
    ↓ (every 15 frames = 0.5s, if confidence ≥30%)
[Tier 4: Interpretation]
    ├─ Combine: episodes + state + context
    ├─ Apply Communication Matrix framework
    ├─ Generate hypothesis with alternatives
    └─ Output: Intent, target, description, confidence
    ↓ (if confidence ≥70% and cooldown expired)
[TTS Output]
    └─ Speak interpretation aloud
```

### Coordination Bonus System:

Interpretations gain confidence when multiple signals coordinate:

| Combination | Base Confidence | Coordination Bonus | Final |
|-------------|----------------|-------------------|-------|
| Reach alone | 0.50 | — | 0.50 |
| Reach + gaze alternation | 0.50 | +0.20 | 0.70 |
| Reach + vocalization | 0.50 | +0.15 | 0.65 |
| Reach + gaze + vocal | 0.50 | +0.35 | 0.85 ⭐ |

**Gaze alternation is the KEY indicator** of intentional communication (Level III in Communication Matrix).

---

## 4. Analytics & Insights

### Session-Level Analytics (Short-Term)

Generated at end of each session:

1. **Communication Frequency**
   - Total interpretations per minute
   - Breakdown by intent (request/reject/social/regulate)
   - Peak communication periods

2. **Behavioral Profile**
   - Most common episodes (e.g., "67 vocalizations, 45 reaches")
   - Average episode duration
   - Coordination rate (% of episodes with multiple signals)

3. **State Distribution**
   - Time spent in each state
   - Number of state transitions
   - Longest sustained state

4. **AAC Usage**
   - Button presses by category
   - Child-initiated vs prompted
   - Correlation with interpreted states

5. **Caregiver Feedback**
   - Confirmation rate (% interpretations confirmed)
   - Common rejection patterns
   - Calibration suggestions

---

### Longitudinal Analytics (Long-Term)

Across multiple sessions (requires database):

1. **Developmental Progress**
   - Communication Matrix level progression (1-7)
   - Increase in coordinated signals over time
   - New behavior patterns learned

2. **Pattern Recognition**
   - Recurring episode sequences
   - Time-of-day effects on communication
   - Environmental triggers (detected objects/people)

3. **Behavior Dictionary Growth**
   - Child-specific patterns (e.g., "arm wave at door = wants outside")
   - Confidence evolution with caregiver feedback
   - Rejected patterns archive

4. **Therapist Reports**
   - Exportable session summaries
   - Trend visualizations (weekly/monthly)
   - Goal tracking (IEP objectives)

---

## 5. Implementation Status

### ✅ Complete (Phase 1 & 2)

- [x] Tier 1: All detection models integrated (pose, gaze, face, audio)
- [x] Tier 2: Episode detection engine (reach, gaze_alt, wave, rock, vocal)
- [x] Tier 3: State machine implementation
- [x] Tier 4: Behavior interpreter with Communication Matrix
- [x] Short-horizon storage (in-memory buffers)
- [x] Session recording to JSON files
- [x] Optional TimescaleDB integration with batching
- [x] Session replay engine

### 🚧 In Progress (Phase 3)

- [ ] Behavior dictionary learning from caregiver feedback
- [ ] Confidence calibration based on confirmation rate
- [ ] Longitudinal analytics dashboard
- [ ] Pattern mining across sessions

### 🔮 Planned (Phase 4)

- [ ] Therapist export formats (PDF, CSV)
- [ ] Parent-facing progress reports
- [ ] Multi-child comparison (anonymized)
- [ ] Integration with AAC device logs

---

## 6. Best Practices for Data Interpretation

### For Developers:

1. **Never treat interpretations as ground truth**
   - Always include alternatives
   - Log confidence scores
   - Track caregiver feedback

2. **Respect privacy**
   - All processing is local by default
   - No data leaves device without explicit export
   - Session data encrypted at rest (future)

3. **Graceful degradation**
   - System must work without database
   - In-memory buffers are primary, DB is enhancement
   - Batch writes to avoid blocking inference

4. **Tunable thresholds**
   - Store per-child calibration (e.g., reach_threshold)
   - Allow therapist adjustment via config
   - Auto-calibrate based on feedback

### For Caregivers/Therapists:

1. **Review interpretations critically**
   - Confirm or reject to train the system
   - Provide correct meaning when rejecting
   - Understand alternatives are possibilities, not certainties

2. **Use session summaries for insights**
   - Look for patterns across sessions
   - Note high-communication periods
   - Track which strategies elicit responses

3. **Privacy control**
   - Delete sessions as needed
   - Export only what you need
   - Keep database optional unless longitudinal tracking is required

---

## 7. Future Enhancements

### Short-Term (Next 3 Months)

1. **SQLite Full Support**
   - Replace asyncpg fallback with aiosqlite
   - Enable full analytics without PostgreSQL
   - Single-file database for portability

2. **Enhanced Session Replay**
   - Web-based replay UI with timeline
   - Overlay detections on video
   - Jump to key episodes

3. **Caregiver Feedback Loop**
   - Simple confirm/reject buttons in UI
   - Voice feedback ("yes, that's right")
   - Auto-adjust confidence thresholds

### Medium-Term (3-12 Months)

1. **Pattern Mining Engine**
   - Unsupervised discovery of recurring behaviors
   - Alert caregivers to new patterns
   - Suggest names for behaviors ("arm flappy = happy")

2. **Multi-Modal Fusion Improvements**
   - Better temporal alignment of signals
   - Context-aware interpretation (time of day, location)
   - Person-specific gaze targets (parent vs therapist)

3. **Longitudinal Dashboard**
   - Weekly/monthly trend charts
   - Communication Matrix level tracker
   - Export for IEP meetings

### Long-Term (12+ Months)

1. **Federated Learning**
   - Learn from multiple children without sharing data
   - Improve baseline models while preserving privacy
   - Community-contributed behavior patterns

2. **Predictive Analytics**
   - Anticipate distress before escalation
   - Suggest interventions based on patterns
   - Optimal timing for communication prompts

3. **Integration Ecosystem**
   - AAC device sync (Proloquo, TouchChat)
   - Wearable sensor fusion (Empatica, Apple Watch)
   - Smart home triggers (lights, music based on state)

---

## Appendix A: Database Schema Reference

See `/workspace/config/init.sql` for complete TimescaleDB schema.

**Key Tables:**
- `detections`: Per-frame raw data (hypertable, compressed after 7 days)
- `episodes`: Behavioral episodes (hypertable, compressed after 30 days)
- `interpretations`: Semantic interpretations with feedback
- `sessions`: Session metadata and summaries
- `behavior_dictionary`: Learned child-specific patterns

---

## Appendix B: Configuration Options

**Environment Variables:**
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/cuecatcher  # Optional
REDIS_URL=redis://localhost:6379                                # Optional
SESSION_DIR=/data/sessions                                      # Required
RECORD_VIDEO=false                                              # Optional
BATCH_SIZE=30                                                   # Tunable
```

**Behavior Thresholds (per-child calibration):**
```yaml
# config/behaviors.yaml
reach_threshold: 0.25          # Arm extension (normalized 0-1)
reach_sustain_frames: 15       # Frames to count as reach
gaze_shift_degrees: 15         # Head turn threshold
wave_variance_threshold: 0.005 # Arm wave sensitivity
rock_min_cycles: 2             # Minimum rocking cycles
```

---

## Appendix C: File Structure

```
/workspace/
├── server/
│   ├── recorder.py          # Session recording logic
│   └── main.py              # WebSocket streaming + API
├── inference/
│   ├── pipeline.py          # Main inference orchestrator
│   ├── temporal.py          # Tier 2 & 3 engine
│   ├── interpreter.py       # Tier 4 semantic interpretation
│   ├── pose.py              # Tier 1 pose detection
│   ├── gaze.py              # Tier 1 gaze detection
│   ├── face.py              # Tier 1 face detection
│   └── audio.py             # Tier 1 audio detection
├── config/
│   ├── init.sql             # TimescaleDB schema
│   └── behaviors.yaml       # Tunable thresholds
└── data/
    └── sessions/            # JSON session storage
        └── {session_id}/
            ├── summary.json
            └── button_presses.json
```

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Maintained By:** CueCatcher Development Team
