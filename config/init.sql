-- CueCatcher TimescaleDB schema

-- Behavioral detections (high-frequency, per-frame)
CREATE TABLE IF NOT EXISTS detections (
    time        TIMESTAMPTZ     NOT NULL,
    session_id  UUID            NOT NULL,
    frame_idx   BIGINT          NOT NULL,
    pose        JSONB,          -- 133 keypoints
    gaze        JSONB,          -- head_yaw, head_pitch, eye_gaze_vector
    face        JSONB,          -- action_units, expression
    audio       JSONB,          -- vocalization class, pitch, energy
    objects     JSONB           -- detected objects near child
);
SELECT create_hypertable('detections', 'time', if_not_exists => TRUE);

-- Behavioral episodes (Tier 2: 1-5 second aggregations)
CREATE TABLE IF NOT EXISTS episodes (
    time            TIMESTAMPTZ     NOT NULL,
    session_id      UUID            NOT NULL,
    episode_type    TEXT            NOT NULL,  -- 'reach', 'gaze_alt', 'vocalize', etc.
    duration_ms     INTEGER         NOT NULL,
    confidence      REAL            NOT NULL,
    features        JSONB,
    keyframes       JSONB           -- representative frame indices
);
SELECT create_hypertable('episodes', 'time', if_not_exists => TRUE);

-- Interpretations (what we think the child is communicating)
CREATE TABLE IF NOT EXISTS interpretations (
    time            TIMESTAMPTZ     NOT NULL,
    session_id      UUID            NOT NULL,
    intent          TEXT            NOT NULL,   -- 'request', 'reject', 'social', 'regulate'
    target          TEXT,                       -- 'cup', 'door', 'parent', etc.
    description     TEXT            NOT NULL,   -- human-readable interpretation
    confidence      REAL            NOT NULL,
    comm_level      INTEGER,                   -- Communication Matrix level (1-7)
    evidence        JSONB,                     -- which episodes support this
    spoken          BOOLEAN DEFAULT FALSE,     -- was this spoken aloud via TTS?
    caregiver_feedback TEXT                     -- 'confirmed', 'rejected', NULL
);
SELECT create_hypertable('interpretations', 'time', if_not_exists => TRUE);

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY,
    started_at      TIMESTAMPTZ     NOT NULL,
    ended_at        TIMESTAMPTZ,
    total_frames    BIGINT DEFAULT 0,
    total_episodes  INTEGER DEFAULT 0,
    total_interpretations INTEGER DEFAULT 0,
    notes           TEXT
);

-- Behavior dictionary (learned patterns specific to this child)
CREATE TABLE IF NOT EXISTS behavior_dictionary (
    id              SERIAL PRIMARY KEY,
    pattern_name    TEXT            NOT NULL,   -- e.g., "arm_wave_at_door"
    description     TEXT            NOT NULL,   -- "wants to go outside"
    pattern_data    JSONB           NOT NULL,   -- feature signature
    confidence      REAL            NOT NULL,
    times_confirmed INTEGER DEFAULT 0,
    times_rejected  INTEGER DEFAULT 0,
    first_seen      TIMESTAMPTZ     NOT NULL,
    last_seen       TIMESTAMPTZ     NOT NULL,
    active          BOOLEAN DEFAULT TRUE
);

-- Create useful indices
CREATE INDEX IF NOT EXISTS idx_detections_session ON detections (session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes (session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes (episode_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_interpretations_session ON interpretations (session_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_interpretations_feedback ON interpretations (caregiver_feedback, time DESC);

-- Retention policy: compress old data after 7 days
SELECT add_compression_policy('detections', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('episodes', INTERVAL '30 days', if_not_exists => TRUE);
