"""
CueCatcher Temporal Analysis Engine

Three tiers of temporal aggregation that progressively build meaning:

  Tier 1 (per-frame, 0ms):     Raw keypoints, AUs, gaze vectors, audio events
  Tier 2 (1-5s episodes):      Behavioral episodes — reaching, gaze alternation, vocalization bursts
  Tier 3 (30s-5min states):    Child's overall state — calm, distressed, communicating, engaged

The key insight: a single frame of reaching means nothing.
Two seconds of sustained reaching + gaze alternation + vocalization = probable request.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np
from loguru import logger


# ── Tier 2: Behavioral Episodes ────────────────────────────────

class EpisodeType(str, Enum):
    REACH = "reach"
    GAZE_ALTERNATION = "gaze_alternation"
    ARM_WAVE = "arm_wave"
    ARMS_UP = "arms_up"
    ROCKING = "rocking"
    HEAD_TURN = "head_turn"
    VOCALIZATION_BURST = "vocalization_burst"
    DISTRESS_CRY = "distress_cry"
    SMILE = "smile"
    WITHDRAWAL = "withdrawal"        # going still, turning away
    APPROACH = "approach"             # moving toward something/someone
    HAND_LEADING = "hand_leading"     # taking adult's hand
    OBJECT_GIVE = "object_give"       # handing object to adult
    IMITATION = "imitation"           # copying an action (highest-signal for development)


@dataclass
class Episode:
    """A detected behavioral episode spanning 1-5 seconds."""
    type: EpisodeType
    start_frame: int
    end_frame: int
    start_time: float
    duration_ms: int
    confidence: float
    features: dict = field(default_factory=dict)
    comm_relevance: float = 0.0    # how likely this is communicative (0-1)
    comm_function: str = ""        # request, reject, social, regulate

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "features": self.features,
            "comm_relevance": self.comm_relevance,
            "comm_function": self.comm_function,
        }


# ── Tier 3: Child State ───────────────────────────────────────

class ChildState(str, Enum):
    IDLE = "idle"                   # no significant activity
    ATTENDING = "attending"         # focused on something
    COMMUNICATING = "communicating" # active communication attempt
    DISTRESSED = "distressed"       # crying, agitated
    ENGAGED = "engaged"             # in a social interaction
    REGULATING = "regulating"       # stimming, rocking, self-soothing
    TRANSITIONING = "transitioning" # shifting between states
    WITHDRAWN = "withdrawn"         # shutdown, disengaged


@dataclass
class StateSnapshot:
    """Current child state with history."""
    state: ChildState
    since: float                    # timestamp when state began
    confidence: float
    recent_episodes: list = field(default_factory=list)
    transition_from: Optional[str] = None

    def duration_seconds(self) -> float:
        return time.time() - self.since


# ── Temporal Analysis Engine ──────────────────────────────────

class TemporalEngine:
    """
    Processes a stream of per-frame detections into behavioral episodes
    and tracks the child's overall state.
    """

    def __init__(self, fps: int = 30):
        self.fps = fps

        # Tier 1: Frame buffer (ring buffer of raw detections)
        self._frame_buffer: deque = deque(maxlen=fps * 10)  # 10 seconds

        # Tier 2: Episode detection state
        self._active_episodes: dict[str, dict] = {}  # type → tracking state
        self._completed_episodes: deque = deque(maxlen=100)
        self._episode_window = fps * 5  # 5 second analysis window

        # Tier 3: State machine
        self._current_state = StateSnapshot(
            state=ChildState.IDLE,
            since=time.time(),
            confidence=0.5,
        )
        self._state_history: deque = deque(maxlen=50)

        # Thresholds (tunable per child)
        self.reach_threshold = 0.25        # arm extension (normalized)
        self.reach_sustain_frames = 15     # 0.5s sustained reach
        self.gaze_shift_degrees = 15       # head turn threshold
        self.gaze_alt_min_shifts = 2       # minimum shifts for alternation
        self.wave_variance_threshold = 0.005
        self.rock_min_cycles = 2
        self.voc_burst_min_frames = 10     # 0.33s of vocalization

    def update(self, detection: dict, frame_idx: int) -> dict:
        """
        Process a new frame detection.

        Returns a dict with:
          - new_episodes: list of newly completed episodes
          - state: current child state
          - state_changed: bool
        """
        self._frame_buffer.append({
            "detection": detection,
            "frame_idx": frame_idx,
            "time": time.time(),
        })

        # Tier 2: Detect episodes every 5 frames (~6Hz)
        new_episodes = []
        if frame_idx % 5 == 0 and len(self._frame_buffer) >= self.fps:
            new_episodes = self._detect_episodes(frame_idx)
            for ep in new_episodes:
                self._completed_episodes.append(ep)

        # Tier 3: Update state every 15 frames (~2Hz)
        state_changed = False
        if frame_idx % 15 == 0:
            state_changed = self._update_state()

        return {
            "new_episodes": [e.to_dict() for e in new_episodes],
            "state": self._current_state.state.value,
            "state_confidence": self._current_state.confidence,
            "state_duration_s": self._current_state.duration_seconds(),
            "state_changed": state_changed,
        }

    # ── Tier 2: Episode Detection ──────────────────────────────

    def _detect_episodes(self, current_frame: int) -> list[Episode]:
        """Analyze recent frame buffer for behavioral episodes."""
        episodes = []
        frames = list(self._frame_buffer)

        if len(frames) < self.fps:
            return episodes

        # Get recent N frames for analysis
        window = frames[-self._episode_window:]

        # Extract time-series signals
        poses = []
        head_yaws = []
        vocalizing = []
        expressions = []

        for f in window:
            d = f["detection"]
            poses.append(d.get("pose_keypoints"))
            head_yaws.append(d.get("head_yaw", 0))
            vocalizing.append(d.get("is_vocalization", False))
            expressions.append(d.get("expression", "neutral"))

        # ── Detect: Reaching ──
        ep = self._detect_reaching(window, poses, current_frame)
        if ep:
            episodes.append(ep)

        # ── Detect: Gaze Alternation ──
        ep = self._detect_gaze_alternation(window, head_yaws, current_frame)
        if ep:
            episodes.append(ep)

        # ── Detect: Arm Waving ──
        ep = self._detect_arm_waving(window, poses, current_frame)
        if ep:
            episodes.append(ep)

        # ── Detect: Vocalization Burst ──
        ep = self._detect_vocalization_burst(window, vocalizing, current_frame)
        if ep:
            episodes.append(ep)

        # ── Detect: Rocking ──
        ep = self._detect_rocking(window, poses, current_frame)
        if ep:
            episodes.append(ep)

        # ── Detect: Withdrawal ──
        ep = self._detect_withdrawal(window, poses, head_yaws, current_frame)
        if ep:
            episodes.append(ep)

        return episodes

    def _detect_reaching(self, window, poses, current_frame) -> Optional[Episode]:
        """Detect sustained arm extension."""
        if not poses or poses[-1] is None:
            return None

        # Check last 1 second of frames
        recent_poses = [p for p in poses[-self.fps:] if p is not None]
        if len(recent_poses) < self.reach_sustain_frames:
            return None

        reach_frames = 0
        max_extension = 0
        reaching_side = None

        for kp in recent_poses:
            kp = np.array(kp) if not isinstance(kp, np.ndarray) else kp
            if kp.shape[0] < 17:
                continue

            # MediaPipe: 11=L shoulder, 15=L wrist, 12=R shoulder, 16=R wrist
            for side, (sh_idx, wr_idx) in [("left", (11, 15)), ("right", (12, 16))]:
                shoulder = kp[sh_idx, :2]
                wrist = kp[wr_idx, :2]
                dist = np.linalg.norm(wrist - shoulder)
                if dist > self.reach_threshold:
                    reach_frames += 1
                    if dist > max_extension:
                        max_extension = dist
                        reaching_side = side
                    break

        if reach_frames >= self.reach_sustain_frames and reaching_side:
            # Check if this is a new reach (not already detected recently)
            if self._is_duplicate_episode(EpisodeType.REACH, current_frame):
                return None

            return Episode(
                type=EpisodeType.REACH,
                start_frame=current_frame - reach_frames,
                end_frame=current_frame,
                start_time=time.time() - reach_frames / self.fps,
                duration_ms=int(reach_frames / self.fps * 1000),
                confidence=min(1.0, max_extension / 0.4),
                features={"side": reaching_side, "extension": float(max_extension)},
                comm_relevance=0.7,
                comm_function="request",
            )

        return None

    def _detect_gaze_alternation(self, window, head_yaws, current_frame) -> Optional[Episode]:
        """
        Detect gaze alternation — the hallmark of intentional communication.
        Child looks at object, then at person, then back.
        """
        if len(head_yaws) < self.fps * 2:
            return None

        recent = np.array(head_yaws[-self.fps * 2:])  # last 2 seconds
        if np.all(recent == 0):
            return None

        # Count significant direction changes
        diffs = np.abs(np.diff(recent))
        large_shifts = np.where(diffs > self.gaze_shift_degrees)[0]

        if len(large_shifts) >= self.gaze_alt_min_shifts:
            if self._is_duplicate_episode(EpisodeType.GAZE_ALTERNATION, current_frame):
                return None

            # Calculate the range of gaze directions
            gaze_range = float(np.max(recent) - np.min(recent))

            return Episode(
                type=EpisodeType.GAZE_ALTERNATION,
                start_frame=current_frame - self.fps * 2,
                end_frame=current_frame,
                start_time=time.time() - 2.0,
                duration_ms=2000,
                confidence=min(1.0, len(large_shifts) / 4),
                features={
                    "shift_count": int(len(large_shifts)),
                    "gaze_range_degrees": gaze_range,
                },
                comm_relevance=0.9,  # highest communicative signal
                comm_function="social",
            )

        return None

    def _detect_arm_waving(self, window, poses, current_frame) -> Optional[Episode]:
        """Detect rapid bilateral arm movement."""
        recent_poses = [p for p in poses[-self.fps:] if p is not None]
        if len(recent_poses) < self.fps // 2:
            return None

        kps = np.array(recent_poses)
        if kps.shape[1] < 17:
            return None

        wrists = kps[:, [15, 16], :2]
        variance = np.var(wrists, axis=0).mean()

        if variance > self.wave_variance_threshold:
            if self._is_duplicate_episode(EpisodeType.ARM_WAVE, current_frame):
                return None

            return Episode(
                type=EpisodeType.ARM_WAVE,
                start_frame=current_frame - len(recent_poses),
                end_frame=current_frame,
                start_time=time.time() - len(recent_poses) / self.fps,
                duration_ms=int(len(recent_poses) / self.fps * 1000),
                confidence=min(1.0, variance / 0.02),
                features={"variance": float(variance)},
                comm_relevance=0.5,
                comm_function="regulate",
            )

        return None

    def _detect_vocalization_burst(self, window, vocalizing, current_frame) -> Optional[Episode]:
        """Detect a sustained vocalization (not just a brief sound)."""
        recent = vocalizing[-self.fps:]
        if len(recent) < self.voc_burst_min_frames:
            return None

        # Count consecutive vocalization frames
        consecutive = 0
        max_consecutive = 0
        for v in recent:
            if v:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if max_consecutive >= self.voc_burst_min_frames:
            if self._is_duplicate_episode(EpisodeType.VOCALIZATION_BURST, current_frame):
                return None

            # Get the dominant vocalization class
            voc_classes = []
            for f in window[-self.fps:]:
                d = f["detection"]
                if d.get("is_vocalization"):
                    voc_classes.append(d.get("vocalization_class", "unknown"))

            dominant_class = max(set(voc_classes), key=voc_classes.count) if voc_classes else "unknown"

            comm_fn = {
                "distress_cry": "reject",
                "attention_seeking": "request",
                "pleasure": "social",
                "babble": "social",
            }.get(dominant_class, "regulate")

            return Episode(
                type=EpisodeType.VOCALIZATION_BURST if dominant_class != "distress_cry" else EpisodeType.DISTRESS_CRY,
                start_frame=current_frame - max_consecutive,
                end_frame=current_frame,
                start_time=time.time() - max_consecutive / self.fps,
                duration_ms=int(max_consecutive / self.fps * 1000),
                confidence=0.65,
                features={
                    "duration_frames": max_consecutive,
                    "dominant_class": dominant_class,
                },
                comm_relevance=0.6,
                comm_function=comm_fn,
            )

        return None

    def _detect_rocking(self, window, poses, current_frame) -> Optional[Episode]:
        """Detect periodic body oscillation."""
        recent_poses = [p for p in poses[-self.fps * 2:] if p is not None]
        if len(recent_poses) < self.fps:
            return None

        kps = np.array(recent_poses)
        if kps.shape[1] < 25:
            return None

        # Track hip center (midpoint of landmarks 23, 24)
        hip_center = (kps[:, 23, :2] + kps[:, 24, :2]) / 2
        hip_var = np.var(hip_center, axis=0).sum()

        if hip_var > 0.001:
            centered = hip_center[:, 0] - hip_center[:, 0].mean()
            zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)

            if zero_crossings >= self.rock_min_cycles * 2:
                if self._is_duplicate_episode(EpisodeType.ROCKING, current_frame):
                    return None

                return Episode(
                    type=EpisodeType.ROCKING,
                    start_frame=current_frame - len(recent_poses),
                    end_frame=current_frame,
                    start_time=time.time() - len(recent_poses) / self.fps,
                    duration_ms=int(len(recent_poses) / self.fps * 1000),
                    confidence=min(1.0, zero_crossings / 8),
                    features={
                        "cycles": int(zero_crossings // 2),
                        "amplitude": float(np.std(centered)),
                    },
                    comm_relevance=0.3,
                    comm_function="regulate",
                )

        return None

    def _detect_withdrawal(self, window, poses, head_yaws, current_frame) -> Optional[Episode]:
        """Detect stillness + head turn away = possible shutdown / rejection."""
        recent_poses = [p for p in poses[-self.fps:] if p is not None]
        if len(recent_poses) < self.fps // 2:
            return None

        kps = np.array(recent_poses)
        if kps.shape[1] < 17:
            return None

        # Very low body movement
        body_var = np.var(kps[:, :17, :2], axis=0).mean()
        head_turned = abs(np.mean(head_yaws[-self.fps:])) > 30

        if body_var < 0.0005 and head_turned:
            if self._is_duplicate_episode(EpisodeType.WITHDRAWAL, current_frame):
                return None

            return Episode(
                type=EpisodeType.WITHDRAWAL,
                start_frame=current_frame - self.fps,
                end_frame=current_frame,
                start_time=time.time() - 1.0,
                duration_ms=1000,
                confidence=0.55,
                features={"body_movement": float(body_var), "head_turned": head_turned},
                comm_relevance=0.6,
                comm_function="reject",
            )

        return None

    def _is_duplicate_episode(self, ep_type: EpisodeType, current_frame: int) -> bool:
        """Check if we've already detected this episode type recently."""
        cooldown = self.fps * 2  # 2 second cooldown per episode type
        for ep in reversed(list(self._completed_episodes)):
            if ep.type == ep_type and (current_frame - ep.end_frame) < cooldown:
                return True
        return False

    # ── Tier 3: State Machine ──────────────────────────────────

    def _update_state(self) -> bool:
        """
        Update the child's overall behavioral state based on recent episodes.
        Returns True if state changed.
        """
        # Gather episodes from last 30 seconds
        cutoff = time.time() - 30
        recent = [ep for ep in self._completed_episodes if ep.start_time > cutoff]

        old_state = self._current_state.state

        # State determination logic (priority-ordered)

        # DISTRESSED: recent distress cry
        if any(ep.type == EpisodeType.DISTRESS_CRY for ep in recent[-5:]):
            new_state = ChildState.DISTRESSED
            confidence = 0.85

        # COMMUNICATING: coordinated multi-signal behavior
        elif self._has_coordinated_signals(recent[-10:]):
            new_state = ChildState.COMMUNICATING
            confidence = 0.75

        # REGULATING: stimming behaviors
        elif any(ep.type in (EpisodeType.ROCKING, EpisodeType.ARM_WAVE)
                 and ep.comm_function == "regulate" for ep in recent[-5:]):
            new_state = ChildState.REGULATING
            confidence = 0.65

        # ATTENDING: gaze focused, some reaching
        elif any(ep.type in (EpisodeType.REACH, EpisodeType.GAZE_ALTERNATION)
                 for ep in recent[-5:]):
            new_state = ChildState.ATTENDING
            confidence = 0.60

        # WITHDRAWN: stillness + turned away
        elif any(ep.type == EpisodeType.WITHDRAWAL for ep in recent[-3:]):
            new_state = ChildState.WITHDRAWN
            confidence = 0.55

        # IDLE: no significant episodes
        else:
            new_state = ChildState.IDLE
            confidence = 0.50

        # Apply state change with hysteresis (don't flicker)
        if new_state != old_state:
            # Require sustained signal: don't change from a "strong" state
            # to "idle" without a transition period
            if old_state in (ChildState.COMMUNICATING, ChildState.DISTRESSED):
                if new_state == ChildState.IDLE:
                    new_state = ChildState.TRANSITIONING
                    confidence = 0.40

            self._state_history.append(self._current_state)
            self._current_state = StateSnapshot(
                state=new_state,
                since=time.time(),
                confidence=confidence,
                recent_episodes=[ep.to_dict() for ep in recent[-5:]],
                transition_from=old_state.value,
            )
            return True

        # Update confidence even if state didn't change
        self._current_state.confidence = confidence
        return False

    def _has_coordinated_signals(self, recent_episodes: list[Episode]) -> bool:
        """
        Check for coordinated multi-modal signals — the hallmark of
        intentional communication (Communication Matrix Level III+).

        Coordination = two or more different signal types occurring
        within a short time window.
        """
        if len(recent_episodes) < 2:
            return False

        # Look for episodes within 3 seconds of each other
        types_seen = set()
        for ep in recent_episodes:
            if ep.comm_relevance > 0.5:
                types_seen.add(ep.type)

        # Coordinated = 2+ different high-relevance episode types
        high_relevance_types = {
            EpisodeType.REACH,
            EpisodeType.GAZE_ALTERNATION,
            EpisodeType.VOCALIZATION_BURST,
            EpisodeType.ARMS_UP,
            EpisodeType.HAND_LEADING,
        }

        return len(types_seen & high_relevance_types) >= 2

    # ── Public API ─────────────────────────────────────────────

    @property
    def current_state(self) -> StateSnapshot:
        return self._current_state

    @property
    def recent_episodes(self) -> list[Episode]:
        return list(self._completed_episodes)

    def get_communication_summary(self, seconds: int = 300) -> dict:
        """Get a summary of communication activity over the last N seconds."""
        cutoff = time.time() - seconds
        episodes = [ep for ep in self._completed_episodes if ep.start_time > cutoff]

        by_function = {}
        for ep in episodes:
            fn = ep.comm_function or "unknown"
            by_function[fn] = by_function.get(fn, 0) + 1

        by_type = {}
        for ep in episodes:
            by_type[ep.type.value] = by_type.get(ep.type.value, 0) + 1

        high_confidence = [ep for ep in episodes if ep.confidence > 0.7]
        communicative = [ep for ep in episodes if ep.comm_relevance > 0.5]

        return {
            "period_seconds": seconds,
            "total_episodes": len(episodes),
            "communicative_episodes": len(communicative),
            "high_confidence_episodes": len(high_confidence),
            "by_function": by_function,
            "by_type": by_type,
            "state_changes": len(self._state_history),
            "current_state": self._current_state.state.value,
        }
