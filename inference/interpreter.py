"""
CueCatcher Behavior Interpreter

Translates raw detections into probabilistic communicative interpretations.
Based on the Communication Matrix framework (Rowland, 2004).

Core principle: ALL behavior is potential communication.
Core safeguard: ALL interpretations are hypotheses, never facts.
"""

import time
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from loguru import logger


@dataclass
class Interpretation:
    """A probabilistic communicative interpretation."""
    id: str = ""
    timestamp: float = 0.0
    intent: str = ""            # request, reject, social, regulate, explore
    target: Optional[str] = None
    description: str = ""       # human-readable
    spoken_text: str = ""       # what TTS should say (shorter, simpler)
    confidence: float = 0.0
    comm_level: int = 1         # Communication Matrix level (1-7)
    evidence: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)  # other possible meanings
    should_speak: bool = False

    def to_dict(self) -> dict:
        return self.__dict__


class BehaviorInterpreter:
    """
    Interprets behavioral detections as communicative acts.

    Uses a sliding window of recent detections to identify
    coordinated multi-signal communication attempts.
    """

    def __init__(self):
        self._history: deque = deque(maxlen=150)  # ~5 seconds at 30fps
        self._recent_interpretations: deque = deque(maxlen=20)
        self._last_spoken_time: float = 0.0
        self._cooldown_seconds: float = 3.0

        # Learned behavior dictionary (grows with caregiver feedback)
        self._dictionary: dict = {}

        # Feedback counts for confidence calibration
        self._confirmed: int = 0
        self._rejected: int = 0

    def interpret(self, detections: dict, frame_idx: int) -> Optional[dict]:
        """
        Given current frame detections, decide if the child is
        communicating and what they might be trying to say.
        """
        self._history.append(detections)

        # Don't interpret every frame — check every 15 frames (~0.5s)
        if frame_idx % 15 != 0:
            return None

        interp = self._analyze_current_state(detections)

        if interp and interp.confidence >= 0.30:
            # Decide if this should be spoken
            now = time.time()
            interp.should_speak = (
                interp.confidence >= 0.70
                and (now - self._last_spoken_time) >= self._cooldown_seconds
            )
            if interp.should_speak:
                self._last_spoken_time = now

            self._recent_interpretations.append(interp)
            return interp.to_dict()

        return None

    def _analyze_current_state(self, det: dict) -> Optional[Interpretation]:
        """Analyze the current detection context for communicative signals."""

        signals = []

        # ── Check for action-level signals ──
        actions = det.get("nearby_objects", [])
        for action in actions:
            if isinstance(action, dict):
                signals.append(action)

        # ── Check gaze direction ──
        head_yaw = det.get("head_yaw", 0)
        head_pitch = det.get("head_pitch", 0)

        # Gaze alternation detection: has the child looked at different
        # targets in the last 2 seconds? This is the KEY indicator of
        # intentional communication (Level III).
        gaze_alternation = self._detect_gaze_alternation()

        # ── Check vocalization ──
        is_vocalizing = det.get("is_vocalization", False)
        voc_class = det.get("vocalization_class", "silence")

        # ── Check facial expression ──
        expression = det.get("expression", "neutral")

        # ── Build interpretation from combined signals ──

        # PRIORITY 1: Distress / rejection
        if voc_class == "distress_cry" and det.get("vocalization_confidence", 0) > 0.5:
            return Interpretation(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                intent="reject" if self._is_rejecting(det) else "regulate",
                description="The child appears distressed — crying detected",
                spoken_text="I think she's upset about something",
                confidence=0.75,
                comm_level=1,
                evidence=["distress_vocalization"],
                alternatives=["pain or discomfort", "frustration", "sensory overload"],
            )

        # PRIORITY 2: Request with coordinated signals
        reaching_signals = [s for s in signals if "reaching" in s.get("action", "")]
        if reaching_signals:
            best_reach = max(reaching_signals, key=lambda s: s.get("confidence", 0))
            coordination_bonus = 0.0

            evidence = [best_reach["action"]]

            # Gaze alternation + reaching = strong intentional request (Level III)
            if gaze_alternation:
                coordination_bonus += 0.20
                evidence.append("gaze_alternation")

            # Vocalization + reaching = even stronger
            if is_vocalizing and voc_class in ("attention_seeking", "babble"):
                coordination_bonus += 0.15
                evidence.append(f"vocalization_{voc_class}")

            confidence = min(1.0, best_reach.get("confidence", 0.5) + coordination_bonus)
            comm_level = 3 if gaze_alternation else 2

            return Interpretation(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                intent="request",
                target=best_reach.get("direction", "something"),
                description=f"The child may be reaching toward something ({best_reach['action']})",
                spoken_text="She might be reaching for something",
                confidence=confidence,
                comm_level=comm_level,
                evidence=evidence,
                alternatives=[
                    "exploring/touching",
                    "pointing at something interesting",
                    "stretching",
                ],
            )

        # PRIORITY 3: Arms up = request pickup
        arms_up = [s for s in signals if s.get("action") == "arms_up"]
        if arms_up:
            return Interpretation(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                intent="request",
                target="pickup",
                description="The child has both arms raised — may want to be picked up",
                spoken_text="She might want to be picked up",
                confidence=0.85,
                comm_level=3,
                evidence=["arms_up"],
                alternatives=["celebrating", "stretching"],
            )

        # PRIORITY 4: Arm waving
        waving = [s for s in signals if s.get("action") == "arm_waving"]
        if waving:
            best = waving[0]
            # Context matters: waving + vocalization = excitement/attention
            # waving alone = possibly stimming (self-regulation)
            if is_vocalizing:
                return Interpretation(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    intent="social",
                    description="Arm waving with vocalization — may be expressing excitement or trying to get attention",
                    spoken_text="She seems excited about something",
                    confidence=min(1.0, best.get("confidence", 0.5) + 0.1),
                    comm_level=2,
                    evidence=["arm_waving", f"vocalization_{voc_class}"],
                    alternatives=[
                        "self-stimulation (happy stimming)",
                        "sensory seeking",
                        "expressing joy",
                    ],
                )
            else:
                return Interpretation(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    intent="regulate",
                    description="Arm waving detected — could be stimming for self-regulation or expressing emotion",
                    spoken_text="She's moving her arms a lot",
                    confidence=best.get("confidence", 0.5),
                    comm_level=1,
                    evidence=["arm_waving"],
                    alternatives=[
                        "excitement",
                        "anxiety",
                        "sensory processing",
                        "not communicative — just movement",
                    ],
                )

        # PRIORITY 5: Rocking
        rocking = [s for s in signals if s.get("action") == "rocking"]
        if rocking:
            return Interpretation(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                intent="regulate",
                description="Rocking motion detected — likely self-soothing or sensory regulation",
                spoken_text="She seems to be self-soothing",
                confidence=0.55,
                comm_level=1,
                evidence=["rocking"],
                alternatives=[
                    "contentment (rhythmic comfort)",
                    "overwhelm (calming strategy)",
                    "not communicative",
                ],
            )

        # PRIORITY 6: Vocalization alone
        if is_vocalizing and voc_class == "attention_seeking":
            return Interpretation(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                intent="social",
                description="Vocalization that may be directed at getting attention",
                spoken_text="She's making sounds — might want attention",
                confidence=0.45,
                comm_level=2,
                evidence=[f"vocalization_{voc_class}"],
                alternatives=[
                    "vocal play / babbling practice",
                    "expressing a feeling",
                    "not directed at anyone",
                ],
            )

        return None

    def _detect_gaze_alternation(self) -> bool:
        """
        Check if the child has looked at different targets in the last 2 seconds.
        Gaze alternation (looking at object, then at person, then back) is the
        hallmark of intentional communication (Level III in Communication Matrix).
        """
        if len(self._history) < 30:
            return False

        recent = list(self._history)[-60:]  # last 2 seconds
        yaws = [d.get("head_yaw", 0) for d in recent]

        if not yaws:
            return False

        # Look for significant direction changes
        import numpy as np
        yaw_arr = np.array(yaws)
        yaw_diff = np.abs(np.diff(yaw_arr))
        large_shifts = np.sum(yaw_diff > 15)  # >15 degree head turns

        # Two or more large shifts in 2 seconds suggests alternation
        return large_shifts >= 2

    def _is_rejecting(self, det: dict) -> bool:
        """Check if signals suggest rejection rather than general distress."""
        actions = det.get("nearby_objects", [])
        has_pushing = any(
            "push" in a.get("action", "") for a in actions if isinstance(a, dict)
        )
        head_turned = abs(det.get("head_yaw", 0)) > 45
        return has_pushing or head_turned

    def record_feedback(self, interpretation_id: str, action: str, correct_meaning: str = None):
        """
        Record caregiver feedback on an interpretation.
        This trains the system over time.
        """
        if action == "confirmed":
            self._confirmed += 1
            logger.info(f"✅ Interpretation confirmed: {interpretation_id}")
        elif action == "rejected":
            self._rejected += 1
            if correct_meaning:
                logger.info(f"❌ Interpretation rejected: {interpretation_id} → actual: {correct_meaning}")
            else:
                logger.info(f"❌ Interpretation rejected: {interpretation_id}")

        # TODO: Update behavior dictionary with confirmed patterns
        # TODO: Adjust confidence thresholds based on confirmation rate
