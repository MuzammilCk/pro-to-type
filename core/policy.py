"""Policy layer — pure decision functions over WorldState, AgentState, and Memory.

Formalizes the agent's behavioral doctrine:
- Greet decisions (presence, identity, cooldowns)
- Interrupt decisions (barge-in, departure during speech, security revocation)
- Follow-up decisions (revisiting past topics or pending questions)
- Nudge decisions (idle threshold, strict budget)
- Proactive remark decisions (room occupancy, rate limits, hourly budget)

Strictly pure functions: no LLM calls, no network I/O, no timers.
Accepts explicit `now` parameter for 100% deterministic, hermetic unit testing.
"""
from dataclasses import dataclass
from typing import Any
import time


@dataclass(frozen=True)
class PolicyDecision:
    """Represents the outcome of an evaluated policy rule."""
    allowed: bool
    action: str | None = None
    reason: str = ""


# ----------------------------------------------------------------------
# 1. Greeting Policy
# ----------------------------------------------------------------------

def should_greet(
    world_state: Any | None,
    agent_state: Any | None,
    identity: str | None = None,
    last_greeted: float = 0.0,
    cooldown_sec: float = 300.0,
    is_authorized: bool = False,
    now: float | None = None,
) -> PolicyDecision:
    """Evaluate whether ARIA should speak a greeting to a detected visitor."""
    current_time = now if now is not None else time.time()

    # Rule 1: Must have physical presence (faces in view)
    has_faces = False
    if world_state is not None:
        faces = getattr(world_state, "faces", [])
        has_faces = len(faces) > 0
    elif identity is not None:
        has_faces = True

    if not has_faces:
        return PolicyDecision(allowed=False, action=None, reason="no_person_present")

    # Rule 2: Agent busy speaking or in alert
    if agent_state is not None:
        if getattr(agent_state, "mode", "idle") == "alert":
            return PolicyDecision(allowed=False, action=None, reason="agent_in_alert")
        # Do not greet if already actively conversing with this exact person
        active_person = getattr(agent_state, "active_person", None)
        mode = getattr(agent_state, "mode", "idle")
        if mode == "conversing" and active_person and active_person == identity:
            return PolicyDecision(allowed=False, action=None, reason="already_conversing")

    # Rule 3: Identity-level / track cooldown
    if last_greeted > 0 and (current_time - last_greeted < cooldown_sec):
        return PolicyDecision(allowed=False, action=None, reason="greeting_cooldown")

    # Rule 4: Action determines greeting type
    action = "greet_known" if (is_authorized and identity not in (None, "", "unknown")) else "greet_unknown"
    return PolicyDecision(allowed=True, action=action, reason="presence_confirmed")


# ----------------------------------------------------------------------
# 2. Interruption Policy
# ----------------------------------------------------------------------

def should_interrupt(
    agent_state: Any | None,
    event_type: str,
    event_data: dict[str, Any] | None = None,
    is_speaking: bool = False,
) -> PolicyDecision:
    """Evaluate whether an incoming event should interrupt ongoing speech/activity."""
    event_data = event_data or {}

    # Rule 1: User speaks while agent is speaking (full-duplex barge-in)
    if event_type == "USER_UTTERANCE":
        if is_speaking:
            return PolicyDecision(allowed=True, action="barge_in", reason="user_spoke_during_speech")
        return PolicyDecision(allowed=False, action=None, reason="agent_not_speaking")

    # Rule 2: Active interlocutor left during speech or dialogue
    if event_type == "PERSON_LEFT":
        departed_id = event_data.get("identity") or event_data.get("name")
        active_id = getattr(agent_state, "active_person", None) if agent_state else None
        if active_id and departed_id == active_id:
            return PolicyDecision(allowed=True, action="stop_and_reset", reason="active_person_departed")
        return PolicyDecision(allowed=False, action=None, reason="non_active_person_departed")

    # Rule 3: Security revocation mid-conversation
    if event_type in ("REVOKED", "person_unrecognized"):
        active_id = getattr(agent_state, "active_person", None) if agent_state else None
        revoked_id = event_data.get("previous_identity")
        if active_id and revoked_id == active_id:
            return PolicyDecision(allowed=True, action="suspend_auth", reason="security_revocation")
        return PolicyDecision(allowed=False, action=None, reason="revocation_not_active_person")

    # Benign events (vision summary, frame detection) do not interrupt
    return PolicyDecision(allowed=False, action=None, reason="benign_event")


# ----------------------------------------------------------------------
# 3. Follow-up Policy
# ----------------------------------------------------------------------

def should_follow_up(
    agent_state: Any | None,
    memory: Any | None,
    min_turns: int = 1,
) -> PolicyDecision:
    """Evaluate whether the agent should bring up a past topic or unresolved question."""
    if agent_state is None or getattr(agent_state, "mode", "idle") != "conversing":
        return PolicyDecision(allowed=False, action=None, reason="no_active_conversation")

    working = getattr(memory, "working", None)
    turns = getattr(working, "turns", []) if working else []
    if len(turns) < min_turns:
        return PolicyDecision(allowed=False, action=None, reason="insufficient_conversation_turns")

    # Check if there is an unresolved pending question
    pending_q = getattr(working, "pending_question", None) if working else None
    if pending_q:
        return PolicyDecision(allowed=True, action="revisit_question", reason=f"pending_question: {pending_q}")

    # Check if episodic or semantic topics are available for callback
    episodic = getattr(memory, "episodic", None)
    episodes = getattr(episodic, "episodes", []) if episodic else []
    if episodes:
        recent_ep = episodes[-1]
        summary = getattr(recent_ep, "summary", "")
        return PolicyDecision(allowed=True, action="callback_past_topic", reason=f"past_episode: {summary[:40]}")

    semantic = getattr(memory, "semantic", None)
    purpose = getattr(semantic, "purpose", "") if semantic else ""
    if purpose:
        return PolicyDecision(allowed=True, action="callback_purpose", reason=f"purpose: {purpose[:40]}")

    return PolicyDecision(allowed=False, action=None, reason="no_past_context_available")


# ----------------------------------------------------------------------
# 4. Nudge Policy
# ----------------------------------------------------------------------

def should_nudge(
    agent_state: Any | None,
    idle_seconds: float,
    nudges_done: int,
    nudge_threshold_sec: float = 90.0,
    max_nudges: int = 1,
    is_busy: bool = False,
) -> PolicyDecision:
    """Evaluate whether to speak a gentle nudge after sustained conversational silence."""
    if is_busy:
        return PolicyDecision(allowed=False, action=None, reason="agent_busy")

    if nudges_done >= max_nudges:
        return PolicyDecision(allowed=False, action=None, reason="max_nudges_exceeded")

    if idle_seconds < nudge_threshold_sec:
        return PolicyDecision(allowed=False, action=None, reason="idle_threshold_not_met")

    mode = getattr(agent_state, "mode", "idle") if agent_state else "idle"
    if mode not in ("conversing", "idle"):
        return PolicyDecision(allowed=False, action=None, reason="incompatible_mode")

    return PolicyDecision(allowed=True, action="nudge", reason="idle_threshold_exceeded")


# ----------------------------------------------------------------------
# 5. Proactive Remark Policy
# ----------------------------------------------------------------------

def should_proactive_remark(
    world_state: Any | None,
    agent_state: Any | None,
    last_spoke_time: float,
    hour_timestamps: list[float] | None = None,
    min_interval_sec: float = 90.0,
    max_per_hour: int = 6,
    only_when_seen: bool = True,
    is_busy: bool = False,
    now: float | None = None,
) -> PolicyDecision:
    """Evaluate whether ARIA should make a spontaneous, unprompted remark about the room."""
    current_time = now if now is not None else time.time()
    hour_timestamps = hour_timestamps or []

    # Rule 1: Stay silent if agent is currently busy or actively conversing
    if is_busy:
        return PolicyDecision(allowed=False, action=None, reason="agent_busy")

    if agent_state is not None and getattr(agent_state, "mode", "idle") == "conversing":
        return PolicyDecision(allowed=False, action=None, reason="currently_conversing")

    # Rule 2: Stay silent if nobody is in view
    if only_when_seen:
        faces = getattr(world_state, "faces", []) if world_state else []
        if len(faces) == 0:
            return PolicyDecision(allowed=False, action=None, reason="no_one_in_view")

    # Rule 3: Minimum interval between remarks
    if current_time - last_spoke_time < min_interval_sec:
        return PolicyDecision(allowed=False, action=None, reason="min_interval_not_met")

    # Rule 4: Hourly frequency cap
    recent_in_hour = [t for t in hour_timestamps if current_time - t < 3600.0]
    if len(recent_in_hour) >= max_per_hour:
        return PolicyDecision(allowed=False, action=None, reason="hourly_limit_reached")

    return PolicyDecision(allowed=True, action="proactive_remark", reason="conditions_satisfied")
