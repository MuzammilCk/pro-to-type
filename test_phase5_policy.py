"""Test suite for Phase 5 — Policy Layer.

Verifies pure decision functions in policy.py:
1. should_greet: presence, agent state, identity, and cooldown rules.
2. should_interrupt: full-duplex barge-in, active person departure, security revocation.
3. should_follow_up: active conversation requirement, pending questions, and past topic callbacks.
4. should_nudge: idle duration threshold and strict one-shot budget.
5. should_proactive_remark: occupancy gating, conversation quietness, rate limits, hourly budget.

100% hermetic: zero network, zero LLM, zero hardware, deterministic timestamps.
"""
from dataclasses import dataclass, field
import pytest

from core.policy import (
    PolicyDecision,
    should_greet,
    should_interrupt,
    should_follow_up,
    should_nudge,
    should_proactive_remark,
)


# ----------------------------------------------------------------------
# Test Doubles / Data Fixtures
# ----------------------------------------------------------------------

@dataclass
class DummyWorldState:
    faces: list[dict] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)


@dataclass
class DummyAgentState:
    mode: str = "idle"                        # "idle", "conversing", "alert"
    active_person: str | None = None
    active_goal: str | None = None
    last_activity: float = 1000.0


class DummyWorkingMemory:
    def __init__(self, turns=None, pending_question=None):
        self.turns = list(turns or [])
        self.pending_question = pending_question


class DummyEpisodicMemory:
    def __init__(self, episodes=None):
        self.episodes = list(episodes or [])


class DummySemanticMemory:
    def __init__(self, purpose="", name="Alice"):
        self.purpose = purpose
        self.name = name


class DummyPersonMemory:
    def __init__(self, working=None, episodic=None, semantic=None):
        self.working = working or DummyWorkingMemory()
        self.episodic = episodic or DummyEpisodicMemory()
        self.semantic = semantic or DummySemanticMemory()


# ----------------------------------------------------------------------
# 1. should_greet Unit Tests
# ----------------------------------------------------------------------

def test_should_greet_no_person_present():
    ws = DummyWorldState(faces=[])
    decision = should_greet(ws, DummyAgentState(), identity=None)
    assert decision.allowed is False
    assert decision.reason == "no_person_present"


def test_should_greet_agent_in_alert():
    ws = DummyWorldState(faces=[{"track_id": "face_0"}])
    astate = DummyAgentState(mode="alert")
    decision = should_greet(ws, astate, identity="Alice")
    assert decision.allowed is False
    assert decision.reason == "agent_in_alert"


def test_should_greet_already_conversing_with_same_person():
    ws = DummyWorldState(faces=[{"track_id": "face_0"}])
    astate = DummyAgentState(mode="conversing", active_person="Alice")
    decision = should_greet(ws, astate, identity="Alice")
    assert decision.allowed is False
    assert decision.reason == "already_conversing"


def test_should_greet_respects_cooldown():
    ws = DummyWorldState(faces=[{"track_id": "face_0"}])
    astate = DummyAgentState(mode="idle")
    decision = should_greet(ws, astate, identity="Alice", last_greeted=1000.0, cooldown_sec=300.0, now=1100.0)
    assert decision.allowed is False
    assert decision.reason == "greeting_cooldown"


def test_should_greet_known_and_unknown_actions():
    ws = DummyWorldState(faces=[{"track_id": "face_0"}])
    astate = DummyAgentState(mode="idle")

    # Known authorized person
    dec_known = should_greet(ws, astate, identity="Bob", is_authorized=True, now=2000.0)
    assert dec_known.allowed is True
    assert dec_known.action == "greet_known"
    assert dec_known.reason == "presence_confirmed"

    # Unknown visitor
    dec_unknown = should_greet(ws, astate, identity="unknown", is_authorized=False, now=2000.0)
    assert dec_unknown.allowed is True
    assert dec_unknown.action == "greet_unknown"


# ----------------------------------------------------------------------
# 2. should_interrupt Unit Tests
# ----------------------------------------------------------------------

def test_should_interrupt_barge_in_when_speaking():
    astate = DummyAgentState(mode="conversing", active_person="Alice")

    # User speaks while agent is speaking -> barge in
    dec_speak = should_interrupt(astate, "USER_UTTERANCE", is_speaking=True)
    assert dec_speak.allowed is True
    assert dec_speak.action == "barge_in"

    # User speaks while agent is not speaking -> no interruption
    dec_listen = should_interrupt(astate, "USER_UTTERANCE", is_speaking=False)
    assert dec_listen.allowed is False


def test_should_interrupt_active_person_left():
    astate = DummyAgentState(mode="conversing", active_person="Alice")

    # Active person departed -> stop and reset
    dec_left = should_interrupt(astate, "PERSON_LEFT", event_data={"identity": "Alice"}, is_speaking=True)
    assert dec_left.allowed is True
    assert dec_left.action == "stop_and_reset"

    # Other non-active person departed -> do not interrupt
    dec_other = should_interrupt(astate, "PERSON_LEFT", event_data={"identity": "Bob"}, is_speaking=True)
    assert dec_other.allowed is False


def test_should_interrupt_security_revocation():
    astate = DummyAgentState(mode="conversing", active_person="Alice")

    dec_rev = should_interrupt(astate, "person_unrecognized", event_data={"previous_identity": "Alice"})
    assert dec_rev.allowed is True
    assert dec_rev.action == "suspend_auth"


def test_should_interrupt_benign_event():
    astate = DummyAgentState(mode="conversing", active_person="Alice")
    dec = should_interrupt(astate, "frame_summary", is_speaking=True)
    assert dec.allowed is False


# ----------------------------------------------------------------------
# 3. should_follow_up Unit Tests
# ----------------------------------------------------------------------

def test_should_follow_up_requires_active_conversation():
    astate = DummyAgentState(mode="idle")
    mem = DummyPersonMemory()
    dec = should_follow_up(astate, mem)
    assert dec.allowed is False
    assert dec.reason == "no_active_conversation"


def test_should_follow_up_requires_minimum_turns():
    astate = DummyAgentState(mode="conversing", active_person="Alice")
    mem = DummyPersonMemory(working=DummyWorkingMemory(turns=[]))
    dec = should_follow_up(astate, mem, min_turns=1)
    assert dec.allowed is False
    assert dec.reason == "insufficient_conversation_turns"


def test_should_follow_up_revisit_pending_question():
    astate = DummyAgentState(mode="conversing", active_person="Alice")
    wm = DummyWorkingMemory(turns=[{"role": "user", "content": "hi"}], pending_question="Which model do you use?")
    mem = DummyPersonMemory(working=wm)
    dec = should_follow_up(astate, mem)
    assert dec.allowed is True
    assert dec.action == "revisit_question"


def test_should_follow_up_past_topic_callback():
    astate = DummyAgentState(mode="conversing", active_person="Alice")
    wm = DummyWorkingMemory(turns=[{"role": "user", "content": "hi"}])
    @dataclass
    class Ep:
        summary: str = "Trained YOLOv5 model on custom robotics dataset"
    em = DummyEpisodicMemory(episodes=[Ep()])
    mem = DummyPersonMemory(working=wm, episodic=em)
    dec = should_follow_up(astate, mem)
    assert dec.allowed is True
    assert dec.action == "callback_past_topic"


# ----------------------------------------------------------------------
# 4. should_nudge Unit Tests
# ----------------------------------------------------------------------

def test_should_nudge_budget_and_threshold():
    astate = DummyAgentState(mode="conversing", active_person="Alice")

    # Idle duration too short -> no nudge
    dec_short = should_nudge(astate, idle_seconds=40.0, nudges_done=0, nudge_threshold_sec=90.0)
    assert dec_short.allowed is False
    assert dec_short.reason == "idle_threshold_not_met"

    # Idle duration met -> nudge allowed
    dec_ready = should_nudge(astate, idle_seconds=95.0, nudges_done=0, nudge_threshold_sec=90.0)
    assert dec_ready.allowed is True
    assert dec_ready.action == "nudge"

    # Max nudges exceeded -> strictly blocked
    dec_max = should_nudge(astate, idle_seconds=120.0, nudges_done=1, nudge_threshold_sec=90.0, max_nudges=1)
    assert dec_max.allowed is False
    assert dec_max.reason == "max_nudges_exceeded"

    # Busy agent -> blocked
    dec_busy = should_nudge(astate, idle_seconds=120.0, nudges_done=0, is_busy=True)
    assert dec_busy.allowed is False
    assert dec_busy.reason == "agent_busy"


# ----------------------------------------------------------------------
# 5. should_proactive_remark Unit Tests
# ----------------------------------------------------------------------

def test_should_proactive_remark_silence_empty_room():
    ws = DummyWorldState(faces=[])
    astate = DummyAgentState(mode="idle")
    dec = should_proactive_remark(ws, astate, last_spoke_time=0.0, only_when_seen=True, now=1000.0)
    assert dec.allowed is False
    assert dec.reason == "no_one_in_view"


def test_should_proactive_remark_silence_when_conversing():
    ws = DummyWorldState(faces=[{"track_id": "face_0"}])
    astate = DummyAgentState(mode="conversing", active_person="Alice")
    dec = should_proactive_remark(ws, astate, last_spoke_time=0.0, now=1000.0)
    assert dec.allowed is False
    assert dec.reason == "currently_conversing"


def test_should_proactive_remark_rate_limit_and_hourly_budget():
    ws = DummyWorldState(faces=[{"track_id": "face_0"}])
    astate = DummyAgentState(mode="idle")

    # Spoke too recently (< min_interval_sec)
    dec_recent = should_proactive_remark(ws, astate, last_spoke_time=950.0, min_interval_sec=90.0, now=1000.0)
    assert dec_recent.allowed is False
    assert dec_recent.reason == "min_interval_not_met"

    # Exceeded hourly cap (6 per hour)
    recent_hours = [1000.0 - (i * 60) for i in range(6)]  # 6 remarks in last 6 minutes
    dec_hourly = should_proactive_remark(
        ws, astate, last_spoke_time=500.0, hour_timestamps=recent_hours, max_per_hour=6, now=1000.0
    )
    assert dec_hourly.allowed is False
    assert dec_hourly.reason == "hourly_limit_reached"

    # All conditions satisfied
    dec_ok = should_proactive_remark(
        ws, astate, last_spoke_time=500.0, hour_timestamps=[100.0], min_interval_sec=90.0, max_per_hour=6, now=1000.0
    )
    assert dec_ok.allowed is True
    assert dec_ok.action == "proactive_remark"
    assert dec_ok.reason == "conditions_satisfied"
