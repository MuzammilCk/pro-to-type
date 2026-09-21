"""test_phase2_events.py — Unit tests for Phase 2 events formalization.

Verifies:
1. Exactly the 6 required EventType members exist (no unauthorized event types).
2. Event dataclass serialization, dict indexing, and alias compatibility.
3. PresenceManager emits PERSON_ENTERED once per arrival (not once per frame).
4. PresenceManager emits IDENTITY_CONFIRMED on authorization and PERSON_LEFT on departure.
5. Voice stack emits USER_UTTERANCE once per turn.
6. Voice stack emits AGENT_STARTED_SPEAKING and AGENT_STOPPED_SPEAKING on speech output.
"""
import queue
import time
import pytest

from core.events import Event, EventType
from perception.presence import PresenceManager
from interaction.voice import SarvamVoice


def test_phase2_exactly_six_event_types():
    expected = {
        "PERSON_ENTERED",
        "PERSON_LEFT",
        "IDENTITY_CONFIRMED",
        "USER_UTTERANCE",
        "AGENT_STARTED_SPEAKING",
        "AGENT_STOPPED_SPEAKING",
    }
    actual = {e.name for e in EventType}
    assert actual == expected, f"EventType must have exactly the 6 Phase 2 types, got: {actual}"


def test_phase2_event_dataclass_and_dict_access():
    evt = Event(
        type=EventType.IDENTITY_CONFIRMED,
        data={"name": "Alice", "score": 0.88, "track_id": "face_0"},
    )
    # Typed access
    assert evt.type == EventType.IDENTITY_CONFIRMED
    assert evt.data["name"] == "Alice"
    assert isinstance(evt.timestamp, float)

    # Dict-like backwards compatibility
    assert evt["type"] == EventType.IDENTITY_CONFIRMED
    assert evt["type"] == "identity_confirmed"
    assert evt["name"] == "Alice"
    assert evt.get("score") == 0.88
    assert evt.get("nonexistent", "fallback") == "fallback"
    assert "name" in evt
    assert "type" in evt

    # Serialization roundtrip
    d = evt.to_dict()
    assert d["type"] == "identity_confirmed"
    assert d["name"] == "Alice"

    restored = Event.from_dict(d)
    assert restored.type == EventType.IDENTITY_CONFIRMED
    assert restored["name"] == "Alice"


def test_phase2_person_entered_fires_once_per_arrival_not_per_frame():
    q = queue.Queue()
    pm = PresenceManager(q)

    fr = {
        "track_id": "face_1",
        "identity": "unknown",
        "authorized": False,
        "distance": 0.1,
        "face_bbox": (10, 10, 50, 50),
    }

    # Frame 1: person arrives
    pm.update([fr])
    events = []
    while not q.empty():
        events.append(q.get_nowait())

    entered_events = [e for e in events if e.type == EventType.PERSON_ENTERED]
    assert len(entered_events) == 1, "PERSON_ENTERED must fire on first arrival"
    assert entered_events[0]["track_id"] == "face_1"

    # Frame 2 to 5: same person remains on camera
    for _ in range(4):
        pm.update([fr])
    
    subsequent_events = []
    while not q.empty():
        subsequent_events.append(q.get_nowait())

    subsequent_entered = [e for e in subsequent_events if e.type == EventType.PERSON_ENTERED]
    assert len(subsequent_entered) == 0, (
        "PERSON_ENTERED must NOT fire on subsequent frames for the same track"
    )


def test_phase2_identity_confirmed_and_person_left():
    q = queue.Queue()
    pm = PresenceManager(q)

    # Unknown visitor arrives
    pm.update([{
        "track_id": "face_2",
        "identity": "unknown",
        "authorized": False,
        "distance": 0.1,
    }])
    while not q.empty():
        q.get_nowait()

    # Track recognized as Bob
    now = time.time()
    pm.update([{
        "track_id": "face_2",
        "identity": "Bob",
        "authorized": True,
        "distance": 0.85,
    }], now=now)

    rec_events = []
    while not q.empty():
        rec_events.append(q.get_nowait())

    id_confirmed = [e for e in rec_events if e.type == EventType.IDENTITY_CONFIRMED]
    assert len(id_confirmed) == 1
    assert id_confirmed[0]["name"] == "Bob"
    assert id_confirmed[0]["score"] == 0.85

    # Track leaves (simulate 3s absence)
    pm.update([], now=now + 3.0)
    left_events = []
    while not q.empty():
        left_events.append(q.get_nowait())

    person_left = [e for e in left_events if e.type == EventType.PERSON_LEFT]
    assert len(person_left) == 1
    assert person_left[0]["track_id"] == "face_2"
    assert person_left[0]["identity"] == "Bob"


def test_phase2_voice_emits_user_utterance_and_agent_speaking():
    event_q = queue.Queue()
    voice = SarvamVoice(api_key=None)  # forces print fallback
    voice.set_event_queue(event_q)

    # 1. Test speak() emits AGENT_STARTED_SPEAKING and AGENT_STOPPED_SPEAKING
    voice.speak("Hello there!", interrupt=False)

    speech_events = []
    while not event_q.empty():
        speech_events.append(event_q.get_nowait())

    types = [e.type for e in speech_events]
    assert EventType.AGENT_STARTED_SPEAKING in types
    assert EventType.AGENT_STOPPED_SPEAKING in types
    start_idx = types.index(EventType.AGENT_STARTED_SPEAKING)
    stop_idx = types.index(EventType.AGENT_STOPPED_SPEAKING)
    assert start_idx < stop_idx, "AGENT_STARTED_SPEAKING must precede AGENT_STOPPED_SPEAKING"

    # 2. Test listen() with dummy mic emits USER_UTTERANCE once per turn
    class DummyMic:
        def wait_for_utterance(self, max_wait=10.0, max_utterance=8.0):
            import numpy as np
            # Return dummy 16k PCM frame
            return np.zeros(1600, dtype=np.int16)

    # Mock _transcribe_pcm or local recognizer to return text
    voice.available = lambda: True
    voice._transcribe_pcm = lambda pcm: "What is your name?"

    transcript = voice.listen(mic=DummyMic())
    assert transcript == "What is your name?"

    utterance_events = []
    while not event_q.empty():
        utterance_events.append(event_q.get_nowait())

    user_utterance = [e for e in utterance_events if e.type == EventType.USER_UTTERANCE]
    assert len(user_utterance) == 1
    assert user_utterance[0]["text"] == "What is your name?"
