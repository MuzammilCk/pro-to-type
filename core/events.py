"""events.py — Formalized event types for ARIA companion agent.

Phase 2 deliverables:
- EventType enum with exactly 6 events:
    PERSON_ENTERED
    PERSON_LEFT
    IDENTITY_CONFIRMED
    USER_UTTERANCE
    AGENT_STARTED_SPEAKING
    AGENT_STOPPED_SPEAKING
- Event dataclass holding type, data payload, and timestamp.
- Backwards-compatible dict interface (e.g. event["type"], event.get(...), to_dict()).
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class EventType(str, Enum):
    PERSON_ENTERED = "person_entered"
    PERSON_LEFT = "person_left"
    IDENTITY_CONFIRMED = "identity_confirmed"
    USER_UTTERANCE = "user_utterance"
    AGENT_STARTED_SPEAKING = "agent_started_speaking"
    AGENT_STOPPED_SPEAKING = "agent_stopped_speaking"


# Phase 12 autonomous trigger event type constant
VISUAL_ANOMALY_DETECTED = "visual_anomaly_detected"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __getitem__(self, key: str) -> Any:
        if key == "type":
            return self.type
        if key == "timestamp":
            return self.timestamp
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        if key in ("type", "timestamp"):
            return True
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        if key == "type":
            return self.type
        if key == "timestamp":
            return self.timestamp
        return self.data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "timestamp": self.timestamp,
            **self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        raw_type = d.get("type", "")
        if isinstance(raw_type, EventType):
            evt_type = raw_type
        elif raw_type in EventType._value2member_map_:
            evt_type = EventType(raw_type)
        else:
            evt_type = raw_type
        ts = d.get("timestamp", time.time())
        data = {k: v for k, v in d.items() if k not in ("type", "timestamp")}
        return cls(type=evt_type, data=data, timestamp=ts)

    def __repr__(self) -> str:
        t_name = self.type.name if isinstance(self.type, EventType) else str(self.type)
        return f"Event({t_name}, data={self.data}, ts={self.timestamp:.3f})"
