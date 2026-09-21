"""Working memory layer — session-scoped, ephemeral conversational buffer.

Tracks the immediate conversational context for the active session:
- Last N interaction turns
- Current active topic
- Pending question awaiting response
Never persisted to disk (matching stranger privacy and session scope rules).
"""
import datetime
from typing import Any


class WorkingMemory:
    """Session-scoped conversational buffer for immediate turn context."""

    DEFAULT_MAX_TURNS = 20

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self.max_turns = max_turns
        self.turns: list[dict] = []
        self.current_topic: str | None = None
        self.pending_question: str | None = None

    def add_turn(self, role: str, content: str, timestamp: str | None = None):
        """Append a conversational turn to the rolling buffer."""
        ts = timestamp or datetime.datetime.now().isoformat()
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": ts,
        })
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def set_topic(self, topic: str | None):
        """Update active topic of discussion."""
        self.current_topic = topic

    def set_pending_question(self, question: str | None):
        """Record a question the agent asked that is awaiting a response."""
        self.pending_question = question

    def clear(self):
        """Wipe working memory state (e.g. on visitor departure or session end)."""
        self.turns.clear()
        self.current_topic = None
        self.pending_question = None

    def recent_turns(self, n: int = 6) -> list[dict]:
        """Return the last n interaction turns."""
        return self.turns[-n:] if self.turns else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": list(self.turns),
            "current_topic": self.current_topic,
            "pending_question": self.pending_question,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        wm = cls()
        wm.turns = list(data.get("turns", []))
        wm.current_topic = data.get("current_topic")
        wm.pending_question = data.get("pending_question")
        return wm
