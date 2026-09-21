import os
import json
import datetime
from typing import Any

from memory.working import WorkingMemory
from memory.semantic import SemanticMemory, PERSONA_DIR
from memory.episodic import EpisodicMemory, Episode
from memory.writer import MemoryWriter

SESSION_FILE = "current_session.json"


class PersonaGraph(SemanticMemory):
    """Knowledge graph of facts about a person (name, preferences, context).

    Inherits from SemanticMemory to unify persistent persona and semantic memory layers.
    """
    pass


class PersonMemory:
    """Multi-layered memory for a person: working context + semantic facts + episodic memory.

    Layer 1: Working memory   — short-term conversation buffer for THIS session (ephemeral)
    Layer 2: Semantic memory  — persistent facts and procedural preferences across all encounters
    Layer 3: Episodic memory  — dated summaries of past encounters
    """

    MAX_WORKING_HISTORY = 20
    MAX_TOTAL_HISTORY = 200

    def __init__(self, identity: str = "unknown"):
        self.identity = identity
        self.working = WorkingMemory(max_turns=self.MAX_WORKING_HISTORY)
        self.semantic = (
            SemanticMemory.load(identity)
            if identity != "unknown" and not identity.startswith("anon_")
            else SemanticMemory(identity)
        )
        self.episodic = (
            EpisodicMemory.load(identity)
            if identity != "unknown" and not identity.startswith("anon_")
            else EpisodicMemory(identity)
        )
        self.first_seen = datetime.datetime.now().isoformat()
        self.last_seen = self.first_seen
        self.interaction_count = 0
        self.session_id = self._generate_session_id()

    @property
    def persona(self) -> SemanticMemory:
        """Backwards compatibility alias for semantic memory layer."""
        return self.semantic

    @persona.setter
    def persona(self, value: SemanticMemory):
        self.semantic = value

    @property
    def working_context(self) -> list[dict]:
        """Backwards compatibility alias for working memory turns."""
        return self.working.turns

    @working_context.setter
    def working_context(self, value: list[dict]):
        self.working.turns = value

    @staticmethod
    def _generate_session_id() -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def add_interaction(self, role: str, content: str):
        """Record an interaction turn in working memory."""
        self.working.add_turn(role, content)
        self.interaction_count += 1
        if self.working.turns:
            self.last_seen = self.working.turns[-1]["timestamp"]

    def add_fact(self, key: str, value: Any):
        """Add to semantic memory + working context."""
        if hasattr(self.semantic, key):
            setattr(self.semantic, key, value)
        self.semantic.add_fact(key, value)
        self.add_interaction("fact", f"{key}: {value}")

    def is_dirty(self) -> bool:
        """True iff persistent layers (semantic memory or episodic memory) have changed.

        Unknown / stranger identities are session-scoped and never marked dirty for disk.
        """
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        return self.semantic.is_dirty() or self.episodic.is_dirty()

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "working_context": self.working.turns,
            "persona": self.semantic.to_dict(),
            "episodic": self.episodic.to_dict(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonMemory":
        identity = data.get("identity", "unknown")
        mem = cls(identity)
        mem.working.turns = list(data.get("working_context", []))
        mem.semantic = SemanticMemory.from_dict(data.get("persona", {}))
        mem.episodic = EpisodicMemory.from_dict(data.get("episodic", {}), identity=identity)
        mem.first_seen = data.get("first_seen", mem.first_seen)
        mem.last_seen = data.get("last_seen", mem.last_seen)
        mem.interaction_count = data.get("interaction_count", 0)
        mem.session_id = data.get("session_id", mem.session_id)
        return mem

    def save(self) -> bool:
        """Persist semantic memory + episodic memory if changed via write-gate.

        Unrecognized / stranger identities are session-scoped and non-persistent:
        never written to disk.
        """
        return MemoryWriter.save_person(self)

    @classmethod
    def load(cls, identity: str) -> "PersonMemory":
        """Load persistent layers; working context starts empty for new session.

        Unknown identities are always initialized fresh and not loaded from disk.
        """
        mem = cls(identity)
        return mem


class SessionManager:
    """Checkpoint/resume session state for conversation continuity."""

    @staticmethod
    def save_current_state(identities: dict[str, PersonMemory]):
        """Save all active person memories to checkpoint file."""
        path = os.path.join(PERSONA_DIR, SESSION_FILE)
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "active_memories": {k: v.to_dict() for k, v in identities.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load_current_state() -> dict[str, PersonMemory]:
        """Resume from last checkpoint if available."""
        path = os.path.join(PERSONA_DIR, SESSION_FILE)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            k: PersonMemory.from_dict(v)
            for k, v in data.get("active_memories", {}).items()
        }
