import os
import json
import datetime
from typing import Any

PERSONA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona")
os.makedirs(PERSONA_DIR, exist_ok=True)

SESSION_FILE = "current_session.json"


class EpisodicMemory:
    """Stores summarized records of past encounters (one per visit)."""

    def __init__(self):
        self.episodes: list[dict] = []

    def add_episode(self, identity: str, summary: str):
        self.episodes.append({
            "identity": identity,
            "summary": summary,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        if len(self.episodes) > 50:
            self.episodes = self.episodes[-50:]


class PersonaGraph:
    """Knowledge graph of facts about a person (name, preferences, context).

    Stored as JSON. This is the 'persona' layer from research — facts that
    persist across all encounters and inform future conversations.
    """

    def __init__(self, identity: str = "unknown"):
        self.identity = identity
        self.name: str | None = None
        self.purpose: str = ""
        self.traits: list[str] = []
        self.preferences: dict[str, Any] = {}
        self.relationship_tier: str = "unknown"  # unknown -> familiar -> trusted
        self.tags: list[str] = []  # e.g. ["delivery", "frequent_visitor", "staff"]

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "name": self.name,
            "purpose": self.purpose,
            "traits": self.traits,
            "preferences": self.preferences,
            "relationship_tier": self.relationship_tier,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaGraph":
        pg = cls(data.get("identity", "unknown"))
        pg.name = data.get("name")
        pg.purpose = data.get("purpose", "")
        pg.traits = data.get("traits", [])
        pg.preferences = data.get("preferences", {})
        pg.relationship_tier = data.get("relationship_tier", "unknown")
        pg.tags = data.get("tags", [])
        return pg

    def save(self):
        key = self.identity if self.identity != "unknown" else f"anon_{datetime.datetime.now().strftime('%Y%m%d')}"
        path = os.path.join(PERSONA_DIR, f"{key}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, identity: str) -> "PersonaGraph":
        """Try loading by exact identity, then by common aliases."""
        path = os.path.join(PERSONA_DIR, f"{identity}.json")
        if os.path.exists(path):
            with open(path) as f:
                return cls.from_dict(json.load(f))
        # Try "anon_<date>" fallback for unknown faces
        return cls(identity)


class PersonMemory:
    """Multi-layered memory for a person: working context + persistent persona.

    Layer 1: Working context — short-term conversation buffer for THIS session
    Layer 2: Persona graph — persistent facts across all encounters
    Layer 3: Episodic memory — summaries of past encounters
    """

    MAX_WORKING_HISTORY = 20
    MAX_TOTAL_HISTORY = 200

    def __init__(self, identity: str = "unknown"):
        self.identity = identity
        self.working_context: list[dict] = []  # current session only
        self.persona = PersonaGraph(identity)
        self.episodic = EpisodicMemory()
        self.first_seen = datetime.datetime.now().isoformat()
        self.last_seen = self.first_seen
        self.interaction_count = 0
        self.session_id = self._generate_session_id()

    @staticmethod
    def _generate_session_id() -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def add_interaction(self, role: str, content: str):
        self.working_context.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        if len(self.working_context) > self.MAX_WORKING_HISTORY:
            self.working_context = self.working_context[-self.MAX_WORKING_HISTORY:]
        self.interaction_count += 1
        self.last_seen = self.working_context[-1]["timestamp"]

    def add_fact(self, key: str, value: Any):
        """Add to persona graph + working context."""
        setattr(self.persona, key, value)
        self.add_interaction("fact", f"{key}: {value}")

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "working_context": self.working_context,
            "persona": self.persona.to_dict(),
            "episodic": {"episodes": self.episodic.episodes},
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonMemory":
        mem = cls(data.get("identity", "unknown"))
        mem.working_context = data.get("working_context", [])
        mem.persona = PersonaGraph.from_dict(data.get("persona", {}))
        mem.episodic = EpisodicMemory()
        mem.episodic.episodes = data.get("episodic", {}).get("episodes", [])
        mem.first_seen = data.get("first_seen", mem.first_seen)
        mem.last_seen = data.get("last_seen", mem.last_seen)
        mem.interaction_count = data.get("interaction_count", 0)
        mem.session_id = data.get("session_id", mem.session_id)
        return mem

    def save(self):
        """Persist persona graph + episodic memory. Working context is session-only."""
        self.persona.save()
        # Save episodic summary file too
        epi_path = os.path.join(PERSONA_DIR, f"{self.identity}_episodes.json")
        with open(epi_path, "w") as f:
            json.dump({"episodes": self.episodic.episodes}, f, indent=2)

    @classmethod
    def load(cls, identity: str) -> "PersonMemory":
        """Load persistent layers; working context starts empty for new session."""
        mem = cls(identity)
        mem.persona = PersonaGraph.load(identity)
        # Restore episodic memory if available
        epi_path = os.path.join(PERSONA_DIR, f"{identity}_episodes.json")
        if os.path.exists(epi_path):
            with open(epi_path) as f:
                mem.episodic.episodes = json.load(f).get("episodes", [])
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
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load_current_state() -> dict[str, PersonMemory]:
        """Resume from last checkpoint if available."""
        path = os.path.join(PERSONA_DIR, SESSION_FILE)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            data = json.load(f)
        return {
            k: PersonMemory.from_dict(v)
            for k, v in data.get("active_memories", {}).items()
        }
