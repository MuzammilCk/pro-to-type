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
        self._last_saved_state: dict | None = None

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "name": self.name,
            "purpose": self.purpose,
            "traits": list(self.traits),
            "preferences": dict(self.preferences),
            "relationship_tier": self.relationship_tier,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaGraph":
        pg = cls(data.get("identity", "unknown"))
        pg.name = data.get("name")
        pg.purpose = data.get("purpose", "")
        pg.traits = list(data.get("traits", []))
        pg.preferences = dict(data.get("preferences", {}))
        pg.relationship_tier = data.get("relationship_tier", "unknown")
        pg.tags = list(data.get("tags", []))
        pg._last_saved_state = pg.to_dict()
        return pg

    def is_dirty(self) -> bool:
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        if self._last_saved_state is None:
            return True
        return self.to_dict() != self._last_saved_state

    def save(self) -> bool:
        """Persist persona graph to disk if identity is known and data changed.
        
        Unrecognized / stranger identities are session-scoped and non-persistent:
        they are never written to disk.
        """
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        if not self.is_dirty():
            return False
        state = self.to_dict()
        path = os.path.join(PERSONA_DIR, f"{self.identity}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        self._last_saved_state = state
        return True

    @classmethod
    def load(cls, identity: str) -> "PersonaGraph":
        """Try loading by exact identity for known/enrolled individuals.
        
        Unknown identities are session-scoped and never loaded from disk.
        """
        if identity == "unknown" or identity.startswith("anon_"):
            pg = cls(identity)
            pg._last_saved_state = pg.to_dict()
            return pg

        path = os.path.join(PERSONA_DIR, f"{identity}.json")
        if os.path.exists(path):
            with open(path) as f:
                return cls.from_dict(json.load(f))
        pg = cls(identity)
        pg._last_saved_state = pg.to_dict()
        return pg


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
        self._last_saved_episodes_count = 0

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

    def is_dirty(self) -> bool:
        """True iff persistent layers (persona graph or episodic memory) have changed.
        
        Unknown / stranger identities are session-scoped and never marked dirty for disk.
        """
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        if self.persona.is_dirty():
            return True
        if len(self.episodic.episodes) != self._last_saved_episodes_count:
            return True
        return False

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
        mem._last_saved_episodes_count = len(mem.episodic.episodes)
        return mem

    def save(self) -> bool:
        """Persist persona graph + episodic memory if changed. Working context is session-only.
        
        Unrecognized / stranger identities are session-scoped and non-persistent:
        never written to disk.
        """
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        if not self.is_dirty():
            return False
        self.persona.save()
        # Save episodic summary file too
        epi_path = os.path.join(PERSONA_DIR, f"{self.identity}_episodes.json")
        with open(epi_path, "w") as f:
            json.dump({"episodes": self.episodic.episodes}, f, indent=2)
        self._last_saved_episodes_count = len(self.episodic.episodes)
        return True

    @classmethod
    def load(cls, identity: str) -> "PersonMemory":
        """Load persistent layers; working context starts empty for new session.
        
        Unknown identities are always initialized fresh and not loaded from disk.
        """
        mem = cls(identity)
        if identity != "unknown" and not identity.startswith("anon_"):
            mem.persona = PersonaGraph.load(identity)
            # Restore episodic memory if available
            epi_path = os.path.join(PERSONA_DIR, f"{identity}_episodes.json")
            if os.path.exists(epi_path):
                with open(epi_path) as f:
                    mem.episodic.episodes = json.load(f).get("episodes", [])
            mem._last_saved_episodes_count = len(mem.episodic.episodes)
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
