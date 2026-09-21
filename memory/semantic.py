"""Semantic memory layer — stable, persistent facts and preferences about enrolled individuals.

Stores:
- Identity & display name
- Structured facts (knowledge about their work, hobbies, projects)
- Procedural memory (interaction preferences such as verbosity and formality)
- Relationship tier and descriptive traits
"""
import os
import json
from typing import Any

PERSONA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "persona")
os.makedirs(PERSONA_DIR, exist_ok=True)


class SemanticMemory:
    """Persistent knowledge graph and procedural preferences for an enrolled person."""

    def __init__(
        self,
        identity: str = "unknown",
        name: str | None = None,
        purpose: str = "",
        traits: list[str] | None = None,
        facts: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
        relationship_tier: str = "unknown",
        tags: list[str] | None = None,
    ):
        self.identity = identity
        self.name = name
        self.purpose = purpose or ""
        self.traits = list(traits) if traits else []
        self.facts = dict(facts) if facts else {}
        self.preferences = dict(preferences) if preferences else {}  # Absorbs procedural preferences
        self.relationship_tier = relationship_tier  # unknown -> familiar -> trusted
        self.tags = list(tags) if tags else []
        self._last_saved_state: dict | None = None

    def add_fact(self, key: str, value: Any):
        """Record or update a stable semantic fact."""
        self.facts[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Retrieve a semantic fact by key."""
        return self.facts.get(key, default)

    def update_preference(self, key: str, value: Any):
        """Update an interaction preference (procedural tier)."""
        self.preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve an interaction preference by key."""
        return self.preferences.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "name": self.name,
            "purpose": self.purpose,
            "traits": list(self.traits),
            "facts": dict(self.facts),
            "preferences": dict(self.preferences),
            "relationship_tier": self.relationship_tier,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticMemory":
        sm = cls(data.get("identity", "unknown"))
        sm.name = data.get("name") or data.get("identity")
        sm.purpose = data.get("purpose", "")
        sm.traits = list(data.get("traits", []))
        sm.facts = dict(data.get("facts", {}))
        sm.preferences = dict(data.get("preferences", {}))
        sm.relationship_tier = data.get("relationship_tier", "unknown")
        sm.tags = list(data.get("tags", []))
        sm._last_saved_state = sm.to_dict()
        return sm

    def is_dirty(self) -> bool:
        """True iff persistent semantic data has changed since last save."""
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        if self._last_saved_state is None:
            return True
        return self.to_dict() != self._last_saved_state

    def save(self) -> bool:
        """Persist semantic memory to disk if dirty and identity is persistent."""
        if self.identity == "unknown" or self.identity.startswith("anon_"):
            return False
        if not self.is_dirty():
            return False
        state = self.to_dict()
        path = os.path.join(PERSONA_DIR, f"{self.identity}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        self._last_saved_state = state
        return True

    @classmethod
    def load(cls, identity: str) -> "SemanticMemory":
        """Load semantic memory from disk by identity, supporting clean matching."""
        if identity == "unknown" or identity.startswith("anon_"):
            sm = cls(identity)
            sm._last_saved_state = sm.to_dict()
            return sm

        exact_path = os.path.join(PERSONA_DIR, f"{identity}.json")
        if os.path.exists(exact_path):
            try:
                with open(exact_path, "r", encoding="utf-8") as f:
                    return cls.from_dict(json.load(f))
            except Exception:
                pass

        # Case-insensitive / normalized fallback matching (e.g. MuzammilCK vs MuzammilCk)
        try:
            target_low = identity.lower()
            for fname in os.listdir(PERSONA_DIR):
                if fname.endswith(".json") and not fname.endswith("_episodes.json") and fname != "current_session.json":
                    stem = fname[:-5]
                    if stem.lower() == target_low:
                        with open(os.path.join(PERSONA_DIR, fname), "r", encoding="utf-8") as f:
                            sm = cls.from_dict(json.load(f))
                            # Keep exact requested identity
                            sm.identity = identity
                            return sm
        except Exception:
            pass

        sm = cls(identity)
        sm._last_saved_state = sm.to_dict()
        return sm
