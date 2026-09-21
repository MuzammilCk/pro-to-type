"""Episodic memory layer — dated, discrete records of past interactions.

Stores narrative summaries of past sessions:
- What was discussed
- What key events or milestones occurred
- Dated and sequenced for chronological or keyword recall
"""
import os
import json
import datetime
from dataclasses import dataclass, field
from typing import Any

PERSONA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "persona")
os.makedirs(PERSONA_DIR, exist_ok=True)


@dataclass
class Episode:
    """A discrete recorded encounter or conversation episode."""
    id: str
    timestamp: str
    summary: str
    topics: list[str] = field(default_factory=list)
    key_events: list[str] = field(default_factory=list)
    person: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "topics": list(self.topics),
            "key_events": list(self.key_events),
            "person": self.person,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            id=data.get("id") or f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=data.get("timestamp") or datetime.datetime.now().isoformat(),
            summary=data.get("summary", ""),
            topics=list(data.get("topics", [])),
            key_events=list(data.get("key_events", [])),
            person=data.get("person", "unknown"),
        )


class EpisodicMemory:
    """Manages collection of discrete interaction records."""

    MAX_EPISODES = 50

    def __init__(self, identity: str = "unknown"):
        self.identity = identity
        self.episodes: list[Episode] = []
        self._last_saved_count = 0

    def add_episode(
        self,
        summary: str,
        topics: list[str] | None = None,
        key_events: list[str] | None = None,
        person: str | None = None,
        timestamp: str | None = None,
    ) -> Episode:
        """Add a new dated interaction record."""
        ts = timestamp or datetime.datetime.now().isoformat()
        ep_id = f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        ep = Episode(
            id=ep_id,
            timestamp=ts,
            summary=summary,
            topics=list(topics or []),
            key_events=list(key_events or []),
            person=person or self.identity,
        )
        self.episodes.append(ep)
        if len(self.episodes) > self.MAX_EPISODES:
            self.episodes = self.episodes[-self.MAX_EPISODES:]
        return ep

    def add(self, episode: Episode) -> None:
        """Append an Episode directly."""
        self.episodes.append(episode)
        if len(self.episodes) > self.MAX_EPISODES:
            self.episodes = self.episodes[-self.MAX_EPISODES:]

    def get_recent(self, limit: int = 5) -> list[Episode]:
        """Return the most recent episodes in chronological order."""
        return self.episodes[-limit:] if self.episodes else []

    def search(self, query: str, limit: int = 5) -> list[Episode]:
        """Search episodes by keyword match against summary or topics."""
        if not query or not query.strip():
            return self.get_recent(limit)
        q_words = [w.lower() for w in query.strip().split() if len(w) > 2]
        if not q_words:
            return self.get_recent(limit)

        matches: list[tuple[int, Episode]] = []
        for ep in self.episodes:
            score = 0
            text = (ep.summary + " " + " ".join(ep.topics) + " " + " ".join(ep.key_events)).lower()
            for w in q_words:
                if w in text:
                    score += 1
            if score > 0:
                matches.append((score, ep))

        # Sort by match score descending, then by timestamp
        matches.sort(key=lambda m: (m[0], m[1].timestamp), reverse=True)
        return [m[1] for m in matches[:limit]]

    def is_dirty(self) -> bool:
        return len(self.episodes) != self._last_saved_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], identity: str = "unknown") -> "EpisodicMemory":
        em = cls(identity=data.get("identity", identity))
        raw_eps = data.get("episodes", [])
        em.episodes = [
            Episode.from_dict(e) if isinstance(e, dict) else Episode(id="ep", timestamp="", summary=str(e))
            for e in raw_eps
        ]
        em._last_saved_count = len(em.episodes)
        return em

    def save(self, identity: str | None = None) -> bool:
        """Persist episodic memory to disk for known persons."""
        target_id = identity or self.identity
        if target_id == "unknown" or target_id.startswith("anon_"):
            return False
        if not self.is_dirty():
            return False
        path = os.path.join(PERSONA_DIR, f"{target_id}_episodes.json")
        data = {"identity": target_id, "episodes": [ep.to_dict() for ep in self.episodes]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        self._last_saved_count = len(self.episodes)
        return True

    @classmethod
    def load(cls, identity: str) -> "EpisodicMemory":
        """Load episodic memory from disk."""
        em = cls(identity)
        if identity == "unknown" or identity.startswith("anon_"):
            return em

        exact_path = os.path.join(PERSONA_DIR, f"{identity}_episodes.json")
        if os.path.exists(exact_path):
            try:
                with open(exact_path, "r", encoding="utf-8") as f:
                    return cls.from_dict(json.load(f), identity=identity)
            except Exception:
                pass

        # Case-insensitive fallback
        try:
            target_low = f"{identity.lower()}_episodes.json"
            for fname in os.listdir(PERSONA_DIR):
                if fname.lower() == target_low:
                    with open(os.path.join(PERSONA_DIR, fname), "r", encoding="utf-8") as f:
                        return cls.from_dict(json.load(f), identity=identity)
        except Exception:
            pass

        return em
