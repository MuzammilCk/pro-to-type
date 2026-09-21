"""Memory writer layer — formalizes the Phase 1 write-gate.

Enforces:
1. Stranger barrier: Unenrolled visitors ('unknown' or 'anon_*') never write to disk.
2. Dirty-state gate: Skips redundant disk writes if semantic/episodic data hasn't changed.
3. Content validity: Ensures candidate facts and summaries are meaningful before storing.
"""
from datetime import datetime, timezone
import json
import os
import time
from typing import Any


class MemoryWriter:
    """Gatekeeper for all persistent memory mutations."""

    @staticmethod
    def is_persistent_identity(identity: str | None) -> bool:
        """Check if an identity is eligible for disk persistence."""
        if not identity:
            return False
        return identity != "unknown" and not identity.startswith("anon_")

    @classmethod
    def save_person(cls, person_memory: Any) -> bool:
        """Persist semantic and episodic memory layers through write-gate checks.

        Returns True iff a persistent file was actually written.
        """
        identity = getattr(person_memory, "identity", "unknown")
        if not cls.is_persistent_identity(identity):
            return False

        saved_any = False

        # 1. Semantic memory layer
        semantic = getattr(person_memory, "semantic", None)
        if semantic is not None and getattr(semantic, "is_dirty", lambda: False)():
            if semantic.save():
                saved_any = True

        # 2. Episodic memory layer
        episodic = getattr(person_memory, "episodic", None)
        if episodic is not None and getattr(episodic, "is_dirty", lambda: False)():
            if episodic.save(identity):
                saved_any = True

        return saved_any

    @classmethod
    def record_episode(
        cls,
        person_memory: Any,
        summary: str,
        topics: list[str] | None = None,
        key_events: list[str] | None = None,
    ) -> bool:
        """Create a discrete episode record and commit via the write-gate."""
        if not summary or not summary.strip():
            return False
        identity = getattr(person_memory, "identity", "unknown")
        if not cls.is_persistent_identity(identity):
            return False

        episodic = getattr(person_memory, "episodic", None)
        if episodic is None or not hasattr(episodic, "add_episode"):
            return False

        episodic.add_episode(
            summary=summary.strip(),
            topics=topics or [],
            key_events=key_events or [],
            person=identity,
        )
        return cls.save_person(person_memory)

    @classmethod
    def record_audit_trace(
        cls,
        trace_data: dict[str, Any],
        audit_file: str | None = None,
    ) -> str:
        """Append an immutable structured JSONL record to the audit log file.

        Args:
            trace_data: Dictionary containing trace metadata, ReAct steps, decisions, and evidence.
            audit_file: Optional path to JSONL log file. Defaults to './evidence/audit_log.jsonl'.

        Returns:
            The filepath where the audit line was appended.
        """
        target_path = audit_file or os.path.join(".", "evidence", "audit_log.jsonl")
        target_dir = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(target_dir, exist_ok=True)

        entry = dict(trace_data)
        if "timestamp" not in entry:
            entry["timestamp"] = time.time()
        if "iso_timestamp" not in entry:
            entry["iso_timestamp"] = datetime.now(timezone.utc).isoformat()

        def _json_default(obj: Any) -> Any:
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        line = json.dumps(entry, default=_json_default)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        return target_path
