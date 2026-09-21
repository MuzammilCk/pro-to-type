"""Selective memory retrieval layer — recency + person-scoped filtering.

Roadmap specification:
- retrieve(query, person, limit=5): recency + person-scope filtering.
- No embeddings / vector search yet (out of scope for Phase 4).
- Isolates memory strictly per person (zero identity bleed).
- Produces clean structured context and formatted prompt snippets.
"""
from typing import Any


def retrieve(
    query: str,
    person: str,
    memory_obj: Any = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Retrieve relevant memories scoped to `person` using recency and keyword filtering.

    Returns a dict with:
    - semantic: dict of relevant facts, traits, preferences
    - episodic: list of relevant Episode objects (recency + keyword)
    - working: recent turns, current topic, pending question
    - context_snippet: formatted string ready for LLM system prompt injection
    """
    if not person or person in ("", "unknown") or person.startswith("anon_"):
        # Stranger scope: only working context from memory_obj (if available)
        working_turns = []
        if memory_obj and hasattr(memory_obj, "working"):
            working_turns = memory_obj.working.recent_turns(limit)
        return {
            "semantic": {},
            "episodic": [],
            "working": {"turns": working_turns, "topic": None, "pending_question": None},
            "context_snippet": "",
        }

    # Extract or load layers
    semantic = getattr(memory_obj, "semantic", None)
    episodic = getattr(memory_obj, "episodic", None)
    working = getattr(memory_obj, "working", None)
    persona_obj = getattr(memory_obj, "persona", None)

    if semantic is None:
        if persona_obj is not None:
            from memory.semantic import SemanticMemory
            semantic = SemanticMemory(
                identity=person,
                name=getattr(persona_obj, "name", None) or person,
                traits=list(getattr(persona_obj, "traits", []) or []),
                purpose=getattr(persona_obj, "purpose", None),
            )
        else:
            from memory.semantic import SemanticMemory
            semantic = SemanticMemory.load(person)

    if episodic is not None and not hasattr(episodic, "get_recent"):
        # Legacy/mock episodic object with .episodes list
        raw_eps = getattr(episodic, "episodes", [])
        from memory.episodic import EpisodicMemory, Episode
        new_ep = EpisodicMemory(identity=person)
        for e in raw_eps:
            if isinstance(e, dict):
                new_ep.add_episode(
                    summary=e.get("summary", ""),
                    timestamp=e.get("timestamp"),
                    person=person,
                )
            elif isinstance(e, Episode):
                new_ep.add(e)
        episodic = new_ep
    elif episodic is None:
        from memory.episodic import EpisodicMemory
        episodic = EpisodicMemory.load(person)

    # 1. Semantic retrieval: person profile + matching facts
    semantic_data: dict[str, Any] = {
        "name": semantic.name or person,
        "relationship_tier": semantic.relationship_tier,
        "purpose": semantic.purpose,
        "traits": list(semantic.traits),
        "preferences": dict(semantic.preferences),
        "facts": {},
    }

    q_words = [w.lower() for w in query.strip().split() if len(w) > 2] if query else []
    for k, v in semantic.facts.items():
        if not q_words or any(w in k.lower() or w in str(v).lower() for w in q_words):
            semantic_data["facts"][k] = v

    # 2. Episodic retrieval: search matching query, else most recent
    episodes = episodic.search(query, limit=limit) if query else episodic.get_recent(limit=limit)

    # 3. Working context
    working_data = {
        "turns": working.recent_turns(limit) if working else [],
        "topic": working.current_topic if working else None,
        "pending_question": working.pending_question if working else None,
    }

    # 4. Formulate speakable / promptable context snippet
    snippet_parts: list[str] = []

    # Persona / identity anchor
    display_name = semantic.name or person
    snippet_parts.append(f"Interlocutor: {display_name} (tier: {semantic.relationship_tier}).")
    if semantic.purpose:
        snippet_parts.append(f"Purpose: {semantic.purpose}.")
    if semantic.traits:
        snippet_parts.append(f"Traits: {', '.join(semantic.traits[:4])}.")
    if semantic_data["facts"]:
        fact_strs = [f"{k}: {v}" for k, v in list(semantic_data["facts"].items())[:4]]
        snippet_parts.append(f"Known facts: {'; '.join(fact_strs)}.")
    if semantic.preferences:
        pref_strs = [f"{k}={v}" for k, v in semantic.preferences.items()]
        snippet_parts.append(f"Preferences: {', '.join(pref_strs)}.")

    # Episodic recall
    if episodes:
        recent_summaries = [f"[{ep.timestamp[:10]}] {ep.summary}" for ep in episodes[:3]]
        snippet_parts.append(f"Past interactions: {' | '.join(recent_summaries)}")

    if working_data["topic"]:
        snippet_parts.append(f"Current topic: {working_data['topic']}.")
    if working_data["pending_question"]:
        snippet_parts.append(f"Awaiting response to: '{working_data['pending_question']}'.")

    context_snippet = " ".join(snippet_parts)

    return {
        "semantic": semantic_data,
        "episodic": episodes,
        "working": working_data,
        "context_snippet": context_snippet,
    }


def retrieve_greeting_context(person: str, memory_obj: Any = None) -> str | None:
    """Retrieve unprompted callback context for session-opening greetings.

    Definition of Done requirement: references something specific from past
    sessions unprompted upon return.
    """
    if not person or person in ("", "unknown") or person.startswith("anon_"):
        return None

    episodic = getattr(memory_obj, "episodic", None)
    semantic = getattr(memory_obj, "semantic", None)
    persona_obj = getattr(memory_obj, "persona", None)

    if episodic is not None and not hasattr(episodic, "get_recent"):
        raw_eps = getattr(episodic, "episodes", [])
        from memory.episodic import EpisodicMemory, Episode
        new_ep = EpisodicMemory(identity=person)
        for e in raw_eps:
            if isinstance(e, dict):
                new_ep.add_episode(
                    summary=e.get("summary", ""),
                    timestamp=e.get("timestamp"),
                    person=person,
                )
            elif isinstance(e, Episode):
                new_ep.add(e)
        episodic = new_ep
    elif episodic is None:
        from memory.episodic import EpisodicMemory
        episodic = EpisodicMemory.load(person)

    if semantic is None:
        if persona_obj is not None:
            from memory.semantic import SemanticMemory
            semantic = SemanticMemory(
                identity=person,
                name=getattr(persona_obj, "name", None) or person,
                traits=list(getattr(persona_obj, "traits", []) or []),
                purpose=getattr(persona_obj, "purpose", None),
            )
        else:
            from memory.semantic import SemanticMemory
            semantic = SemanticMemory.load(person)

    recent_eps = episodic.get_recent(limit=1)
    if recent_eps:
        latest = recent_eps[-1]
        summary = latest.summary.rstrip(".")
        topics = ", ".join(latest.topics) if latest.topics else ""
        if topics:
            return f"Earlier you worked on {topics} ({summary})."
        return f"Last time, you mentioned: {summary}."

    # Fallback to semantic purpose or facts
    if semantic and semantic.purpose:
        return f"Earlier you mentioned {semantic.purpose}."
    if semantic and semantic.facts:
        first_fact_key = list(semantic.facts.keys())[0]
        return f"You previously mentioned {first_fact_key}: {semantic.facts[first_fact_key]}."

    return None
