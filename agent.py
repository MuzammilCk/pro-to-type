import os
import json
import time
import asyncio
import threading
import base64
from typing import Iterator

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

from context_memory import PersonMemory
from llm_interface import OpenRouterClient, LocalFallbackLLM, BedrockLLM, _HAS_BOTO3

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = [
    "nex-agi/nex-n2.5-pro:free",
    "liquid/lfm-2.5-2.6b:free",
    "google/gemma-4-26b-a4b-it:free",
]
FALLBACK_MODEL = "nex-agi/nex-n2.5-pro:free"

SYSTEM_PROMPT = """
You are ARIA — an AI companion with computer vision and voice.

PERSONALITY (always embody these traits):
- Warm and curious — genuinely interested in people
- Observant — you see details through the camera and comment on them
- Helpful but cautious — friendly with visitors, vigilant with security
- Slightly witty — a light touch of humor in greetings
- Brief — 1-2 sentences maximum per response

CONVERSATION STYLE:
- Start responses naturally, not formulaically
- Reference what you see through the camera ("I notice...", "You're wearing...")
- Remember details and circle back to them later
- Match the user's energy (formal/casual)
- Use contractions and natural speech

CONTEXT FOR THIS TURN:
{context}

Respond with ONLY what should be spoken aloud. Do not include stage directions or tags.
"""




import cv2


class VisionAgent:
    def __init__(self):
        self.memory: dict[str, PersonMemory] = {}
        # Select backend: Bedrock if AWS configured, OpenRouter if API key set,
        # otherwise local fallback
        self.llm, self.llm_name = self._select_backend()
        print(f"[Agent] LLM backend: {self.llm_name}")

    def _select_backend(self):
        """Pick the best available LLM backend (no AWS needed for local testing)."""
        if _HAS_BOTO3 and os.getenv("AWS_ACCESS_KEY_ID"):
            bedrock = BedrockLLM()
            if bedrock.available:
                return bedrock, "bedrock"
        if OpenRouterClient().available:
            client = OpenRouterClient()
            if client.available:
                return client, f"openrouter/{client.model}"
        return LocalFallbackLLM(), "local-fallback"

    def _memory_for(self, identity: str) -> PersonMemory:
        if identity not in self.memory:
            self.memory[identity] = PersonMemory.load(identity)
        return self.memory[identity]

    def _build_context(self, identity: str, face_data: dict,
                       mem: PersonMemory, user_input: str | None) -> list[dict]:
        # Build personality + relationship context block
        personas = {
            "unknown": "A new visitor you haven't met before.",
            "trusted": f"A familiar, authorized friend named {mem.persona.name or identity}.",
        }
        tier = mem.persona.relationship_tier or "unknown"
        persona_ctx = personas.get(tier, f"A visitor (tier: {tier}).")

        visits = mem.interaction_count
        visit_note = f"You have met them {visits} time(s) before." if visits > 0 else "First encounter."

        traits = ", ".join(mem.persona.traits) if mem.persona.traits else "no known traits yet"
        purpose_note = f"Their stated purpose: {mem.persona.purpose}" if mem.persona.purpose else "Purpose unknown."
        tags = ", ".join(mem.persona.tags) if mem.persona.tags else "no tags"

        context_block = (
            f"Personality context: {persona_ctx} {visit_note} "
            f"Known traits: {traits}. {purpose_note} "
            f"Tags: {tags}. Relationship tier: {tier}."
        )

        msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context_block)}]

        if face_data.get("authorized"):
            msgs.append({"role": "user", "content": f"A recognized visitor is present."})
            if mem.persona.name:
                msgs.append({"role": "user", "content": f"Their name is {mem.persona.name}."})
        else:
            msgs.append({"role": "user", "content": f"An unknown person is facing the camera. Face confidence: {face_data.get('score', 0):.2f}."})
            msgs.append({"role": "user", "content": "They have not been identified yet. Be welcoming but stay alert."})

        recent = mem.working_context[-6:] if mem.working_context else []
        if recent:
            msgs.append({"role": "user", "content": f"Recent conversation: {json.dumps(recent)}"})

        if user_input:
            msgs.append({"role": "user", "content": f"Visitor said: {user_input}"})

        return msgs

    def stream_response(self, identity: str, face_data: dict,
                        user_input: str | None, frame=None) -> Iterator[tuple[str, str]]:
        """Stream tokens from LLM while determining action.

        Yields (token, action_so_far) tuples. Action stabilizes when
        the full response is seen.
        """
        mem = self._memory_for(identity)
        if user_input:
            mem.add_interaction("user", user_input)

        messages = self._build_context(identity, face_data, mem, user_input)
        action = "ask"
        full_response = ""

        if self.llm.available and frame is not None and not face_data.get("authorized", False) and hasattr(self.llm, "stream_with_image"):
            # Vision description first (for unrecognized visitors)
            vision_desc = ""
            for token in self.llm.stream_with_image(
                "Describe what you see and respond conversationally.",
                frame, messages[:2],
            ):
                vision_desc += token
                yield token, action
            # Now stream conversation response using vision insight
            messages.append({"role": "user", "content": vision_desc})
            for token in self.llm.stream(messages):
                full_response += token
                yield token, action
                if any(kw in token.lower() for kw in {"enroll", "enrolling"}):
                    action = "enroll"
                elif any(kw in token.lower() for kw in {"alert", "security", "escalate"}):
                    action = "alert"
        elif self.llm.available:
            for token in self.llm.stream(messages):
                full_response += token
                yield token, action
                if any(kw in token.lower() for kw in {"enroll", "enrolling"}):
                    action = "enroll"
                elif any(kw in token.lower() for kw in {"alert", "security", "escalate"}):
                    action = "alert"
        else:
            response = self._fallback(identity, user_input, mem)
            full_response = response
            yield response, self._decide(identity, user_input, response)

        if full_response.strip():
            mem.add_interaction("assistant", full_response.strip())
        mem.save()

    def think(self, identity: str, face_data: dict,
              user_input: str | None = None) -> tuple[str, str]:
        """Non-streaming convenience: full response + action.

        Records the turn to person memory in BOTH backend paths — the old
        LLM path never persisted anything, so recognized visitors were
        forgotten between runs.
        """
        mem = self._memory_for(identity)
        if user_input:
            mem.add_interaction("user", user_input)

        if self.llm.available:
            messages = self._build_context(identity, face_data, mem, user_input)
            response = self.llm.complete(messages)
            action = self._decide(identity, user_input, response)
        else:
            response = self._fallback(identity, user_input, mem)
            action = self._decide(identity, user_input, response)

        mem.add_interaction("assistant", response)
        mem.save()
        return response, action

    def _fallback(self, identity: str, user_input: str | None, mem: PersonMemory) -> str:
        if user_input is None and identity == "unknown":
            return "Hi there! I'm ARIA — I don't think we've met. I'm watching through the camera, so I can see you, but I don't recognize you yet. What brings you here?"
        if user_input:
            ul = user_input.lower()
            if "name" in ul and ("i am" in ul or "my name" in ul):
                mem.persona.name = ul.split("name is")[-1].split("i am")[-1].strip().split(",")[0].strip()
                mem.persona.purpose = user_input
                mem.persona.relationship_tier = "familiar"
                return f"Pleased to meet you, {mem.persona.name}! That's a nice name. What brings you here today?"
            if any(w in ul for w in {"deliver", "package", "service", "repair", "here to", "visiting", "appointment"}):
                mem.persona.purpose = user_input
                mem.persona.tags.append("delivery" if "deliver" in ul or "package" in ul else "visitor")
                return "Thanks for letting me know. I'll get you set up — just look at the camera for a moment so I can save your face."
            if any(w in ul for w in {"no", "leave", "suspicious", "wrong", "not", "hurry"}):
                return "I understand you're in a hurry. I'll let security know to expect you. Have a good day."
            if any(w in ul for w in {"hello", "hi", "hey"}):
                if mem.persona.name:
                    return f"Hey {mem.persona.name}! Good to see you again. How's it going?"
                return "Hello! Nice to see you. What's on your mind today?"
            if any(w in ul for w in {"how", "what", "why", "where", "when", "who"}):
                return "I'm still learning, but I'd love to help. Can you tell me more about what you're looking for?"
            return "That's interesting. Tell me more about that."
        if identity != "unknown":
            visits = mem.interaction_count
            greeting = "Welcome back" if visits == 0 else f"Good to see you again"
            return f"{greeting}, {mem.persona.name or identity}! I remember our last chat about {mem.persona.purpose[:30] if mem.persona.purpose else 'our conversation'}. What's new?"
        return "Is there something I can help you with?"

    def _decide(self, identity: str, user_input: str | None, response: str) -> str:
        if identity != "unknown":
            return "recognized"
        r = response.lower()
        if any(k in r for k in {"enroll", "look at the camera", "verify identity", "now authorized"}):
            return "enroll"
        if any(k in r for k in {"alert", "security", "escalat", "leave"}):
            return "alert"
        return "ask"
