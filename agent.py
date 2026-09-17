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


class OpenRouterClient:
    """OpenRouter API client with streaming + provider routing + vision support.

    Architecture per research: unified /chat/completions endpoint,
    same for text/vision. Uses SSE streaming for real-time token output.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.referer = os.getenv("OPENROUTER_REFERER", "http://localhost:8080")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "ARIA-Companion")
        self.model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODELS[0])
        self.fallback_models = [self.model] + [
            m for m in DEFAULT_MODELS if m != self.model
        ]

    @property
    def available(self) -> bool:
        return self.api_key is not None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer,
            "X-Title": self.app_name,
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict], stream: bool = True) -> dict:
        is_free = ":free" in self.model
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": 300,
            "temperature": 0.7,
        }
        if not is_free:
            payload["provider"] = {
                "sort": "latency",
                "allow_fallbacks": True,
                "data_collection": "deny",
            }
        return payload

    def _payload_with_image(self, text: str, image_b64: str,
                            messages: list[dict] | None = None) -> dict:
        msgs = messages or []
        is_free = ":free" in self.model
        payload = {
            "model": self.model,
            "messages": msgs + [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
            "stream": True,
            "max_tokens": 300,
            "temperature": 0.5,
        }
        if not is_free:
            payload["provider"] = {"sort": "latency", "allow_fallbacks": True}
        return payload

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """SSE streaming — yields tokens as they arrive (sub-100ms latency).

        Retries with fallback models on rate limit or unavailable errors.
        Handles SSE control lines (data: [DONE]) gracefully.
        """
        if not _HAS_HTTPX:
            yield "Streaming requires httpx. Install: pip install httpx"
            return

        for model in self.fallback_models:
            self.model = model
            payload = self._payload(messages)
            try:
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", f"{OPENROUTER_BASE}/chat/completions",
                                       headers=self._headers(),
                                       json=payload) as response:
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    return  # Stream complete
                                try:
                                    data = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue
                                if data.get("choices"):
                                    delta = data["choices"][0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                return  # Success — exit retry loop
            except (httpx.HTTPStatusError, KeyError, IndexError) as e:
                print(f"[OpenRouter] Model {model} failed: {e}, trying fallback...")
                time.sleep(1)
                continue

    def stream_with_image(self, text: str, frame, messages: list[dict] | None = None) -> Iterator[str]:
        """Vision + streaming: send frame as base64 image with text prompt.

        Retries with fallback models on rate limit or unavailable errors.
        """
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            yield ""
            return
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        for model in self.fallback_models:
            self.model = model
            payload = self._payload_with_image(text, b64, messages)
            try:
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", f"{OPENROUTER_BASE}/chat/completions",
                                       headers=self._headers(), json=payload) as response:
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    return
                                try:
                                    data = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue
                                if data.get("choices"):
                                    delta = data["choices"][0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                return
            except (httpx.HTTPStatusError, KeyError, IndexError) as e:
                print(f"[OpenRouter] Vision model {model} failed: {e}, trying fallback...")
                time.sleep(1)
                continue
        yield ""  # All models exhausted

    def complete(self, messages: list[dict]) -> str:
        """Non-streaming completion with model fallback + retry."""
        if not _HAS_HTTPX:
            return ""

        for model in self.fallback_models:
            self.model = model
            payload = self._payload(messages, stream=False)
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(f"{OPENROUTER_BASE}/chat/completions",
                                       headers=self._headers(), json=payload)
                    data = resp.json()
                    if "error" in data:
                        err = data["error"]["message"][:80]
                        print(f"[OpenRouter] {model}: {err}, trying fallback...")
                        time.sleep(1)
                        continue
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[OpenRouter] {model} failed: {e}, trying fallback...")
                time.sleep(1)
                continue

        return ""  # All models exhausted


import cv2


class VisionAgent:
    """Multi-tier AI: OpenRouter (Claude/Llama/Gemma) -> Ollama -> rule-based."""

    def __init__(self):
        self.openrouter = OpenRouterClient()
        self.memory: dict[str, PersonMemory] = {}

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

        if self.openrouter.available and frame is not None and not face_data.get("authorized", False):
            # Vision description first (for unrecognized visitors)
            vision_desc = ""
            for token in self.openrouter.stream_with_image(
                "Describe what you see and respond conversationally.",
                frame, messages[:2],
            ):
                vision_desc += token
                yield token, action
            # Now stream conversation response using vision insight
            messages.append({"role": "user", "content": vision_desc})
            for token in self.openrouter.stream(messages):
                full_response += token
                yield token, action
                if any(kw in token.lower() for kw in {"enroll", "enrolling"}):
                    action = "enroll"
                elif any(kw in token.lower() for kw in {"alert", "security", "escalate"}):
                    action = "alert"
        elif self.openrouter.available:
            for token in self.openrouter.stream(messages):
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
        """Non-streaming convenience: full response + action."""
        if self.openrouter.available:
            messages = self._build_context(identity, face_data,
                                           self._memory_for(identity), user_input)
            response = self.openrouter.complete(messages)
            action = self._decide(identity, user_input, response)
        else:
            mem = self._memory_for(identity)
            if user_input:
                mem.add_interaction("user", user_input)
            response = self._fallback(identity, user_input, mem)
            mem.add_interaction("assistant", response)
            action = self._decide(identity, user_input, response)
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
