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
    "inclusionai/ling-3.0-flash-sante:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
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

        # Architecture doctrine (owner-confirmed): OpenCV is the ONLY eyes.
        # The LLM never receives camera frames — only OpenCV's text facts
        # (already in messages via _build_context). The old stream_with_image
        # path that uploaded the frame as base64 for strangers is removed.
        if self.llm.available:
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


# ======================================================================
# Talkative upgrade (Phases 2-3) — GPT-Live-style split-brain + vision
#
# OpenAI's GPT-Live doctrine applied here with client delegation:
#   * a fast VOICE BRAIN owns speaking behavior (short spoken turns,
#     warmth, pacing) and decides when a question needs depth;
#   * the BACKEND BRAIN (VisionAgent's LLM + memory + tools) handles the
#     substantive answer while ARIA keeps the floor with fillers;
#   * what ARIA SEES is injected into every spoken turn.
# ======================================================================

VOICE_SYSTEM_PROMPT = """You are ARIA's speaking voice — a warm, curious AI companion people talk to out loud.

SPEAKING RULES (critical — this is read aloud by TTS):
- Reply in 1-2 short sentences. Never more unless directly asked for detail.
- Plain conversational words only. No markdown, lists, emoji, or symbols.
- Sound alive: react to what you see, use the person's name when you know it.
- End with ONE light question at most, and not on every turn.

DELEGATION:
If the visitor needs real reasoning, current events, math, code, detailed
facts, or step-by-step help, reply with EXACTLY:
[DELEGATE]
and nothing else. A deeper brain takes over and speaks afterwards.

CONTEXT FOR THIS TURN:
{context}
"""

_SENTENCE_ENDERS = (".", "!", "?")


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split streamed text into complete sentences + the unspoken remainder.

    Decimal points ("3.14") and abbreviations followed by a letter survive
    unsplit; a terminator followed by whitespace/end closes a sentence.
    """
    sentences, start = [], 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDERS:
            nxt = text[i + 1:i + 2]
            if nxt in (" ", "", "\n", "\t"):
                piece = text[start:i + 1].strip()
                if piece:
                    sentences.append(piece)
                start = i + 1
    return sentences, text[start:]


class VisionContext:
    """Compact "what ARIA sees" summary, refreshed each vision frame.

    Injected into the voice brain's context so spoken replies naturally
    reference the room ("I see you brought a friend...") — the fusion that
    makes ARIA more than a voice assistant.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.summary = "No vision input right now."
        self.recent_events: list[str] = []
        self.last_known_name: str | None = None
        self.objects: list[str] = []  # non-person object labels seen by OpenCV

    def update(self, face_results, detections=None):
        known = [fr.get("identity") for fr in face_results
                 if fr.get("authorized") and fr.get("identity") not in (None, "", "unknown")]
        strangers = sum(1 for fr in face_results if not fr.get("authorized"))
        dets = list(detections or [])
        people = sum(1 for d in dets if getattr(d, "label", "") == "person")
        # Object awareness crosses the eyes->brain bridge as TEXT labels
        # (OpenCV's job is seeing; the LLM only ever reads its reports).
        objects = []
        for d in dets:
            label = getattr(d, "label", "")
            if label and label != "person" and label not in objects:
                objects.append(label)
        with self.lock:
            self.objects = objects[:5]
            if known:
                self.last_known_name = known[0]
            if not face_results and not people:
                self.summary = "No one is in view right now."
            else:
                parts = []
                if known:
                    parts.append("Recognized: " + ", ".join(known) + ".")
                if strangers:
                    parts.append(f"{strangers} unrecognized visitor(s) here.")
                extra = people - len(face_results)
                if extra > 0:
                    parts.append(f"{extra} other person(s) in the background.")
                self.summary = " ".join(parts)

    def add_event(self, description: str):
        with self.lock:
            self.recent_events.append(description)
            self.recent_events = self.recent_events[-5:]

    def context_text(self) -> str:
        with self.lock:
            ev = (" Recent: " + " | ".join(self.recent_events[-3:])) if self.recent_events else ""
            obj = (" Nearby objects: " + ", ".join(self.objects) + ".") if self.objects else ""
            return self.summary + obj + ev


class VoiceBrain:
    """The fast spoken-turn model (GPT-Live "speaking behavior" layer).

    Streams a short conversational verdict; if the question needs the
    backend brain, the stream ends with a [DELEGATE] token and the caller
    hands off to VisionAgent's full pipeline.
    """

    _FILLERS = (
        "Hmm, let me think about that.",
        "One second...",
        "Good question — give me a moment.",
    )

    def __init__(self, agent: "VisionAgent"):
        self.agent = agent
        self.delegated = False
        # Dedicated FAST client for spoken turns (GPT-Live: the live model is
        # a different, smaller model than the backend brain). Small-talk only
        # needs 1-2 short sentences — a 2-3B model answers with far lower
        # first-token latency than the big backend model. OPT-IN: set
        # ARIA_VOICE_MODEL in .env to enable; otherwise the shared backend
        # LLM is used (keeps tests hermetic and behavior predictable).
        self._fast_llm = None
        # Benchmark-informed default (2026-09-17 bench_models.py): 1.4s TTFT,
        # clean 1-2 sentence spoken replies. Override with ARIA_VOICE_MODEL.
        fast_model = os.getenv("ARIA_VOICE_MODEL") or "inclusionai/ling-3.0-flash-sante:free"
        try:
            from llm_interface import OpenRouterClient
            # Hermeticity guard: only attach a dedicated fast client when the
            # backend brain itself is the real OpenRouter client (production).
            # With test stubs or the offline LocalFallbackLLM, the voice brain
            # just uses agent.llm — same channel as the backend.
            backend_is_openrouter = isinstance(
                getattr(self.agent, "llm", None), OpenRouterClient)
            if backend_is_openrouter and OpenRouterClient().available:
                client = OpenRouterClient()
                client.model = fast_model
                client.fallback_models = [fast_model] + [
                    m for m in client.fallback_models if m != fast_model]
                self._fast_llm = client
                print(f"[VoiceBrain] fast spoken-turn model: {fast_model}")
        except Exception as e:  # noqa: BLE001 — voice brain must never crash init
            print(f"[VoiceBrain] fast model unavailable ({e}); using backend LLM")

    @property
    def available(self) -> bool:
        return self.agent.llm.available

    def _messages(self, context: str, history: list[dict], user_text: str) -> list[dict]:
        msgs = [{"role": "system",
                 "content": VOICE_SYSTEM_PROMPT.format(context=context or "No vision input right now.")}]
        msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def reply_stream(self, context: str, history: list[dict], user_text: str | None) -> Iterator[str]:
        """Yield spoken-turn tokens; sets self.delegated on a [DELEGATE] verdict."""
        self.delegated = False
        prompt = user_text or "(The person just arrived and is looking at you — greet them warmly in one sentence.)"
        llm = self._fast_llm if self._fast_llm is not None else self.agent.llm
        if llm.available:
            tokens = llm.stream(self._messages(context, history, prompt))
        else:
            tokens = iter([self.agent._fallback("unknown", prompt,
                                                self.agent._memory_for("unknown"))])
        buf = ""
        for tok in tokens:
            buf += tok
            if not self.delegated and "[delegate" in buf.lower():
                self.delegated = True
                return
            yield tok
        if not self.delegated and not buf.strip():
            self.delegated = True  # empty answer -> treat as "needs the backend"

    def filler(self) -> str:
        import random
        return random.choice(self._FILLERS)


class VoiceSession:
    """Turn-taking engine: voice brain answers, delegates depth, speaks live.

    run_turn() BLOCKS the conversation thread (voice I/O is serialized there)
    but speech is enqueued sentence-by-sentence so playback runs while the
    LLM keeps streaming — first audible words land ~one sentence after the
    model starts, not after the whole answer completes.
    """

    NUDGE_MINUTES = 1.5   # idle minutes before a conversational nudge
    MAX_NUDGES = 2        # max nudges before ARIA stops poking the human
    CACHE_TTL = 60.0      # backend answers cached this long per question

    def __init__(self, agent: "VisionAgent", voice, mic=None, vision: VisionContext | None = None):
        self.agent = agent
        self.voice = voice
        self.mic = mic
        self.vision = vision
        self.brain = VoiceBrain(agent)
        self.history: list[dict] = []
        self.busy = False
        self.pending_action = "ask"
        self.last_activity = time.time()
        self._nudges = 0
        self._nudge_lock = threading.Lock()
        self._backend_cache: dict[tuple, tuple[float, str]] = {}
        self._stop_fillers = False

    # ------------------------------------------------------------------

    def touch_activity(self):
        """Mark real interaction (resets the nudge budget)."""
        self.last_activity = time.time()
        self._nudges = 0

    def run_turn(self, identity: str, face_data: dict,
                 user_text: str | None = None, max_fillers: int = 2) -> str:
        """Handle one conversational turn end-to-end (speaks; returns text)."""
        if self.busy:
            return ""
        self.busy = True
        self.pending_action = "ask"
        try:
            return self._run_turn(identity, face_data, user_text, max_fillers)
        finally:
            self.busy = False

    def _run_turn(self, identity: str, face_data: dict,
                  user_text: str | None, max_fillers: int) -> str:
        mem = self.agent._memory_for(identity)
        if user_text:
            mem.add_interaction("user", user_text)

        ctx = self.vision.context_text() if self.vision is not None else ""
        history = list(self.history)

        said_any = False
        fillers_used = 0
        self._stop_fillers = False

        def maybe_filler():
            nonlocal fillers_used
            if self._stop_fillers or said_any or fillers_used >= max_fillers:
                return
            fillers_used += 1
            self.voice.enqueue_speech(self.brain.filler())

        timer = threading.Timer(1.2, maybe_filler)
        timer.daemon = True
        if user_text:
            timer.start()

        voice_full = ""   # complete voice-brain text
        pending = ""      # not-yet-spoken remainder
        try:
            for tok in self.brain.reply_stream(ctx, history, user_text):
                voice_full += tok
                pending += tok
                sentences, pending = _split_sentences(pending)
                for s in sentences:
                    said_any = True
                    self.voice.enqueue_speech(s)
            if not self.brain.delegated and pending.strip():
                tail = pending.replace("[", "").replace("]", "").strip()
                if tail:
                    said_any = True
                    self.voice.enqueue_speech(tail)
        finally:
            self._stop_fillers = True
            timer.cancel()

        if self.brain.delegated:
            full = self._backend_deliver(identity, face_data, user_text, said_any)
        else:
            full = voice_full.replace("[", "").replace("]", "").strip()
            self._record(identity, full)

        self.history.append({"role": "user", "content": user_text or "(arrived)"})
        self.history.append({"role": "assistant", "content": full or ""})
        self.history = self.history[-12:]

        low = full.lower()
        if any(k in low for k in ("enroll", "look at the camera", "now authorized",
                                  "register you", "save your face")):
            self.pending_action = "enroll"
        elif any(k in low for k in ("alert", "security", "escalat")):
            self.pending_action = "alert"
        return full

    # ------------------------------------------------------------------

    def _backend_stream(self, identity: str, face_data: dict, user_text: str | None):
        for tok, _action in self.agent.stream_response(identity, face_data, user_text):
            yield tok

    def _backend_deliver(self, identity: str, face_data: dict,
                         user_text: str | None, voice_said: bool) -> str:
        key = (identity, (user_text or "").lower().strip())
        now = time.time()
        cached = self._backend_cache.get(key)
        if cached is not None and now - cached[0] < self.CACHE_TTL:
            for s in self._sentences_of(cached[1]):
                self.voice.enqueue_speech(s)
            return cached[1]

        if voice_said:
            self.voice.enqueue_speech("Okay, give me a second to think that through.")
        parts: list[str] = []
        buf = ""
        for piece in self._backend_stream(identity, face_data, user_text):
            parts.append(piece)
            buf += piece
            sentences, buf = _split_sentences(buf)
            for s in sentences:
                self.voice.enqueue_speech(s)
        if buf.strip():
            self.voice.enqueue_speech(buf.strip())
        answer = "".join(parts).strip()
        self._backend_cache[key] = (now, answer)
        return answer

    @staticmethod
    def _sentences_of(text: str) -> list[str]:
        out: list[str] = []
        rest = text.strip()
        while rest:
            sentences, rest = _split_sentences(rest)
            if not sentences:
                out.append(rest.strip())
                break
            out.extend(sentences)
        return [s for s in out if s]

    def _record(self, identity: str, answer: str):
        if not (answer or "").strip():
            return
        mem = self.agent._memory_for(identity)
        mem.add_interaction("assistant", answer.strip())
        mem.save()

    # ------------------------------------------------------------------

    def maybe_nudge(self, identity: str, face_data: dict) -> bool:
        """Speak a gentle nudge after long silence (rate-limited)."""
        if self.busy:
            return False
        with self._nudge_lock:
            idle = time.time() - self.last_activity
            if idle < self.NUDGE_MINUTES * 60 or self._nudges >= self.MAX_NUDGES:
                return False
            self._nudges += 1
            self.last_activity = time.time()
        try:
            mem = self.agent._memory_for(identity)
            if mem.interaction_count > 1 and mem.persona.purpose:
                who = mem.persona.name or identity
                text = f"By the way {who}, earlier you mentioned {str(mem.persona.purpose)[:40]}. How did that go?"
            else:
                text = "Still with me? I'm curious what's on your mind today."
        except Exception:
            text = "Still with me?"
        self.voice.enqueue_speech(text)
        self.history.append({"role": "assistant", "content": text})
        self.history = self.history[-12:]
        return True
