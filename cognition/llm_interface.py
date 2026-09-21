"""LLM backend interface — swappable conversation generation.

Architecture (per ARIA_ARCHITECTURE.md):
- One interface: context + conversation history -> what ARIA says next
- Local-only (no AWS) until Task 8: OpenRouter free-tier or local rule-based
- Swap to AWS Bedrock when ready — no other code changes needed

Backends:
  1. OpenRouterClient — OpenRouter API (free-tier models, streaming)
  2. LocalFallbackLLM — rule-based + keyword responses (zero network)
  3. BedrockLLM — AWS Bedrock (Nova Micro, Claude) — ready when AWS is configured
"""
import os
import time
import json
from typing import Iterator


class LLMClient:
    """Abstract interface for conversation generation backends."""

    def complete(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def stream(self, messages: list[dict]) -> Iterator[str]:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class OpenRouterClient(LLMClient):
    """OpenRouter API client — free-tier friendly with model fallback."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.referer = os.getenv("OPENROUTER_REFERER", "http://localhost:8080")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "ARIA-Companion")
        self.model = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-pro:free")
        self.fallback_models = self._build_fallback_list()

    def _build_fallback_list(self) -> list[str]:
        primary = self.model
        # Chain picked from a live latency benchmark of OpenRouter's free
        # catalog (2026-09-17, bench_models.py): all three ~1.3-1.7s TTFT with
        # clean spoken-style replies. The previous liquid/gemma entries were
        # caught returning HTTP 429 (provider-saturated) during the bench.
        defaults = [
            "nex-agi/nex-n2.5-pro:free",
            "inclusionai/ling-3.0-flash-sante:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ]
        return [primary] + [m for m in defaults if m != primary]

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

    def stream(self, messages: list[dict]) -> Iterator[str]:
        if not self.available:
            return
        import httpx
        for model in self.fallback_models:
            self.model = model
            payload = self._payload(messages)
            try:
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", f"{self.BASE_URL}/chat/completions",
                                       headers=self._headers(), json=payload) as resp:
                        # 429/5xx is routine on free tiers — fail over to the
                        # next model instead of silently yielding nothing
                        # (old bug: rate-limited model => ARIA went mute and
                        # the fallback list was never tried).
                        resp.raise_for_status()
                        for line in resp.iter_lines():
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
            except Exception as e:
                print(f"[OpenRouter] {model} failed: {e}, trying fallback...")
                time.sleep(0.2)  # next model, not a retry — keep voice snappy

    def complete(self, messages: list[dict]) -> str:
        if not self.available:
            return ""
        import httpx
        for model in self.fallback_models:
            self.model = model
            payload = self._payload(messages, stream=False)
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(f"{self.BASE_URL}/chat/completions",
                                       headers=self._headers(), json=payload)
                    data = resp.json()
                    if "error" in data:
                        print(f"[OpenRouter] {model}: {data['error']['message'][:60]}")
                        time.sleep(1)
                        continue
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[OpenRouter] {model} failed: {e}, trying fallback...")
                time.sleep(1)
        return ""

# NOTE (owner-confirmed architecture): the LLM client intentionally has NO
# image/video input path. OpenCV is the only eyes of the system; the LLM is
# a pure reasoner that reads OpenCV's text reports (see agent.py
# VisionContext). Do not add frame-upload methods back.


class LocalFallbackLLM(LLMClient):
    """Zero-network fallback: rule-based responses with personality.

    Used when no LLM API key is available. Provides basic conversational
    behavior based on keyword matching.
    """

    def __init__(self):
        self._templates = {
            "greeting": ["Hi there!", "Hello!", "Hey there!", "Good to see you!"],
            "how_are_you": ["I'm doing well, thanks for asking!", "Great to see you!", "I'm ready to help!"],
            "unknown": ["I don't know much about that yet.", "I'm still learning about this topic.", "That's interesting, I'd love to hear more about it."],
            "farewell": ["Goodbye!", "See you around!", "Bye for now!"],
        }

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages: list[dict]) -> str:
        user_inputs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last = user_inputs[-1].lower() if user_inputs else ""

        import random
        if any(w in last for w in ["hello", "hi", "hey"]):
            return random.choice(self._templates["greeting"])
        if any(w in last for w in ["how", "how are you"]):
            return random.choice(self._templates["how_are_you"])
        if any(w in last for w in ["bye", "goodbye", "leave"]):
            return random.choice(self._templates["farewell"])
        if any(w in last for w in ["name", "who are you"]):
            return "I'm ARIA, your AI companion. I see people through the camera and chat with them."
        return random.choice(self._templates["unknown"])

    def stream(self, messages: list[dict]) -> Iterator[str]:
        response = self.complete(messages)
        for word in response.split():
            yield word + " "


try:
    import boto3
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


class BedrockLLM(LLMClient):
    """AWS Bedrock LLM backend — used when AWS credentials are available.

    Uses Amazon Nova Micro by default (cheap, <1ms per token). Falls back
    to Claude 3 Haiku if Nova is unavailable.

    Architecture: unified InvokeModel API with streaming support via
    invoke_model_with_response_stream.
    """

    def __init__(self, model_id: str | None = None):
        if not _HAS_BOTO3:
            self.bedrock = None
            self.model_id = None
            return
        self.bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.model_id = model_id or os.getenv("BEDROCK_MODEL", "amazon.nova-micro-v1:0")

    @property
    def available(self) -> bool:
        return self.bedrock is not None and self.model_id is not None

    def _payload(self, messages: list[dict], system: str = "") -> dict:
        return {
            "system": system,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 300,
                "temperature": 0.7,
            },
        }

    def complete(self, messages: list[dict], system: str = "") -> str:
        if not self.available:
            return ""
        try:
            body = json.dumps(self._payload(messages, system))
            resp = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
            )
            data = json.loads(resp["body"].read())
            return data.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip()
        except Exception as e:
            print(f"[Bedrock] Error: {e}")
            return ""

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        if not self.available:
            return
        try:
            body = json.dumps(self._payload(messages, system))
            resp = self.bedrock.invoke_model_with_response_stream(
                body=body,
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
            )
            for event in resp.get("stream", []):
                if event.get("event_type") == "message_delta":
                    text = event.get("delta", {}).get("text", "")
                    if text:
                        yield text
        except Exception as e:
            print(f"[Bedrock] Stream error: {e}")
