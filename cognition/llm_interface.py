"""LLM backend interface — swappable conversation generation and native tool calling.

Architecture (per ARIA_ARCHITECTURE.md & Phase 10):
- Unified interface: messages + tools -> LLMResponse (content, tool_calls, finish_reason)
- Native JSON tool calling across all clients (tools=[{"type": "function", ...}])
- Swappable backends:
  1. OpenRouterClient — OpenRouter API (free-tier / cloud models)
  2. OllamaClient — 100% local, offline OpenAI-compatible endpoint (e.g. Ollama, LMStudio, vLLM)
  3. LocalFallbackLLM — zero-network rule-based fallback with simulated tool calling
  4. BedrockLLM — AWS Bedrock (Nova Micro, Claude)
"""
from dataclasses import dataclass, field
import os
import time
import json
from typing import Any, Iterator


@dataclass
class LLMResponse:
    """Structured response from an LLM call supporting function/tool calls."""
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_message: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """Abstract interface for conversation generation backends."""

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Convenience method: returns textual content only."""
        res = self.chat_complete(messages, tools=tools)
        return res.content

    def chat_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        """Full completion method returning structured LLMResponse with tool_calls."""
        raise NotImplementedError

    def stream(self, messages: list[dict]) -> Iterator[str]:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class OpenRouterClient(LLMClient):
    """OpenRouter API client — free-tier friendly with model fallback and tool calling."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.referer = os.getenv("OPENROUTER_REFERER", "http://localhost:8080")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "ARIA-Companion")
        self.model = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-pro:free")
        self.fallback_models = self._build_fallback_list()

    def _build_fallback_list(self) -> list[str]:
        primary = self.model
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

    def _payload(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = True,
    ) -> dict:
        is_free = ":free" in self.model
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": 400,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if not is_free:
            payload["provider"] = {
                "sort": "latency",
                "allow_fallbacks": True,
                "data_collection": "deny",
            }
        return payload

    def chat_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        if not self.available:
            return LLMResponse(content="", finish_reason="error")

        import httpx
        for model in self.fallback_models:
            self.model = model
            payload = self._payload(messages, tools=tools, tool_choice=tool_choice, stream=False)
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                    data = resp.json()
                    if "error" in data:
                        print(f"[OpenRouter] {model}: {data['error']['message'][:60]}")
                        time.sleep(1)
                        continue

                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    content = (msg.get("content") or "").strip()
                    tool_calls = msg.get("tool_calls") or []
                    finish_reason = choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
                    return LLMResponse(
                        content=content,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                        raw_message=msg,
                    )
            except Exception as e:
                print(f"[OpenRouter] {model} failed: {e}, trying fallback...")
                time.sleep(1)

        return LLMResponse(content="", finish_reason="error")

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        res = self.chat_complete(messages, tools=tools)
        return res.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        if not self.available:
            return
        import httpx
        for model in self.fallback_models:
            self.model = model
            payload = self._payload(messages)
            try:
                with httpx.Client(timeout=30) as client:
                    with client.stream(
                        "POST",
                        f"{self.BASE_URL}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    ) as resp:
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
                time.sleep(0.2)


class OllamaClient(LLMClient):
    """Local Ollama / OpenAI-compatible endpoint client (zero network dependencies)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", str(timeout)))

    @property
    def available(self) -> bool:
        if os.getenv("OLLAMA_ENABLED", "1") == "0":
            return False
        import httpx
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=0.8)
            return r.status_code == 200
        except Exception:
            return False

    def _payload(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        return payload

    def chat_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        import httpx
        payload = self._payload(messages, tools=tools, tool_choice=tool_choice, stream=False)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = (msg.get("content") or "").strip()
                tool_calls = msg.get("tool_calls") or []
                finish_reason = choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    raw_message=msg,
                )
        except Exception as e:
            print(f"[Ollama] Request failed: {e}")
            return LLMResponse(content="", finish_reason="error")

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        res = self.chat_complete(messages, tools=tools)
        return res.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        import httpx
        payload = self._payload(messages, stream=True)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                return
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                continue
        except Exception as e:
            print(f"[Ollama] Stream error: {e}")


class LocalFallbackLLM(LLMClient):
    """Zero-network fallback: rule-based responses with simulated tool calling."""

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

    def chat_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        user_inputs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last = user_inputs[-1].lower() if user_inputs else ""

        # Simulated tool calling for offline testing & hermetic verification
        if tools:
            # Check if user prompted for visual inspection or color segmentation
            if any(w in last for w in ("inspect color", "check badge", "color test", "green")):
                for t in tools:
                    fn_name = t.get("function", {}).get("name", "")
                    if fn_name == "vision.inspect_color_hsv":
                        return LLMResponse(
                            content="Inspecting color region.",
                            tool_calls=[{
                                "id": "call_inspect_1",
                                "type": "function",
                                "function": {
                                    "name": "vision.inspect_color_hsv",
                                    "arguments": json.dumps({
                                        "bbox": [10, 10, 50, 50],
                                        "lower_hsv": [35, 50, 50],
                                        "upper_hsv": [85, 255, 255],
                                    }),
                                },
                            }],
                            finish_reason="tool_calls",
                        )

            elif any(w in last for w in ("crop", "enhance", "roi")):
                for t in tools:
                    fn_name = t.get("function", {}).get("name", "")
                    if fn_name == "vision.crop_and_enhance":
                        return LLMResponse(
                            content="Cropping ROI for enhancement.",
                            tool_calls=[{
                                "id": "call_crop_1",
                                "type": "function",
                                "function": {
                                    "name": "vision.crop_and_enhance",
                                    "arguments": json.dumps({"bbox": [0, 0, 100, 100], "enhance": True}),
                                },
                            }],
                            finish_reason="tool_calls",
                        )

            elif any(w in last for w in ("search memory", "remember")):
                for t in tools:
                    fn_name = t.get("function", {}).get("name", "")
                    if fn_name == "memory.search":
                        return LLMResponse(
                            content="Searching memory.",
                            tool_calls=[{
                                "id": "call_mem_1",
                                "type": "function",
                                "function": {
                                    "name": "memory.search",
                                    "arguments": json.dumps({"query": "project", "person": "Alice"}),
                                },
                            }],
                            finish_reason="tool_calls",
                        )

        # Standard keyword conversation matching
        import random
        if any(w in last for w in ["hello", "hi", "hey"]):
            resp_text = random.choice(self._templates["greeting"])
        elif any(w in last for w in ["how", "how are you"]):
            resp_text = random.choice(self._templates["how_are_you"])
        elif any(w in last for w in ["bye", "goodbye", "leave"]):
            resp_text = random.choice(self._templates["farewell"])
        elif any(w in last for w in ["name", "who are you"]):
            resp_text = "I'm ARIA, your AI companion. I see people through the camera and chat with them."
        else:
            resp_text = random.choice(self._templates["unknown"])

        return LLMResponse(content=resp_text, tool_calls=[], finish_reason="stop")

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        res = self.chat_complete(messages, tools=tools)
        return res.content

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
    """AWS Bedrock LLM backend — used when AWS credentials are available."""

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

    def chat_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        if not self.available:
            return LLMResponse(content="", finish_reason="error")
        try:
            body = json.dumps(self._payload(messages))
            resp = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
            )
            data = json.loads(resp["body"].read())
            text = data.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip()
            return LLMResponse(content=text, finish_reason="stop")
        except Exception as e:
            print(f"[Bedrock] Error: {e}")
            return LLMResponse(content="", finish_reason="error")

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        res = self.chat_complete(messages, tools=tools)
        return res.content

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
