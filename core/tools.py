"""Tool and capability layer — 'LLM requests, system executes' pattern (Phase 6).

Implements exactly four tools:
1. memory.search   — Query past episodic records and semantic facts.
2. memory.store    — Persist a fact/preference (enforcing MemoryWriter stranger gate).
3. vision.get_state— Inspect current visual observations (detected objects, faces, lighting).
4. speech.speak    — Explicitly enqueue and trigger spoken audio.

Controlled by a thin, decoupled ToolDispatcher with structured invocation logging.
"""
from dataclasses import dataclass, field
from typing import Any, Callable
import json
import re


@dataclass
class ToolResult:
    """Represents the execution result of a tool call."""
    tool: str
    success: bool
    output: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


# ----------------------------------------------------------------------
# Tool Schemas / Declarations
# ----------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "memory.search": {
        "name": "memory.search",
        "description": "Search past episodic records and semantic facts scoped to a person.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword or topic to search for."},
                "person": {"type": "string", "description": "Identity of the person to search memories for."},
                "limit": {"type": "integer", "description": "Max results to return (default: 5)."},
            },
            "required": ["query"],
        },
    },
    "memory.store": {
        "name": "memory.store",
        "description": "Store a persistent fact or note for an enrolled person (unenrolled strangers blocked).",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Fact key or category (e.g. 'project', 'editor')."},
                "value": {"type": "string", "description": "Fact value or detail to remember."},
                "person": {"type": "string", "description": "Identity of the enrolled person."},
            },
            "required": ["key", "value", "person"],
        },
    },
    "vision.get_state": {
        "name": "vision.get_state",
        "description": "Retrieve active visual observations: detected objects, visible faces, lighting.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    "speech.speak": {
        "name": "speech.speak",
        "description": "Speak a response aloud via the voice synthesizer.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesize and speak aloud."},
                "interrupt": {"type": "boolean", "description": "Whether to interrupt current speech."},
            },
            "required": ["text"],
        },
    },
}


def format_tools_prompt() -> str:
    """Format available tools as a short prompt section for LLMs."""
    return """AVAILABLE TOOLS:
You can request tool actions when needed. Use this exact syntax:
[TOOL: tool_name({"arg1": "value1", ...})]

Available tools:
- memory.search({"query": "...", "person": "..."}) - Search memories for a person
- memory.store({"key": "...", "value": "...", "person": "..."}) - Store a fact for an enrolled person
- vision.get_state({}) - Get current visual observations (faces, objects, lighting)
- speech.speak({"text": "..."}) - Explicitly trigger spoken speech
"""


def strip_tool_calls(text: str) -> str:
    """Remove tool invocation markup from spoken text so TTS only speaks natural speech."""
    if not text:
        return ""
    cleaned = re.sub(r"\[TOOL:\s*[\w\.]+\s*\(.*?\)\]", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*\{.*?\"tool\":.*?\}\s*```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ----------------------------------------------------------------------
# Tool Handlers
# ----------------------------------------------------------------------

def _handle_memory_search(agent: Any, arguments: dict[str, Any]) -> ToolResult:
    query = str(arguments.get("query", "")).strip()
    person = arguments.get("person")
    limit = int(arguments.get("limit", 5))

    if not person and hasattr(agent, "memory"):
        # Default to first non-unknown person if available
        known = [k for k in getattr(agent, "memory", {}) if k not in ("unknown", "")]
        person = known[0] if known else "unknown"

    person = person or "unknown"

    from memory.retrieval import retrieve
    mem = None
    if hasattr(agent, "_memory_for"):
        mem = agent._memory_for(person)
    elif hasattr(agent, "memory") and hasattr(agent.memory, "get"):
        mem = agent.memory.get(person)

    if mem is None and person not in ("unknown", ""):
        try:
            from memory.context_memory import PersonMemory
            mem = PersonMemory.load(person)
            if hasattr(agent, "memory") and isinstance(agent.memory, dict):
                agent.memory[person] = mem
        except Exception:
            pass

    ret = retrieve(query=query, person=person, memory_obj=mem, limit=limit)
    out = {
        "query": query,
        "person": person,
        "facts": ret.get("semantic", {}).get("facts", {}),
        "episodes": [
            e.to_dict() if hasattr(e, "to_dict") else {"summary": str(e)}
            for e in ret.get("episodic", [])
        ],
        "context_snippet": ret.get("context_snippet", ""),
    }
    return ToolResult(tool="memory.search", success=True, output=out)


def _handle_memory_store(agent: Any, arguments: dict[str, Any]) -> ToolResult:
    key = str(arguments.get("key", "")).strip()
    value = str(arguments.get("value", "")).strip()
    person = str(arguments.get("person", "")).strip()

    if not key or not value or not person:
        return ToolResult(tool="memory.store", success=False, error="Missing required arguments: key, value, person")

    from memory.writer import MemoryWriter
    if not MemoryWriter.is_persistent_identity(person):
        return ToolResult(
            tool="memory.store",
            success=False,
            error=f"Privacy gate rejected: '{person}' is an unenrolled stranger and cannot write to disk",
        )

    mem = None
    if hasattr(agent, "_memory_for"):
        mem = agent._memory_for(person)
    elif hasattr(agent, "memory") and hasattr(agent.memory, "get"):
        mem = agent.memory.get(person)

    if mem is None:
        try:
            from memory.context_memory import PersonMemory
            mem = PersonMemory.load(person)
            if hasattr(agent, "memory") and isinstance(agent.memory, dict):
                agent.memory[person] = mem
        except Exception:
            pass

    if mem is None:
        return ToolResult(tool="memory.store", success=False, error=f"Could not resolve memory for '{person}'")

    if hasattr(mem, "add_fact"):
        mem.add_fact(key, value)
    elif hasattr(mem, "semantic"):
        mem.semantic.add_fact(key, value)

    if hasattr(mem, "save"):
        mem.save()
    else:
        MemoryWriter.save_person(mem)

    return ToolResult(tool="memory.store", success=True, output={"status": "stored", "key": key, "value": value, "person": person})


def _handle_vision_get_state(vision_ctx: Any, arguments: dict[str, Any]) -> ToolResult:
    if vision_ctx is None:
        return ToolResult(tool="vision.get_state", success=False, error="VisionContext unavailable")

    world = getattr(vision_ctx, "world_state", None)
    summary = vision_ctx.context_text() if hasattr(vision_ctx, "context_text") else ""

    known = list(getattr(world, "known_people", [])) if world else []
    unrec = getattr(world, "unrecognized_count", 0) if world else 0
    total_faces = len(known) + unrec

    out = {
        "summary": summary,
        "faces_count": total_faces,
        "known_people": known,
        "unrecognized_count": unrec,
        "objects": list(getattr(world, "objects", [])) if world else [],
        "ambient_light": getattr(world, "ambient_light", "normal") if world else "normal",
        "motion_level": getattr(world, "motion_level", 0.0) if world else 0.0,
    }
    return ToolResult(tool="vision.get_state", success=True, output=out)


def _handle_speech_speak(voice: Any, arguments: dict[str, Any]) -> ToolResult:
    text = str(arguments.get("text", "")).strip()
    interrupt = bool(arguments.get("interrupt", False))

    if not text:
        return ToolResult(tool="speech.speak", success=False, error="Missing required argument: text")

    if voice is None:
        return ToolResult(tool="speech.speak", success=False, error="Voice synthesizer unavailable")

    if hasattr(voice, "speak_async"):
        voice.speak_async(text)
    elif hasattr(voice, "enqueue_speech"):
        voice.enqueue_speech(text)
    elif hasattr(voice, "speak"):
        voice.speak(text, interrupt=interrupt)

    return ToolResult(tool="speech.speak", success=True, output={"spoken": text, "interrupted": interrupt})


# ----------------------------------------------------------------------
# ToolDispatcher
# ----------------------------------------------------------------------

class ToolDispatcher:
    """Dispatches tool calls requested by the LLM or internal workflows."""

    def __init__(
        self,
        agent: Any = None,
        voice: Any = None,
        vision_ctx: Any = None,
    ):
        self.agent = agent
        self.voice = voice
        self.vision_ctx = vision_ctx
        self.call_log: list[dict[str, Any]] = []

        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "memory.search": lambda args: _handle_memory_search(self.agent, args),
            "memory.store": lambda args: _handle_memory_store(self.agent, args),
            "vision.get_state": lambda args: _handle_vision_get_state(self.vision_ctx, args),
            "speech.speak": lambda args: _handle_speech_speak(self.voice, args),
        }

    def register(self, tool_name: str, handler: Callable[[dict[str, Any]], ToolResult]):
        """Register a custom tool handler."""
        self._handlers[tool_name] = handler

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Execute a tool by name with arguments and log the invocation."""
        args = arguments or {}
        handler = self._handlers.get(tool_name)
        if not handler:
            res = ToolResult(tool=tool_name, success=False, error=f"Unknown tool: '{tool_name}'")
            self._log_call(tool_name, args, res)
            return res

        try:
            res = handler(args)
        except Exception as e:
            res = ToolResult(tool=tool_name, success=False, error=str(e))

        self._log_call(tool_name, args, res)
        print(f"[Tool] Executed {tool_name}({args}) -> {'OK' if res.success else 'FAILED'}: {res.output or res.error}")
        return res

    def _log_call(self, tool_name: str, arguments: dict[str, Any], result: ToolResult):
        self.call_log.append({
            "tool": tool_name,
            "arguments": dict(arguments),
            "success": result.success,
            "output": result.output,
            "error": result.error,
        })

    def parse_calls(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Extract tool calls from text (supporting [TOOL: name(args)] and JSON blocks)."""
        calls: list[tuple[str, dict[str, Any]]] = []
        if not text:
            return calls

        # 1. Bracket syntax: [TOOL: name({"arg": "val"})]
        bracket_pattern = re.compile(r"\[TOOL:\s*([\w\.]+)\s*\((.*?)\)\]", re.DOTALL)
        for match in bracket_pattern.finditer(text):
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}
            calls.append((tool_name, args))

        # 2. JSON code block: ```json {"tool": "name", "arguments": {...}} ```
        json_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        for match in json_block_pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "tool" in data:
                    calls.append((data["tool"], data.get("arguments", {})))
            except json.JSONDecodeError:
                pass

        return calls

    def parse_and_execute(self, text: str) -> list[ToolResult]:
        """Parse all requested tool calls in text and execute them."""
        parsed = self.parse_calls(text)
        return [self.execute(tool_name, args) for tool_name, args in parsed]
