"""Core package — fundamental domain models, event structures, pure policies, and tool execution."""

from .events import Event, EventType
from .policy import (
    PolicyDecision,
    should_greet,
    should_interrupt,
    should_follow_up,
    should_nudge,
    should_proactive_remark,
)
from .tools import (
    ToolResult,
    TOOL_SCHEMAS,
    ToolDispatcher,
    format_tools_prompt,
    strip_tool_calls,
)

__all__ = [
    "Event",
    "EventType",
    "PolicyDecision",
    "should_greet",
    "should_interrupt",
    "should_follow_up",
    "should_nudge",
    "should_proactive_remark",
    "ToolResult",
    "TOOL_SCHEMAS",
    "ToolDispatcher",
    "format_tools_prompt",
    "strip_tool_calls",
]
