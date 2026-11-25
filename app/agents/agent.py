import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..config import settings
from ..foundation.context import count_tokens
from .tools import Tool, ToolCall, ToolRegistry, ToolResult, execute_call

# 1. The explicit finish tools

FINISH_TOOL = Tool(
    name="finish",
    description="Use when you have the complete final answer for the user. Call it exactly once, with that answer.",
    input_schema={"type": "object", "properties": {"answer": {"type": "string", "minLength": 1}}, "required": ["answer"]},
    executor=lambda args: {"received": True},
)

def with_finish_tool(registry: ToolRegistry) -> ToolRegistry:
    if registry.get("finish") is None:
        registry.register(FINISH_TOOL)
    return registry

# 2. Records

@dataclass
class AgentStep:
    turn: int 
    thought: str 
    tool: str 
    arguments: dict
    ok: bool
    observation: str
    ms: float

@dataclass
class AgentResult:
    answer: Optional[str]
    stop_reason: str
    turns: int
    history: List[dict]
    steps: List[AgentStep] = field(default_factory=list)
    tokens_used: int = 0

def history_tokens(history: List[dict]) -> int:
    total = 0
    for message in history:
        total += count_tokens(message.get("content") or "")
        if message.get("tool_calls"):
            total += count_tokens(json.dumps(message["tool_calls"]))
    return total

async def maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value

def _skipped(call: ToolCall, reason: str) -> ToolResult:
    return ToolResult(call.id, call.name, False, None, reason, 0.0)

# 3. The loop
class AgentLoop:
    def __init__(
        self,
        registry: ToolRegistry,
        decide: Callable,
        confirm: Optional[Callable] = None,
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_calls_per_turn: Optional[int] = None,
        require_finish: bool = False,
    ):
        self.registry = with_finish_tool(registry)
        self.decide = decide
        self.confirm = confirm
        self.max_turns = max_turns or settings.AGENT_MAX_TURNS
        self.max_tokens = max_tokens or settings.AGENT_MAX_TOKENS
        self.max_calls_per_turn = max_calls_per_turn or settings.AGENT_MAX_CALLS_PER_TURN
        self.require_finish = require_finish

    async def run(self, user_message: str) -> AgentResult:
        history: List[dict] = [{"role": "user", "content": user_message}]
        steps: List[AgentStep] = []

        def end(answer: Optional[str], reason: str, turns: int) -> AgentResult:
            return AgentResult(answer, reason, turns, history, steps, history_tokens(history))

        for turn in range(1, self.max_turns + 1):
            if history_tokens(history) > self.max_tokens:
                return end(None, "token_budget", turn - 1)
            decision = await maybe_await(self.decide(history, self.registry.describe()))
            thought = decision.get("thought", "")
            raw_calls = decision.get("tool_calls") or []
            if not raw_calls:
                text = decision.get("content", "")
                history.append({"role": "assistant", "content": text})
                if not self.require_finish:
                    return end(text, "answer", turn)
                history.append({"role": "user", "content": "Reminder: deliver your final answer by calling the finish tool."})
                continue
            history.append({"role": "assistant", "tool_calls": raw_calls})
            final_answer: Optional[str] = None
            for index, raw in enumerate(raw_calls):
                call = ToolCall(
                    raw.get("id", f"call_{turn}_{index}"), raw.get("name", ""), raw.get("arguments", {})
                )
                if final_answer is not None:
                    result = _skipped(call, "skipped: the run already finished")
                elif index >= self.max_calls_per_turn:
                    result = _skipped(call, f"skipped: at most {self.max_calls_per_turn} tool calls are allowed per turn")
                else:
                    result = await execute_call(self.registry, call, self.confirm)
                message = result.to_message()
                history.append(message)
                steps.append(AgentStep(turn, thought, call.name, call.arguments, result.ok, message["content"], result.ms))
                if call.name == "finish" and result.ok:
                    final_answer = call.arguments["answer"]
            if final_answer is not None:
                return end(final_answer, "finish", turn)
        return end(None, "max_turns", self.max_turns)

def pretty_trace(result: AgentResult) -> None:
    for step in result.steps:
        print(f"[turn {step.turn}] thought: {step.thought or '-'}")
        print(f"    action: {step.tool}({json.dumps(step.arguments)})")
        print(f"    observe: [{'ok' if step.ok else 'ERROR'}] {step.observation} ({step.ms} ms)")
    print(f"stopped: {result.stop_reason} after {result.turns} turn(s), about {result.tokens_used} tokens")
    print(f"answer: {result.answer}")
