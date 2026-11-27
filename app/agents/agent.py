import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..config import settings
from ..foundation.context import count_tokens
from .tools import Tool, ToolCall, ToolRegistry, ToolResult, execute_call, execute_calls

# 1. The explicit finish tools

FINISH_TOOL = Tool(
    name="finish",
    description="Use when you have the complete final answer for the user. Call it exactly once, with that answer.",
    input_schema={
        "type": "object",
        "properties": {"answer": {"type": "string", "minLength": 1}},
        "required": ["answer"],
    },
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
        text = message.get("content")  # tool_call messages have no text
        if (
            text
        ):  # only count real text, so count_tokens is never called with an empty string
            total += count_tokens(text)
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
        parallel: bool = False,
    ):
        self.registry = with_finish_tool(registry)
        self.decide = decide
        self.confirm = confirm
        self.max_turns = max_turns or settings.AGENT_MAX_TURNS
        self.max_tokens = max_tokens or settings.AGENT_MAX_TOKENS
        self.max_calls_per_turn = (
            max_calls_per_turn or settings.AGENT_MAX_CALLS_PER_TURN
        )
        self.require_finish = require_finish
        self.parallel = parallel

    async def _run_calls(self, calls: List[ToolCall]) -> List[ToolResult]:
        cap = self.max_calls_per_turn
        if not self.parallel:
            results: List[ToolResult] = []
            finished = False
            for index, call in enumerate(calls):
                if finished:
                    result = _skipped(call, "skipped: the run already finished")
                elif index >= cap:
                    result = _skipped(
                        call, f"skipped: at most {cap} tool calls are allowed per turn"
                    )
                else:
                    result = await execute_call(self.registry, call, self.confirm)
                results.append(result)
                if call.name == "finish" and result.ok:
                    finished = True
            return results
        finish_index = next(
            (i for i, c in enumerate(calls) if c.name == "finish"), None
        )
        last = len(calls) if finish_index is None else finish_index + 1
        runnable = calls[: min(last, cap)]
        results = list(await execute_calls(self.registry, runnable, self.confirm))
        for index in range(len(runnable), len(calls)):
            if index >= last:
                results.append(
                    _skipped(
                        calls[index],
                        "skipped: calls after finish in the same turn are not run",
                    )
                )
            else:
                results.append(
                    _skipped(
                        calls[index],
                        f"skipped: at most {cap} tool calls are allowed per turn",
                    )
                )
        return results

    async def run(self, user_message: str) -> AgentResult:
        history: List[dict] = [{"role": "user", "content": user_message}]
        steps: List[AgentStep] = []
        completed = 0  # turns that finished without ending the run

        def end(answer: Optional[str], reason: str, turns: int) -> AgentResult:
            return AgentResult(
                answer, reason, turns, history, steps, history_tokens(history)
            )

        for turn in range(1, self.max_turns + 1):
            if history_tokens(history) > self.max_tokens:
                return end(None, "token_budget", completed)
            decision = await maybe_await(self.decide(history, self.registry.describe()))
            thought = decision.get("thought", "")
            raw_calls = decision.get("tool_calls") or []
            if not raw_calls:
                text = decision.get("content", "")
                history.append({"role": "assistant", "content": text})
                if not self.require_finish:
                    return end(text, "answer", turn)
                history.append(
                    {
                        "role": "user",
                        "content": "Reminder: deliver your final answer by calling the finish tool.",
                    }
                )
                completed += 1
                continue
            history.append({"role": "assistant", "tool_calls": raw_calls})
            calls = [
                ToolCall(
                    raw.get("id", f"call_{turn}_{index}"),
                    raw.get("name", ""),
                    raw.get("arguments", {}),
                )
                for index, raw in enumerate(raw_calls)
            ]
            results = await self._run_calls(calls)
            final_answer: Optional[str] = None
            for call, result in zip(calls, results):
                message = result.to_message()
                history.append(message)
                steps.append(
                    AgentStep(
                        turn,
                        thought,
                        call.name,
                        call.arguments,
                        result.ok,
                        message["content"],
                        result.ms,
                    )
                )
                if call.name == "finish" and result.ok:
                    final_answer = call.arguments["answer"]
            if final_answer is not None:
                return end(final_answer, "finish", turn)
            completed += 1
        return end(None, "max_turns", completed)


def pretty_trace(result: AgentResult) -> None:
    for step in result.steps:
        print(f"[turn {step.turn}] thought: {step.thought or '-'}")
        print(f"    action: {step.tool}({json.dumps(step.arguments)})")
        print(
            f"    observe: [{'ok' if step.ok else 'ERROR'}] {step.observation} ({step.ms} ms)"
        )
    print(
        f"stopped: {result.stop_reason} after {result.turns} turn(s), about {result.tokens_used} tokens"
    )
    print(f"answer: {result.answer}")
