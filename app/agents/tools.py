"""
the tool interface:
    1. Describe
    2. decide
    3. execute
    4. observe
"""

import asyncio
import inspect
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import math
from ..config import settings

# 1. The tool itself:


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    executor: Callable[[dict], Any]
    consequential: bool = False
    timeout_s: Optional[float] = None
    coerce_args: bool = True


# 2: A small Json validator
_PY_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}
SUPPORTED_TYPES = ["object", "array", "string", "number", "integer", "boolean"]
_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_NUM_RE = re.compile(r"^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$")


def _type_ok(schema_type: str, value: Any) -> bool:
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _PY_TYPES[schema_type])


def check_schema(schema: dict, path: str = "$") -> None:
    schema_type = schema.get("type")
    if schema_type not in SUPPORTED_TYPES:
        raise ValueError(f"{path}: unsupported or missing schema type {schema_type!r}")
    for key, sub in schema.get("properties", {}).items():
        check_schema(sub, f"{path}.{key}")
    if schema_type == "array" and "items" in schema:
        check_schema(schema["items"], f"{path}[]")


def coerce(schema: dict, value: Any) -> Any:
    schema_type = schema.get("type")
    if schema_type == "integer":
        if isinstance(value, str):
            text = value.strip()
            if len(text) <= 30 and _INT_RE.match(text):
                return int(text)
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    if schema_type == "number":
        if isinstance(value, str):
            text = value.strip()
            if len(text) <= 40 and _NUM_RE.match(text):
                number = float(text)
                return number if math.isfinite(number) else value
        return value
    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            k: (coerce(properties[k], v) if k in properties else v)
            for k, v in value.items()
        }
    if schema_type == "array" and isinstance(value, list) and "items" in schema:
        return [coerce(schema["items"], v) for v in value]
    return value


def validate(schema: dict, value: Any, path: str = "$") -> List[str]:
    schema_type = schema.get("type")
    if schema_type is not None and not _type_ok(schema_type, value):
        return [f"{path}: expected {schema_type}, got {type(value).__name__}"]
    errors: List[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if schema_type == "object":
        properties = schema.get("properties", {})
        for field_name in schema.get("required", []):
            if field_name not in value:
                errors.append(f"{path}.{field_name}: required field is missing")
        for key, sub_value in value.items():
            if key in properties:
                errors.extend(validate(properties[key], sub_value, f"{path}.{key}"))
            elif not schema.get("additionalProperties", False):
                errors.append(f"{path}.{key}: unknown field")
    elif schema_type == "array":
        if "items" in schema:
            for i, item in enumerate(value):
                errors.extend(validate(schema["items"], item, f"{path}[{i}]"))
    elif schema_type == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    elif schema_type in ("number", "integer"):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path}: must be a finite number")
        else:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(
                    f"{path}: {value} is below the minimum {schema['minimum']}"
                )
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(
                    f"{path}: {value} is above the maximum {schema['maximum']}"
                )
    return errors


def normalize_arguments(
    schema: dict, arguments: Any, coerce_args: bool = True
) -> Tuple[Any, List[str]]:
    fixed = coerce(schema, arguments) if coerce_args else arguments
    return fixed, validate(schema, fixed)


# 3. The registry (step 1: describe)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not _NAME_RE.match(tool.name):
            raise ValueError(
                f"invalid tool name {tool.name!r}: use lowercase letters, digits and underscores"
            )
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name {tool.name!r}")
        if len(tool.description.strip()) < 20:
            raise ValueError(
                f"{tool.name}: description is too short to tell the model when to use it."
            )
        if tool.input_schema.get("type") != "object":
            raise ValueError(f"{tool.name}: input_schema must have type 'object'")
        if tool.timeout_s is not None and tool.timeout_s <= 0:
            raise ValueError(f"{tool.name}: timeout_s must be positive")
        check_schema(tool.input_schema)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def describe(self) -> List[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]


# 4. Calls and results


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: Any = None
    error: Optional[str] = None
    ms: float = 0.0

    def to_message(self) -> dict:
        payload = self.content if self.ok else {"error": self.error}
        text = json.dumps(payload, ensure_ascii=False)
        limit = settings.MAX_TOOL_RESULT_CHARS
        if len(text) > limit:
            text = text[:limit] + "...[truncated]"
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": text,
        }


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _run_executor(tool: Tool, arguments: dict) -> Any:
    if inspect.iscoroutinefunction(tool.executor):
        return await tool.executor(arguments)
    output = await asyncio.to_thread(tool.executor, arguments)
    return await _maybe_await(output)


# 5. Executing one call (step 3: execute)


async def execute_call(
    registry: ToolRegistry, call: ToolCall, confirm: Optional[Callable] = None
) -> ToolResult:
    start = time.perf_counter()

    def finish(
        ok: bool, content: Any = None, error: Optional[str] = None
    ) -> ToolResult:
        return ToolResult(
            call.id,
            call.name,
            ok,
            content,
            error,
            round((time.perf_counter() - start) * 1000, 2),
        )

    tool = registry.get(call.name)
    if tool is None:
        return finish(
            False, error=f"unknown tool {call.name!r}. Available: {registry.names()}"
        )
    arguments, errors = normalize_arguments(
        tool.input_schema, call.arguments, tool.coerce_args
    )
    if errors:
        return finish(False, error="invalid arguments: " + "; ".join(errors))

    if tool.consequential:
        approved = False
        if confirm is not None:
            try:
                approved = await _maybe_await(confirm(tool, arguments))
            except Exception:
                approved = False
        if not approved:
            return finish(
                False,
                error=f"blocked: {tool.name!r} is consequential and was not approved",
            )
    timeout = tool.timeout_s or settings.TOOL_TIMEOUT_S
    task = asyncio.ensure_future(_run_executor(tool, arguments))
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return finish(
            False, error=f"timeout: {tool.name!r} did not finish within {timeout} s"
        )
    try:
        output = task.result()
    except Exception as exc:
        return finish(False, error=f"{type(exc).__name__}: {exc}")
    try:
        json.dumps(output)
    except (TypeError, ValueError) as exc:
        return finish(False, error=f"result is not JSON-serializable: {exc}")
    return finish(True, content=output)


async def execute_calls(
    registry: ToolRegistry,
    calls: List[ToolCall],
    confirm: Optional[Callable] = None,
    max_concurrency: Optional[int] = None,
) -> List[ToolResult]:
    results: List[Optional[ToolResult]] = [None] * len(calls)
    seen: set = set()
    runnable: List[int] = []
    max_concurrency = max_concurrency or settings.TOOL_MAX_CONCURRENCY
    for i, call in enumerate(calls):
        if call.id in seen:
            results[i] = ToolResult(
                call.id,
                call.name,
                False,
                None,
                f"duplicate call id {call.id!r}: each call in a turn needs its own id",
                0.0,
            )
        else:
            seen.add(call.id)
            runnable.append(i)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(i: int) -> None:
        async with semaphore:
            results[i] = await execute_call(registry, calls[i], confirm)

    def is_consequential(i: int) -> bool:
        tool = registry.get(calls[i].name)
        return tool is not None and tool.consequential

    await asyncio.gather(*(run_one(i) for i in runnable if not is_consequential(i)))
    for i in runnable:
        if is_consequential(i):
            results[i] = await execute_call(registry, calls[i], confirm)
    return results


# 6. The Loop


async def run_loop(
    user_msg: str,
    registry: ToolRegistry,
    decide: Callable,
    confirm: Optional[Callable] = None,
    max_turns: Optional[int] = None,
) -> dict:
    max_turns = max_turns or settings.MAX_TOOL_TURNS
    history: List[dict] = [{"role": "user", "content": user_msg}]
    trace: List[dict] = []
    for turn in range(1, max_turns + 1):
        decision = await _maybe_await(decide(history, registry.describe()))
        raw_calls = decision.get("tool_calls") or []
        if not raw_calls:
            answer = decision.get("content", "")
            history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "stopped": "answer",
                "turns": turn,
                "history": history,
                "trace": trace,
            }
        calls = [ToolCall(c["id"], c["name"], c["arguments"]) for c in raw_calls]
        history.append({"role": "assistant", "tool_calls": raw_calls})
        results = await execute_calls(registry, calls, confirm)
        for call, result in zip(calls, results):
            history.append(result.to_message())
            trace.append(
                {
                    "turn": turn,
                    "tool": call.name,
                    "ok": result.ok,
                    "error": result.error,
                    "ms": result.ms,
                }
            )

    return {
        "answer": None,
        "stopped": "max_turns",
        "turns": max_turns,
        "history": history,
        "trace": trace,
    }


# 7. Example tools


def _add(args: dict) -> dict:  # pure: no side effects
    return {"sum": args["a"] + args["b"]}


def _get_time(args: dict) -> dict:  # pure
    name = args.get("timezone", "UTC")  # default to UTC
    tz = (
        timezone.utc if name == "UTC" else ZoneInfo(name)
    )  # raises an error for unknown names; execute_call catches it
    return {"now": datetime.now(tz).isoformat(timespec="seconds"), "timezone": name}


def make_search_knowledge(
    kb,
) -> Callable:  # wraps the Stage 6 knowledge base as an async tool
    async def _search(args: dict) -> dict:
        hits = await kb.search(args["query"], args.get("top_k", 3))  # semantic search
        return {
            "results": [  # keep only matches above the relevance threshold
                {"source": h["source"], "score": h["score"], "text": h["text"]}
                for h in hits
                if h["score"] >= settings.MIN_SCORE
            ]
        }

    return _search


def make_save_note(path: Path) -> Callable:  # consequential: it writes to disk
    def _save(args: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if needed
        with path.open("a", encoding="utf-8") as f:  # append, never overwrite
            f.write(args["text"].strip() + "\n")
        return {
            "saved": True,
            "file": path.name,
        }  # return the file NAME only, not the full path

    return _save


def build_default_registry(kb=None, notes_path: Optional[Path] = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Use when the user asks for the sum of two numbers. Do not use for subtraction, products or algebra.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            executor=_add,
        )
    )
    registry.register(
        Tool(
            name="get_time",
            description="Use when the user asks what time it is now. Do not use for past dates or scheduling.",
            input_schema={
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": [],
            },
            executor=_get_time,
        )
    )
    if kb is not None:  # only available when a knowledge base is supplied
        registry.register(
            Tool(
                name="search_knowledge",
                description="Use when the user asks about Acme policies, the API, deployments or office rules. "
                "Do not use for general knowledge questions.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
                executor=make_search_knowledge(kb),
            )
        )
    registry.register(
        Tool(
            name="save_note",
            description="Use when the user explicitly asks to save a note. Do not use unless they ask for it.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 500}
                },
                "required": ["text"],
            },
            executor=make_save_note(
                notes_path
                or Path(__file__).resolve().parent.parent / ".cache" / "notes.txt"
            ),
            consequential=True,  # writes to disk, so it must be approved
        )
    )
    return registry
