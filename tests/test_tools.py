"""Stage 9 check: runs offline. It ends with 'All checks passed' when everything behaves."""
import asyncio  # the loop and the executors are async
import re  # the fake decider pulls numbers out of the message
import tempfile  # a throwaway folder for the note-saving test
from pathlib import Path

from ..app.config import settings
from ..app.agents.tools import (  # the pieces under test
    Tool, ToolCall, ToolRegistry, build_default_registry, execute_call, run_loop, validate,
)

def run(coro):  # run an async function from plain code
    return asyncio.run(coro)

class FakeKB:  # stands in for the Stage 6 knowledge base, so no embeddings API is needed
    async def search(self, query, top_k=3):
        return [{"source": "acme_api.txt", "score": 0.9, "text": "Rate limit is 100 requests per minute."}]


async def fake_decide(history, tools):  # stands in for the model: routes by keyword
    last = history[-1]
    if last["role"] == "tool":  # a tool result just came back: answer from it
        return {"content": f"Final answer built from: {last['content']}"}
    msg = last["content"].lower()
    if "add" in msg:
        a, b = [float(n) for n in re.findall(r"-?\d+\.?\d*", msg)[:2]]
        return {"tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": a, "b": b}}]}
    if "note:" in msg:
        return {"tool_calls": [{"id": "call_2", "name": "save_note", "arguments": {"text": msg.split("note:")[1]}}]}
    if "search:" in msg:
        return {"tool_calls": [{"id": "call_3", "name": "search_knowledge", "arguments": {"query": msg.split("search:")[1].strip()}}]}
    return {"content": "I cannot route that to a tool."}


async def endless_decide(history, tools):  # a model stuck in a loop: always calls a tool
    return {"tool_calls": [{"id": "x", "name": "add", "arguments": {"a": 1, "b": 1}}]}


tmp = Path(tempfile.mkdtemp()) / "notes.txt"  # notes go here, not into your project
registry = build_default_registry(kb=FakeKB(), notes_path=tmp)

# 1. registry rules
for bad_name in ("Add Tool", "1add", ""):
    try:
        registry.register(Tool(bad_name, "A description that is long enough.", {"type": "object"}, lambda a: 1))
        raise AssertionError(f"accepted bad name {bad_name!r}")
    except ValueError:
        pass
try:
    registry.register(Tool("add", "A description that is long enough.", {"type": "object"}, lambda a: 1))
    raise AssertionError("accepted a duplicate name")
except ValueError:
    pass
try:
    registry.register(Tool("short_desc", "too short", {"type": "object"}, lambda a: 1))
    raise AssertionError("accepted a short description")
except ValueError:
    pass
assert all("executor" not in d and "consequential" not in d for d in registry.describe())  # the model never sees these
print("PASS registry rules")

# 2. validator
schema = registry.get("add").input_schema
assert validate(schema, {"a": 1, "b": 2}) == []  # valid
assert any("required" in e for e in validate(schema, {"a": 1}))  # missing field
assert any("expected number" in e for e in validate(schema, {"a": "x", "b": 2}))  # wrong type
assert any("expected number" in e for e in validate(schema, {"a": True, "b": 2}))  # booleans are not numbers
assert any("unknown field" in e for e in validate(schema, {"a": 1, "b": 2, "c": 3}))  # unknown field rejected
assert any("maximum" in e for e in validate(registry.get("search_knowledge").input_schema, {"query": "x", "top_k": 99}))
print("PASS validator")

# 3. the loop with a pure tool
out = run(run_loop("please add 7 and 35", registry, fake_decide))
print("answer:", out["answer"])
assert out["stopped"] == "answer" and "42" in out["answer"] and out["trace"][0]["ok"]
print("PASS loop with a pure tool")

# 4. failures come back as results, not crashes
r = run(execute_call(registry, ToolCall("c", "nope", {})))
assert not r.ok and "unknown tool" in r.error
r = run(execute_call(registry, ToolCall("c", "get_time", {"timezone": "Not/AZone"})))
assert not r.ok  # the executor raised, and the loop survived
r = run(execute_call(registry, ToolCall("c", "add", {"a": 1})))
assert not r.ok and "invalid arguments" in r.error
print("PASS errors become results")

# 5. consequential tools are gated
out = run(run_loop("note: buy milk", registry, fake_decide))  # no confirm callback
assert "blocked" in out["answer"] and not tmp.exists()
out = run(run_loop("note: buy milk", registry, fake_decide, confirm=lambda tool, args: True))  # approved
assert tmp.exists() and tmp.read_text().strip() == "buy milk"
print("PASS consequential gate")

# 6. async executor (the Stage 6 search tool)
out = run(run_loop("search: rate limit", registry, fake_decide))
assert "100 requests" in out["answer"]
print("PASS async executor")

# 7. circuit breaker
out = run(run_loop("anything", registry, endless_decide, max_turns=3))
assert out["stopped"] == "max_turns" and out["turns"] == 3 and out["answer"] is None
print("PASS circuit breaker")

# 8. long results are truncated
big = ToolRegistry()
big.register(Tool("big", "Returns a very large text for testing.", {"type": "object"}, lambda a: {"text": "x" * 10_000}))
msg = run(execute_call(big, ToolCall("c", "big", {}))).to_message()
assert len(msg["content"]) <= settings.MAX_TOOL_RESULT_CHARS + len("...[truncated]")
print("PASS result truncation")

print("\nAll checks passed")
