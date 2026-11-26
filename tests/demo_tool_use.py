"""The lesson's five-call batch, run through our dispatcher."""
import asyncio

from app.agents.demo_tools import build_lesson_registry
from app.agents.tools import ToolCall, execute_calls

registry = build_lesson_registry()
print("catalog (as presented to the model)")
for entry in registry.describe():
    print(f"  - {entry['name']}: {entry['description']}")

batch = [
    ToolCall("u01", "add", {"a": 2, "b": 3}),
    ToolCall("u02", "multiply", {"a": "4", "b": 5}),  # "4" is repaired to 4
    ToolCall("u03", "classify", {"status": "in_progress"}),  # not in the enum
    ToolCall("u04", "classify", {"status": "open"}),
    ToolCall("u05", "subtract", {"a": 1, "b": 2}),  # no such tool
]
print("\nparallel dispatch (5 calls in one turn)")
for r in asyncio.run(execute_calls(registry, batch)):
    print(f"  {r.call_id} {'OK ' if r.ok else 'ERR'}: {r.content if r.ok else r.error}")
