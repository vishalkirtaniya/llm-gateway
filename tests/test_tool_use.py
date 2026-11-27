"""Stage 12 check: tool use beyond the basic loop. Runs offline; ends with 'All checks passed'."""

import asyncio  # everything here is async
import tempfile  # throwaway folders
import time  # a blocking tool for the timeout test
from pathlib import Path

from app.agents.agent import AgentLoop
from app.agents.demo_tools import build_lesson_registry, register_document_tools
from app.agents.tools import (
    Tool,
    ToolCall,
    ToolRegistry,
    build_default_registry,
    execute_call,
    execute_calls,
    normalize_arguments,
)


def run(coro):  # run an async function from plain code
    return asyncio.run(coro)


def call(cid, name, **arguments):  # a ToolCall
    return ToolCall(cid, name, arguments)


def raw(cid, name, **arguments):  # a call in the dict shape a model produces
    return {"id": cid, "name": name, "arguments": arguments}


def turn(thought, *calls):  # one scripted model turn
    return {"thought": thought, "tool_calls": list(calls)}


class Scripted:  # a fake model: replays a fixed script
    def __init__(self, script):
        self.script, self.cursor = list(script), 0

    async def __call__(self, history, tools):
        if self.cursor >= len(self.script):
            return {"content": "script finished"}
        self.cursor += 1
        return self.script[self.cursor - 1]


# 1. argument repair: harmless fixes yes, ambiguous or dangerous values no
schema = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "x": {"type": "number", "minimum": 0}},
    "required": ["a"],
}
fixed, errs = normalize_arguments(schema, {"a": "4"})
assert errs == [] and fixed == {"a": 4} and type(fixed["a"]) is int
assert normalize_arguments(schema, {"a": 4.0})[1] == []  # 4.0 becomes 4
for bad in (
    "4.5",
    "abc",
    "",
    True,
    4.5,
    "\u0664",
    "1" * 5000,
):  # includes an Arabic-Indic digit and a huge integer string
    assert normalize_arguments(schema, {"a": bad})[1], f"accepted a={bad!r}"
for bad in (
    "nan",
    "inf",
    "-inf",
    "1e999",
    float("nan"),
    float("inf"),
):  # NaN used to slip past min/max
    assert normalize_arguments(schema, {"a": 1, "x": bad})[1], f"accepted x={bad!r}"
assert normalize_arguments(schema, {"a": 1, "x": "2.5"})[1] == []
assert any("minimum" in e for e in normalize_arguments(schema, {"a": 1, "x": "-1"})[1])
print("PASS argument repair")

# 2. the lesson's five-call batch: order kept, ids matched, repaired and rejected calls as expected
lesson = build_lesson_registry()
batch = [
    call("u01", "add", a=2, b=3),
    call("u02", "multiply", a="4", b=5),
    call("u03", "classify", status="in_progress"),
    call("u04", "classify", status="open"),
    call("u05", "subtract", a=1, b=2),
]
results = run(execute_calls(lesson, batch))
assert [r.call_id for r in results] == ["u01", "u02", "u03", "u04", "u05"]
assert [r.ok for r in results] == [True, True, False, True, False]
assert results[0].content == {"sum": 5} and results[1].content == {"product": 20}
assert "not one of" in results[2].error and "unknown tool" in results[4].error
print("PASS lesson batch")

# 3. calls really overlap, finish in a different order, and are still matched by id
state, done_order = {"active": 0, "peak": 0}, []


async def probe(args):  # records how many calls run at the same moment
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    await asyncio.sleep(args["delay"])
    done_order.append(args["tag"])
    state["active"] -= 1
    return {"tag": args["tag"]}


probe_schema = {
    "type": "object",
    "properties": {
        "tag": {"type": "string"},
        "delay": {"type": "number", "minimum": 0},
    },
    "required": ["tag", "delay"],
}
reg = ToolRegistry()
reg.register(
    Tool("probe", "A probe tool used to observe concurrency.", probe_schema, probe)
)
three = [
    call("a", "probe", tag="slow", delay=0.15),
    call("b", "probe", tag="mid", delay=0.08),
    call("c", "probe", tag="fast", delay=0.01),
]
results = run(execute_calls(reg, three))
assert state["peak"] == 3 and done_order == [
    "fast",
    "mid",
    "slow",
]  # all overlapped; finished out of request order
assert [r.call_id for r in results] == ["a", "b", "c"] and [
    r.content["tag"] for r in results
] == ["slow", "mid", "fast"]
state.update(active=0, peak=0)
done_order.clear()
run(execute_calls(reg, three, max_concurrency=1))  # a limit of 1 forces one at a time
assert state["peak"] == 1 and done_order == ["slow", "mid", "fast"]
print("PASS parallel dispatch and id matching")

# 4. consequential tools run one at a time, in the requested order, and only when approved
state.update(active=0, peak=0)
done_order.clear()
gated = ToolRegistry()
gated.register(
    Tool(
        "write_probe",
        "A probe tool that pretends to change something.",
        probe_schema,
        probe,
        consequential=True,
    )
)
writes = [
    call("x", "write_probe", tag="x", delay=0.05),
    call("y", "write_probe", tag="y", delay=0.03),
    call("z", "write_probe", tag="z", delay=0.01),
]
run(execute_calls(gated, writes, confirm=lambda tool, args: True))
assert state["peak"] == 1 and done_order == ["x", "y", "z"]
assert all(
    not r.ok and "blocked" in r.error for r in run(execute_calls(gated, writes))
)  # no approval callback: all denied
print("PASS consequential calls run serially")

# 5. timeouts: async and blocking tools are cut off; a tool's own TimeoutError is not mistaken for ours


async def sleepy(args):
    await asyncio.sleep(5)


def blocking(args):
    time.sleep(0.4)


def upstream(args):
    raise TimeoutError("upstream was slow")


slow = ToolRegistry()
for name, fn in (("sleepy", sleepy), ("blocking", blocking), ("upstream", upstream)):
    slow.register(
        Tool(
            name, "A tool used to test timeouts.", {"type": "object"}, fn, timeout_s=0.1
        )
    )
start = time.perf_counter()
r = run(execute_call(slow, call("t1", "sleepy")))
assert not r.ok and "timeout" in r.error and time.perf_counter() - start < 1.5
assert "timeout" in run(execute_call(slow, call("t2", "blocking"))).error
r = run(execute_call(slow, call("t3", "upstream")))
assert not r.ok and "TimeoutError" in r.error and "did not finish" not in r.error
print("PASS timeouts")

# 6. duplicate ids are rejected, and a broken approval callback means "denied"
dup = run(
    execute_calls(
        lesson, [call("same", "add", a=1, b=2), call("same", "add", a=3, b=4)]
    )
)
assert dup[0].ok and not dup[1].ok and "duplicate" in dup[1].error


def broken_confirm(tool, args):
    raise RuntimeError("approval service is down")


notes = Path(tempfile.mkdtemp()) / "notes.txt"
r = run(
    execute_call(
        build_default_registry(notes_path=notes),
        call("n", "save_note", text="x"),
        confirm=broken_confirm,
    )
)
assert not r.ok and "blocked" in r.error and not notes.exists()
print("PASS duplicate ids and failing approval")

# 7. real filesystem tools stay inside their folder
root = Path(tempfile.mkdtemp())
docs = root / "knowledge"
docs.mkdir()
(docs / "a.txt").write_text("Rate limit is 100 per minute.", encoding="utf-8")
(root / "secret.txt").write_text("top secret", encoding="utf-8")
files = ToolRegistry()
register_document_tools(files, docs)
assert (
    "100"
    in run(execute_call(files, call("1", "read_document", name="a.txt"))).content[
        "text"
    ]
)
assert run(execute_call(files, call("2", "list_documents"))).content["documents"] == [
    "a.txt"
]
errors = set()
for name in (
    "../secret.txt",
    "/etc/passwd",
    "a.md",
    "missing.txt",
    "..",
    "a.txt/../../secret.txt",
):
    r = run(execute_call(files, call("3", "read_document", name=name)))
    assert not r.ok, f"read {name!r}"
    errors.add(r.error)
assert (
    len(errors) == 1
)  # every failure looks the same, so nothing is revealed about which paths exist
print("PASS document tools")

# 8. the agent loop in parallel mode: ids in call order, repaired arguments, cap, and finish rules
result = run(
    AgentLoop(
        build_lesson_registry(),
        Scripted(
            [
                turn(
                    "three independent calls",
                    raw("p1", "add", a=1, b=2),
                    raw("p2", "multiply", a="3", b=4),
                    raw("p3", "classify", status="open"),
                ),
                turn("done", raw("f", "finish", answer="3, 12, open")),
            ]
        ),
        parallel=True,
    ).run("go")
)
assert (
    result.stop_reason == "finish"
    and result.turns == 2
    and all(s.ok for s in result.steps)
)
assert [m["tool_call_id"] for m in result.history if m["role"] == "tool"] == [
    "p1",
    "p2",
    "p3",
    "f",
]
result = run(
    AgentLoop(
        build_lesson_registry(),
        Scripted(
            [
                turn("", raw("f", "finish", answer="done"), raw("a", "add", a=1, b=1)),
            ]
        ),
        parallel=True,
    ).run("go")
)
assert (
    result.stop_reason == "finish"
    and [s.ok for s in result.steps] == [True, False]
    and "after finish" in result.steps[1].observation
)
result = run(
    AgentLoop(
        build_lesson_registry(),
        Scripted(
            [
                turn(
                    "",
                    raw("a", "add", a=1, b=1),
                    raw("b", "add", a=2, b=2),
                    raw("c", "add", a=3, b=3),
                ),
                turn("", raw("f", "finish", answer="x")),
            ]
        ),
        parallel=True,
        max_calls_per_turn=2,
    ).run("go")
)
assert [s.ok for s in result.steps] == [
    True,
    True,
    False,
    True,
] and "at most 2" in result.steps[2].observation
print("PASS agent loop in parallel mode")

print("\nAll checks passed")
