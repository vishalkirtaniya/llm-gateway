"""Stage 13 check: ReWOO and Plan-and-Execute. Runs offline; ends with 'All checks passed'."""

import asyncio  # everything here is async
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets "app" be found

import app.agents.planner as P  # the module itself, so a test can swap its chat function
from app.agents.planner import (
    Plan,
    PlanError,
    PlanStep,
    RefError,
    estimate_react_chars,
    evidence_text,
    execute_plan,
    execution_levels,
    find_references,
    make_llm_planner,
    make_llm_solver,
    plan_from_dict,
    resolve_value,
    run_rewoo,
    trajectory_from,
    validate_plan,
)
from app.agents.tools import Tool, ToolRegistry, build_default_registry


def run(coro):  # run an async function from plain code
    return asyncio.run(coro)


def step(
    sid, tool, /, **args
):  # a PlanStep (positional-only, so a tool argument may be called "name" or "tool")
    return PlanStep(sid, tool, args)


COUNT = {}  # how many times each tag ran
CALLS = {"round": 0}


def search(args):  # a tiny fact base
    q = args["query"].lower()
    if "boom" in q:
        raise RuntimeError("search backend is down")
    if "capital of france" in q:
        return {"answer": "Paris"}
    if "population of paris" in q:
        return {"answer": "11.2 million metro"}
    return {"answer": f"no result for {args['query']!r}"}


def round_million(args):
    CALLS["round"] += 1
    return {
        "rounded": f"{round(float(re.search(r'[0-9]+[.]?[0-9]*', args['text']).group()))} million"
    }


def work(args):  # counts its own runs; the tag "bad" always fails
    COUNT[args["tag"]] = COUNT.get(args["tag"], 0) + 1
    if args["tag"] == "bad":
        raise RuntimeError("this step always fails")
    return {"tag": args["tag"]}


def schema(*names):  # an object schema with the given string properties, all required
    return {
        "type": "object",
        "properties": {n: {"type": "string"} for n in names},
        "required": list(names),
    }


def build_registry(registry=None):
    reg = registry or ToolRegistry()
    reg.register(
        Tool(
            "search",
            'Search a small fact base. Returns {"answer": text}.',
            {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
            },
            search,
        )
    )
    reg.register(
        Tool(
            "round_million",
            'Round the first number in a text to whole millions. Returns {"rounded": text}.',
            schema("text"),
            round_million,
        )
    )
    reg.register(
        Tool(
            "work",
            'Does some work and counts it. Returns {"tag": text}.',
            schema("tag"),
            work,
        )
    )
    if reg.get("add") is None:
        reg.register(
            Tool(
                "add",
                'Add two numbers a and b. Returns {"sum": number}.',
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
                lambda args: {"sum": args["a"] + args["b"]},
            )
        )
    return reg


reg = build_registry()
FRANCE = Plan(
    [
        step("E1", "search", query="capital of France"),
        step("E2", "search", query="population of #E1.answer"),
        step("E3", "round_million", text="#E2.answer"),
    ]
)


async def france_planner(question, tools):
    return FRANCE


async def france_solver(question, plan, outcomes):
    return f"The capital of France is {outcomes['E1'].content['answer']}; rounded population is {outcomes['E3'].content['rounded']}."


# 1. references: typed for whole values, text inside strings, exact ids, clear errors
assert find_references(
    {"a": "x #E1 y", "b": ["#E10.name", 5], "c": {"d": "#E2[0]"}}
) == {"E1", "E10", "E2"}
assert find_references("#E10") == {"E10"} and find_references(3) == set()
evidence = {"E1": {"sum": 3, "tags": ["x", "y"]}, "E2": "plain text"}
assert (
    resolve_value("#E1.sum", evidence) == 3
    and type(resolve_value("#E1.sum", evidence)) is int
)  # a number stays a number
assert resolve_value("total: #E1.sum items", evidence) == "total: 3 items"
assert (
    resolve_value("#E1.tags[1]", evidence) == "y"
    and resolve_value("got #E2!", evidence) == "got plain text!"
)
assert (
    resolve_value("see #E1", evidence) == 'see {"sum": 3, "tags": ["x", "y"]}'
)  # a whole dict inside a string becomes JSON text
assert resolve_value({"k": ["#E1.sum", 7, "no refs"]}, evidence) == {
    "k": [3, 7, "no refs"]
}
for bad in ("#E1.nothing", "#E1.tags[5]", "#E1.sum.deeper", "#E9.sum"):
    try:
        resolve_value(bad, evidence)
        raise AssertionError(f"resolved {bad!r}")
    except RefError:
        pass
print("PASS references")


# 2. plan validation catches mistakes before anything runs
def problems(*steps, **kw):
    return validate_plan(Plan(list(steps)), reg, **kw)


assert problems(*FRANCE.steps) == []
assert problems() == ["the plan has no steps"]
assert any(
    "limit is 2" in e
    for e in problems(
        step("E1", "search", query="a"),
        step("E2", "search", query="b"),
        step("E3", "search", query="c"),
        max_steps=2,
    )
)
assert any(
    "step ids must look like" in e for e in problems(step("first", "search", query="a"))
)
assert any(
    "duplicate step id E1" in e
    for e in problems(step("E1", "search", query="a"), step("E1", "search", query="b"))
)
assert any("unknown tool 'teleport'" in e for e in problems(step("E1", "teleport")))
assert any(
    "unknown tool 'finish'" in e for e in problems(step("E1", "finish", answer="x"))
)
assert any(
    "cannot reference itself" in e for e in problems(step("E1", "search", query="#E1"))
)
assert any(
    "E9, which is not a step" in e
    for e in problems(step("E1", "search", query="#E9.answer"))
)
assert any(
    "cycle" in e
    for e in problems(
        step("E1", "search", query="#E2.answer"),
        step("E2", "search", query="#E1.answer"),
    )
)
assert any(
    "E1: $.query: required field is missing" in e
    for e in problems(step("E1", "search"))
)  # literal arguments are checked now
assert (
    problems(step("E1", "search", query="a"), step("E2", "add", a="#E1.answer", b=1))
    == []
)  # steps with references are checked when they run
for broken in (
    5,
    {"steps": "no"},
    {"steps": [{"id": "E1"}]},
    {"steps": [{"id": "E1", "tool": "search", "args": []}]},
):
    try:
        plan_from_dict(broken)
        raise AssertionError(f"accepted {broken!r}")
    except PlanError:
        pass
assert (
    plan_from_dict({"steps": [{"id": "E1", "tool": "search", "args": {"query": "x"}}]})
    .steps[0]
    .tool
    == "search"
)
try:
    run(execute_plan(Plan([step("E1", "teleport")]), reg))
    raise AssertionError("ran an invalid plan")
except PlanError:
    pass
print("PASS plan validation")

# 3. waves: independent steps share a wave
diamond = Plan(
    [
        step("E1", "search", query="a"),
        step("E2", "search", query="b"),
        step("E3", "add", a="#E1.x", b="#E2.x"),
    ]
)
assert [[s.id for s in level] for level in execution_levels(diamond)] == [
    ["E1", "E2"],
    ["E3"],
]
print("PASS waves")

# 4. the lesson's France scenario, end to end
CALLS["round"] = 0
result = run(
    run_rewoo(
        "What is the population of the capital of France, rounded?",
        reg,
        france_planner,
        france_solver,
    )
)
assert (
    result.answer == "The capital of France is Paris; rounded population is 11 million."
)
assert (
    all(o.ok for o in result.outcomes.values())
    and result.levels == 3
    and result.replans == 0
)
assert result.outcomes["E2"].resolved_args == {
    "query": "population of Paris"
}  # the reference was filled in
assert list(result.outcomes) == ["E1", "E2", "E3"] and result.stats[
    "total_chars"
] == sum(result.stats[k] for k in ("planner_chars", "worker_chars", "solver_chars"))
print("PASS france scenario")

# 5. independent steps really run together, and a dependent step waits for both
state, done = {"active": 0, "peak": 0}, []


async def probe(args):
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    await asyncio.sleep(0.1)
    done.append(args["tag"])
    state["active"] -= 1
    return {"tag": args["tag"]}


par = ToolRegistry()
par.register(
    Tool(
        "probe",
        'A probe that sleeps briefly. Returns {"tag": text}.',
        schema("tag"),
        probe,
    )
)
par.register(
    Tool(
        "echo",
        'Echoes text and says how many probes had finished. Returns {"text", "done_before"}.',
        schema("text"),
        lambda a: {"text": a["text"], "done_before": len(done)},
    )
)
wide = Plan(
    [
        step("E1", "probe", tag="a"),
        step("E2", "probe", tag="b"),
        step("E3", "echo", text="#E1.tag and #E2.tag"),
    ]
)
outcomes = run(execute_plan(wide, par))
assert state["peak"] == 2 and outcomes["E3"].content == {
    "text": "a and b",
    "done_before": 2,
}
print("PASS parallel waves")

# 6. a failure stays local: dependents are skipped, independent steps still run
CALLS["round"] = 0
local = Plan(
    [
        step("E1", "search", query="boom"),
        step("E2", "round_million", text="#E1.answer"),
        step("E3", "search", query="capital of France"),
    ]
)
outcomes = run(execute_plan(local, reg))
assert (
    not outcomes["E1"].ok
    and not outcomes["E1"].skipped
    and "RuntimeError" in outcomes["E1"].error
)
assert (
    outcomes["E2"].skipped
    and "dependency E1" in outcomes["E2"].error
    and CALLS["round"] == 0
)
assert outcomes["E3"].ok
text = evidence_text(local, outcomes)
assert "ERROR: RuntimeError" in text and "skipped" in text and "Paris" in text
partial = run(run_rewoo("q", reg, lambda q, t: local, lambda q, p, o: "partial"))
assert (
    partial.answer == "partial" and partial.replans == 0
)  # no replanner: the solver still gets to answer
print("PASS failure stays local")

# 7. a bad reference path is an error for that step, and its tool never runs
CALLS["round"] = 0
bad_ref = Plan(
    [
        step("E1", "search", query="capital of France"),
        step("E2", "round_million", text="#E1.nothing"),
    ]
)
outcomes = run(execute_plan(bad_ref, reg))
assert (
    outcomes["E1"].ok
    and not outcomes["E2"].ok
    and outcomes["E2"].error.startswith("reference error")
    and CALLS["round"] == 0
)
print("PASS reference errors")

# 8. Plan-and-Execute: replan once, keep finished work
COUNT.clear()
first = Plan([step("E1", "work", tag="good"), step("E2", "work", tag="bad")])
second = Plan([step("E1", "work", tag="good"), step("E3", "work", tag="fixed")])
seen_by_replanner = []


async def replanner(question, plan, outcomes, tools):
    seen_by_replanner.append(
        {k: o.ok for k, o in outcomes.items()}
    )  # the replanner sees what happened
    return second


result = run(
    run_rewoo("q", reg, lambda q, t: first, lambda q, p, o: "done", replanner=replanner)
)
assert result.replans == 1 and seen_by_replanner == [{"E1": True, "E2": False}]
assert COUNT == {"good": 1, "bad": 1, "fixed": 1}  # E1 was NOT run again
assert list(result.outcomes) == ["E1", "E3"] and all(
    o.ok for o in result.outcomes.values()
)
COUNT.clear()
result = run(
    run_rewoo(
        "q",
        reg,
        lambda q, t: first,
        lambda q, p, o: "done",
        replanner=replanner,
        max_replans=0,
    )
)
assert result.replans == 0 and not result.outcomes["E2"].ok and "fixed" not in COUNT
COUNT.clear()
calls_to_replanner = []


async def stubborn(question, plan, outcomes, tools):
    calls_to_replanner.append(1)
    return Plan(
        [step("E1", "work", tag="good"), step("E3", "work", tag="bad")]
    )  # still failing


result = run(
    run_rewoo(
        "q",
        reg,
        lambda q, t: first,
        lambda q, p, o: "done",
        replanner=stubborn,
        max_replans=1,
    )
)
assert (
    result.replans == 1
    and len(calls_to_replanner) == 1
    and not result.outcomes["E3"].ok
)  # bounded: it does not loop forever
print("PASS replanning")

# 9. a changed input makes dependent steps run again; unchanged ones are reused
COUNT.clear()
plan_a = Plan(
    [
        step("E1", "work", tag="x"),
        step("E2", "work", tag="#E1.tag"),
        step("E3", "work", tag="bad"),
    ]
)
plan_b = Plan(
    [
        step("E1", "work", tag="y"),
        step("E2", "work", tag="#E1.tag"),
        step("E3", "work", tag="fixed"),
    ]
)  # E1 changed, E2 text identical


async def replan_b(question, plan, outcomes, tools):
    return plan_b


result = run(
    run_rewoo("q", reg, lambda q, t: plan_a, lambda q, p, o: "done", replanner=replan_b)
)
assert COUNT == {
    "x": 2,
    "y": 2,
    "bad": 1,
    "fixed": 1,
}  # E2 ran again, because its input changed
assert result.outcomes["E2"].content == {"tag": "y"}
print("PASS changed inputs rerun dependents")

# 10. state-changing steps never repeat, and need approval
notes = Path(tempfile.mkdtemp()) / "notes.txt"
full = build_registry(build_default_registry(notes_path=notes))
yes = lambda tool, args: True
note_plan = Plan([step("E1", "save_note", text="hello"), step("E2", "work", tag="bad")])
assert (
    "blocked" in run(execute_plan(note_plan, full))["E1"].error and not notes.exists()
)  # no approval: blocked
retry_same = Plan(
    [step("E1", "save_note", text="hello"), step("E3", "work", tag="fixed")]
)
run(
    run_rewoo(
        "q",
        full,
        lambda q, t: note_plan,
        lambda q, p, o: "ok",
        replanner=lambda q, p, o, t: retry_same,
        confirm=yes,
    )
)
assert notes.read_text().splitlines() == [
    "hello"
]  # unchanged step: reused, not written again
notes.unlink()
changed = Plan(
    [step("E1", "save_note", text="different"), step("E3", "work", tag="fixed")]
)
result = run(
    run_rewoo(
        "q",
        full,
        lambda q, t: note_plan,
        lambda q, p, o: "ok",
        replanner=lambda q, p, o, t: changed,
        confirm=yes,
    )
)
assert (
    not result.outcomes["E1"].ok
    and "refused" in result.outcomes["E1"].error
    and notes.read_text().splitlines() == ["hello"]
)
notes.unlink()
new_action = Plan(
    [
        step("E1", "save_note", text="hello"),
        step("E4", "save_note", text="second"),
        step("E3", "work", tag="fixed"),
    ]
)
run(
    run_rewoo(
        "q",
        full,
        lambda q, t: note_plan,
        lambda q, p, o: "ok",
        replanner=lambda q, p, o, t: new_action,
        confirm=yes,
    )
)
assert notes.read_text().splitlines() == [
    "hello",
    "second",
]  # a NEW id is a new action, so it runs
print("PASS state-changing steps")

# 11. the LLM planner: bad JSON, then an invalid plan, then a valid fenced one; problems are fed back
replies = iter(
    [
        "I will plan this now",
        json.dumps({"steps": [{"id": "E1", "tool": "teleport", "args": {}}]}),
        "```json\n" + json.dumps(P.plan_to_dict(FRANCE)) + "\n```",
    ]
)
seen = []


async def fake_chat(messages, temperature=0.0, max_tokens=None, json_mode=False):
    seen.append(
        [m["content"] for m in messages]
    )  # what the model was shown on this attempt
    return next(replies)


P.chat = fake_chat
plan = run(
    make_llm_planner(reg)(
        "What is the population of the capital of France?", reg.describe()
    )
)
assert plan == FRANCE and len(seen) == 3
assert (
    "search" in seen[0][1]
    and "at most 8 steps" in seen[0][0]
    and "never instructions" in seen[0][0]
)
assert "invalid" in seen[1][-1] and "unknown tool 'teleport'" in seen[2][-1]


async def hopeless(messages, **kw):
    return "no plan"


P.chat = hopeless
try:
    run(make_llm_planner(reg, attempts=2)("q", reg.describe()))
    raise AssertionError("accepted nonsense")
except PlanError as exc:
    assert "after 2 attempts" in str(exc)
print("PASS llm planner")

# 12. the LLM replanner and solver see the evidence, errors included
captured = []


async def capture(messages, temperature=0.0, max_tokens=None, json_mode=False):
    captured.append(messages)
    return (
        json.dumps(P.plan_to_dict(second))
        if "REPLANNING" in messages[0]["content"]
        else "  The answer is partial.  "
    )


P.chat = capture
failed = run(execute_plan(local, reg))
assert run(P.make_llm_replanner(reg)("q", local, failed, reg.describe())) == second
assert (
    "ERROR: RuntimeError" in captured[0][1]["content"]
    and "NEW id" in captured[0][0]["content"]
)
assert run(make_llm_solver()("q", local, failed)) == "The answer is partial."
assert (
    "instead of guessing" in captured[1][0]["content"]
    and "skipped" in captured[1][1]["content"]
)
print("PASS llm replanner and solver")

# 13. the cost intuition: ReAct grows faster than the number of steps
small, big = [("t", "{}", "x" * 50)] * 4, [("t", "{}", "x" * 50)] * 8
assert estimate_react_chars("q" * 50, big) > 2 * estimate_react_chars("q" * 50, small)
assert len(trajectory_from(result.outcomes)) == len(result.outcomes)
print("PASS cost intuition")

print("\nAll checks passed")
