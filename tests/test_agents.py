"""Stage 11 check: runs offline. It ends with 'All checks passed' when everything behaves."""
import asyncio  # the loop is async
import tempfile  # a throwaway folder for the note tool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agents.agent import AgentLoop, pretty_trace  # the loop under test
from app.agents.tools import build_default_registry  # Stage 9's tools

def run(coro):  # run an async function from plain code
    return asyncio.run(coro)


def fresh():  # a new registry whose notes go to a temp file
    path = Path(tempfile.mkdtemp()) / "notes.txt"
    return build_default_registry(notes_path=path), path


def call(cid, name, **arguments):  # one tool call in the shape the loop expects
    return {"id": cid, "name": name, "arguments": arguments}


def turn(thought, *calls):  # one scripted model turn that calls tools
    return {"thought": thought, "tool_calls": list(calls)}


class Scripted:  # a fake model: replays a fixed script, like the lesson's ToyLLM
    def __init__(self, script):
        self.script, self.cursor = list(script), 0

    async def __call__(self, history, tools):
        if self.cursor >= len(self.script):
            return {"content": "script finished"}
        self.cursor += 1
        return self.script[self.cursor - 1]


# 1. the basic ReAct run: act, observe, finish
registry, _ = fresh()
result = run(AgentLoop(registry, Scripted([
    turn("Add the base price and the tax.", call("c1", "add", a=120, b=18)),
    turn("I have the total.", call("c2", "finish", answer="The total including tax is 138.0")),
])).run("What is 120 plus 18?"))
pretty_trace(result)
assert result.stop_reason == "finish" and "138" in result.answer and result.turns == 2
assert [s.tool for s in result.steps] == ["add", "finish"] and "138" in result.steps[0].observation
print("PASS react run\n")

# 2. plain text with no tool calls ends the run (default)
result = run(AgentLoop(fresh()[0], Scripted([{"content": "hello"}])).run("hi"))
assert result.stop_reason == "answer" and result.answer == "hello" and result.turns == 1
print("PASS stop on plain answer")

# 3. require_finish: plain text is rejected with a reminder (exercise 2)
result = run(AgentLoop(fresh()[0], Scripted([
    {"content": "I think it is 138"},
    turn("", call("f", "finish", answer="138")),
]), require_finish=True).run("total?"))
assert result.stop_reason == "finish" and result.turns == 2
assert any("finish tool" in m.get("content", "") for m in result.history if m["role"] == "user")
print("PASS require_finish")

# 4. finish with bad arguments is an error, and the loop continues
result = run(AgentLoop(fresh()[0], Scripted([
    turn("", call("f1", "finish")),  # missing "answer"
    turn("", call("f2", "finish", answer="done")),
])).run("x"))
assert not result.steps[0].ok and "required" in result.steps[0].observation and result.turns == 2
print("PASS finish validation")

# 5. a malformed call becomes an observation, and the model recovers (exercise 3)
result = run(AgentLoop(fresh()[0], Scripted([
    turn("", call("c1", "add", a=120)),  # missing b
    turn("", call("c2", "add", a=120, b=18)),
    turn("", call("c3", "finish", answer="138")),
])).run("sum"))
assert [s.ok for s in result.steps] == [False, True, True] and "required" in result.steps[0].observation
print("PASS error recovery")

# 6. consequential tools stay gated inside the loop
registry, note_path = fresh()
script = [turn("", call("n1", "save_note", text="total is 138")), turn("", call("f", "finish", answer="done"))]
result = run(AgentLoop(registry, Scripted(script)).run("save it"))
assert not result.steps[0].ok and "blocked" in result.steps[0].observation and not note_path.exists()
registry, note_path = fresh()
result = run(AgentLoop(registry, Scripted(script), confirm=lambda tool, args: True).run("save it"))
assert result.steps[0].ok and note_path.read_text().strip() == "total is 138"
print("PASS consequential gate")


# 7. turn budget stops a loop that never ends
async def endless(history, tools):
    return turn("again", call("x", "add", a=1, b=1))


result = run(AgentLoop(fresh()[0], endless, max_turns=3).run("go"))
assert result.stop_reason == "max_turns" and result.answer is None and len(result.steps) == 3
print("PASS turn budget")

# 8. calls per turn are capped, but every call id still gets a result (exercise 1)
result = run(AgentLoop(fresh()[0], Scripted([
    turn("", call("a", "add", a=1, b=1), call("b", "add", a=2, b=2), call("c", "add", a=3, b=3)),
    turn("", call("f", "finish", answer="done")),
]), max_calls_per_turn=2).run("go"))
assert [s.ok for s in result.steps] == [True, True, False, True] and "skipped" in result.steps[2].observation
assert [m["tool_call_id"] for m in result.history if m["role"] == "tool"][:3] == ["a", "b", "c"]
print("PASS calls per turn cap")

# 9. calls after a valid finish are skipped
result = run(AgentLoop(fresh()[0], Scripted([
    turn("", call("f", "finish", answer="done"), call("a", "add", a=1, b=1)),
])).run("go"))
assert result.stop_reason == "finish" and not result.steps[1].ok and "already finished" in result.steps[1].observation
print("PASS skip after finish")

# 10. token budget stops the run before the model is even called
decider = Scripted([{"content": "never reached"}])
result = run(AgentLoop(fresh()[0], decider, max_tokens=5).run("word " * 50))
assert result.stop_reason == "token_budget" and result.turns == 0 and decider.cursor == 0
print("PASS token budget")

print("\nAll checks passed")
