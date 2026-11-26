"""Run the lesson's scripted ReAct scenario through our agent loop and print the trace."""
import asyncio

from app.agents.agent import AgentLoop, pretty_trace
from app.agents.demo_tools import build_demo_registry


def call(cid, name, **arguments):  # one tool call in the shape the loop expects
    return {"id": cid, "name": name, "arguments": arguments}


SCRIPT = [  # the lesson's script: store the base, compute tax, store it, compute the total, check, finish
    {"thought": "store the base price", "tool_calls": [call("c1", "kv_set", key="base", value="120")]},
    {"thought": "compute 15% tax", "tool_calls": [call("c2", "calculator", expr="120 * 0.15")]},
    {"thought": "store the tax", "tool_calls": [call("c3", "kv_set", key="tax", value="18.0")]},
    {"thought": "compute the total", "tool_calls": [call("c4", "calculator", expr="120 + 18.0")]},
    {"thought": "confirm the stored base", "tool_calls": [call("c5", "kv_get", key="base")]},
    {"thought": "I have everything", "tool_calls": [call("c6", "finish", answer="the total including 15% tax is 138.0")]},
]


async def scripted_model(history, tools):  # stands in for the LLM: replays the script, one entry per turn
    step = sum(1 for m in history if m["role"] == "assistant")  # how many turns the assistant has already taken
    return SCRIPT[step] if step < len(SCRIPT) else {"content": "no more actions"}


result = asyncio.run(AgentLoop(build_demo_registry(), scripted_model).run("What is 120 plus 15% tax, stored in kv?"))
pretty_trace(result)
print(f"tools used: {sorted({s.tool for s in result.steps})}")
