"""Stage 13 demo. `uv run demo_rewoo.py` runs the lesson's scripted scenario offline; `uv run demo_rewoo.py live` uses the real model."""

import asyncio  # everything is async
import re  # the round_million tool
import sys  # reads the command-line argument

from app.agents.planner import (
    Plan,
    PlanStep,
    make_llm_planner,
    make_llm_replanner,
    make_llm_solver,
    print_run,
    run_rewoo,
)
from app.agents.tools import Tool, ToolRegistry


def search(args):  # a tiny fact base standing in for a real search tool
    q = args["query"].lower()
    if "capital of france" in q:
        return {"answer": "Paris"}
    if "population of paris" in q:
        return {"answer": "11.2 million metro"}
    return {"answer": f"no result for {args['query']!r}"}


def round_million(args):  # "11.2 million metro" -> "11 million"
    return {
        "rounded": f"{round(float(re.search(r'[0-9]+[.]?[0-9]*', args['text']).group()))} million"
    }


def build_registry() -> (
    ToolRegistry
):  # descriptions say what each tool RETURNS, so the planner can write #E1.answer
    registry = ToolRegistry()
    registry.register(
        Tool(
            "search",
            'Look up a fact. Returns {"answer": text}. Use for factual lookups.',
            {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
            },
            search,
        )
    )
    registry.register(
        Tool(
            "round_million",
            'Round the first number found in a text to whole millions. Returns {"rounded": text}.',
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            round_million,
        )
    )
    return registry


QUESTION = "What is the population of the capital of France, rounded to millions?"
registry = build_registry()

if (
    len(sys.argv) > 1 and sys.argv[1] == "live"
):  # the real model writes the plan and the answer
    from app.foundation.llm_client import (
        start_usage_tracking,
    )  # counts LLM calls for this run

    tracker = start_usage_tracking()
    result = asyncio.run(
        run_rewoo(
            QUESTION,
            registry,
            make_llm_planner(registry),
            make_llm_solver(),
            replanner=make_llm_replanner(registry),
        )
    )
    print_run(result)
    print(
        f"LLM calls: {tracker['llm_calls']} (ReWOO needs 2 when nothing fails; a ReAct loop needs one per step plus one)"
    )
else:  # scripted: the plan and the solver are fixed, so it runs with no API key
    plan = Plan(
        [
            PlanStep("E1", "search", {"query": "capital of France"}),
            PlanStep("E2", "search", {"query": "population of #E1.answer"}),
            PlanStep("E3", "round_million", {"text": "#E2.answer"}),
        ]
    )

    async def planner(question, tools):
        return plan

    async def solver(question, plan, outcomes):
        return f"The capital of France is {outcomes['E1'].content['answer']}; its rounded population is {outcomes['E3'].content['rounded']}."

    print_run(asyncio.run(run_rewoo(QUESTION, registry, planner, solver)))
