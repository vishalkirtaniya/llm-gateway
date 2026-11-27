"""Demo tools for the Stage 11 agent loop: a safe calculator and a tiny key-value store."""

import ast  # parses expressions without running them
import operator  # the arithmetic functions we allow
from typing import Dict

from .tools import Tool, ToolRegistry  # Stage 9's tool interface

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}  # no Pow on purpose
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}  # +x and -x


def _eval_node(
    node: ast.AST,
) -> float:  # evaluate a parsed expression, allowing only plain arithmetic
    if isinstance(node, ast.Expression):  # the top-level wrapper
        return _eval_node(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value  # a plain number
    if (
        isinstance(node, ast.BinOp) and type(node.op) in _BINARY
    ):  # a + b, a - b, a * b, a / b
        return _BINARY[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:  # +a or -a
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError(
        "unsupported expression: only numbers, + - * / and parentheses are allowed"
    )


def calculator(args: dict) -> dict:  # the executor
    expr = args["expr"]
    if len(expr) > 100:  # keep inputs small
        raise ValueError("expression is too long (100 characters max)")
    return {
        "result": _eval_node(ast.parse(expr, mode="eval"))
    }  # SyntaxError and ZeroDivisionError become error results


class KVStore:
    def __init__(self):
        self._store: Dict[str, str] = {}

    def set(self, args: dict) -> dict:
        self._store[args["key"]] = args["value"]
        return {"stored": args["key"]}

    def get(self, args: dict) -> dict:
        return {
            "found": args["key"] in self._store,
            "value": self._store.get(args["key"]),
        }


def build_demo_registry() -> ToolRegistry:
    kv = KVStore()
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Use when you need exact arithmetic on numbers. Supports + - + / and parenthese only.",
            input_schema={
                "type": "object",
                "properties": {"expr": {"type": "string", "minLength": 1}},
                "required": ["expr"],
            },
            executor=calculator,
        )
    )
    registry.register(
        Tool(
            name="kv_set",
            description="Use to remember a value under a key for later steps in this run.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "minLength": 1},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
            executor=kv.set,
        )
    )
    registry.register(
        Tool(
            name="kv_get",
            description="Use to read back a value that was saved earlier with kv_set.",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string", "minLength": 1}},
                "required": ["key"],
            },
            executor=kv.get,
        )
    )
    return registry


def build_lesson_registry() -> ToolRegistry:
    registry = ToolRegistry()
    int_pair = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }
    registry.register(
        Tool(
            "add",
            "Add two integers a and b. Use for any integer addition. ",
            int_pair,
            lambda args: {"sum": args["a"] + args["b"]},
        )
    )
    registry.register(
        Tool(
            "multiply",
            "Multiple two integers a and b. Prefer multiplication over looped addition.",
            int_pair,
            lambda args: {"product": args["a"] * args["b"]},
        )
    )
    registry.register(
        Tool(
            "classify",
            "Classify a status as one of the allowed labels.",
            {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["open", "closed", "pending"]}
                },
                "required": ["status"],
            },
            lambda args: {"classified_as": args["status"]},
        )
    )
    return registry


def register_document_tools(
    registry: ToolRegistry, root: Path, max_chars: int = 2000
) -> None:
    base = root.resolve()  # the only folder these tools may ever read from

    def _list(args: dict) -> dict:  # names of the text files in the folder
        return {"documents": sorted(p.name for p in base.glob("*.txt"))}

    def _read(args: dict) -> dict:  # read one text file, safely
        target = (
            base / args["name"]
        ).resolve()  # resolve() collapses ".." and follows symlinks
        if (
            not target.is_relative_to(base)
            or target.suffix != ".txt"
            or not target.is_file()
        ):
            raise ValueError(
                "document not found"
            )  # ONE message for every failure: do not reveal which paths exist
        text = target.read_text(encoding="utf-8")
        return {
            "name": target.name,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    registry.register(
        Tool(
            "list_documents",
            "Use to see which knowledge documents exist. Do not use to search inside them.",
            {"type": "object", "properties": {}, "required": []},
            _list,
        )
    )
    registry.register(
        Tool(
            "read_document",
            "Use to read the full text of one knowledge document by its file name from list_documents.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 100}
                },
                "required": ["name"],
            },
            _read,
        )
    )
