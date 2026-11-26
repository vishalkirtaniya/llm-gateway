import ast 
import operator
from typing import Dict

from .tools import Tool, ToolRegistry

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression: only numbers, + - + / and parenthese are allowed")

def calculator(args: dict) -> dict:
    expr = args["expr"]
    if len(expr) > 100:
        raise ValueError("expression is too long (100 characters max)")
    return {"result": _eval_node(ast.parse(expr, mode="eval"))}
"""Demo tools for the Stage 11 agent loop: a safe calculator and a tiny key-value store."""
import ast  # parses expressions without running them
import operator  # the arithmetic functions we allow
from typing import Dict

from .tools import Tool, ToolRegistry  # Stage 9's tool interface

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}  # no Pow on purpose
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}  # +x and -x


def _eval_node(node: ast.AST) -> float:  # evaluate a parsed expression, allowing only plain arithmetic
    if isinstance(node, ast.Expression):  # the top-level wrapper
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value  # a plain number
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:  # a + b, a - b, a * b, a / b
        return _BINARY[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:  # +a or -a
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression: only numbers, + - * / and parentheses are allowed")


def calculator(args: dict) -> dict:  # the executor
    expr = args["expr"]
    if len(expr) > 100:  # keep inputs small
        raise ValueError("expression is too long (100 characters max)")
    return {"result": _eval_node(ast.parse(expr, mode="eval"))}  # SyntaxError and ZeroDivisionError become error results

class KVStore:
    def __init__(self):
        self._store: Dict[str, str] = {}

    def set(self, args: dict) -> dict:
        self._store[args["key"]] = args["value"]
        return {"stored": args["key"]}

    def get(self, args: dict) -> dict:
        return {"found": args["key"] in self._store, "value": self._store.get(args["key"])}

def build_demo_registry() -> ToolRegistry:
    kv = KVStore()
    registry = ToolRegistry()
    registry.register(Tool(
        name="calculator",
        description="Use when you need exact arithmetic on numbers. Supports + - + / and parenthese only.",
        input_schema={"type": "object", "properties": {"expr": {"type": "string", "minLength": 1}}, "required": ["expr"]},
        executor=calculator,
    ))
    registry.register(Tool(
        name="kv_set",
        description="Use to remember a value under a key for later steps in this run.",
        input_schema={"type": "object", "properties": {"key": {"type": "string", "minLength": 1}, "value": {"type": "string"}},
                      "required": ["key", "value"]},
        executor=kv.set,
    ))
    registry.register(Tool(
        name="kv_get",
        description="Use to read back a value that was saved earlier with kv_set.",
        input_schema={"type": "object", "properties": {"key": {"type": "string", "minLength": 1}}, "required": ["key"]},
        executor=kv.get,
    ))
    return registry
