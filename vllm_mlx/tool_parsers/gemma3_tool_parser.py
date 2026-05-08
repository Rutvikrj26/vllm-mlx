# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3 / MedGemma tool call parser for vllm-mlx.

Handles Gemma 3's training-time tool call format (Google's Gemma 2/3 format):

  ```tool_code
  print(FunctionName(arg1=value1, arg2="value2"))
  ```

The block may appear anywhere in the response; text before the fence is
preserved as ``content``. Multiple ``print(...)`` calls within a single block
are all extracted. Arguments are parsed with ``ast.parse`` — never ``eval`` or
``exec`` — and serialised to JSON.

Reference: MedGemma tool-call observations; Gemma 2/3 training-time format.
"""

import ast
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)

# Match the ```tool_code ... ``` fence (handles leading/trailing whitespace
# and any text before/after the fence on the same line).
_FENCE_RE = re.compile(
    r"```tool_code\s*\n(.*?)```",
    re.DOTALL,
)

# Max fence body length to guard against runaway input (1 MB)
_MAX_FENCE_LEN = 1_048_576


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _ast_node_to_python(node: ast.expr) -> Any:
    """Convert a supported AST expression node to a Python value.

    Supported types: str, int, float, bool, None, list, dict (recursively).
    Uses ``ast.literal_eval`` on Constant nodes and recurses through
    List/Dict/Tuple nodes.

    Args:
        node: An ``ast.expr`` node from a parsed call argument.

    Returns:
        The corresponding Python value.

    Raises:
        ValueError: If the node type is not supported.
    """
    if isinstance(node, ast.Constant):
        # Covers str, int, float, bool, None, bytes
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ast_node_to_python(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                raise ValueError("Dict unpacking (**) is not supported in tool args")
            result[_ast_node_to_python(k)] = _ast_node_to_python(v)
        return result
    # Fallback: attempt ast.literal_eval on the node (handles UnaryOp for -1 etc.)
    try:
        return ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Unsupported AST node type {type(node).__name__} in tool args"
        ) from exc


def _parse_print_call(stmt: ast.stmt) -> dict[str, Any] | None:
    """Extract a single tool call dict from a ``print(Func(...))`` statement.

    Expected shape:
        Expr(value=Call(func=Name(id='print'), args=[Call(func=Name/Attr, keywords=[...])]))

    The outer ``print`` wrapper is mandatory (Gemma 3 training format).

    Args:
        stmt: An AST statement node.

    Returns:
        ``{"id": ..., "name": ..., "arguments": "<json-string>"}`` or ``None``
        if the statement does not match the expected shape.
    """
    # Must be an expression statement
    if not isinstance(stmt, ast.Expr):
        return None
    outer = stmt.value
    # Outer call must be print(...)
    if not (
        isinstance(outer, ast.Call)
        and isinstance(outer.func, ast.Name)
        and outer.func.id == "print"
        and len(outer.args) == 1
        and not outer.keywords
    ):
        return None

    inner = outer.args[0]
    # Inner call: Name(...) or Attr(...) for dotted names like Pkg.Func(...)
    if not isinstance(inner, ast.Call):
        return None

    func = inner.func
    if isinstance(func, ast.Name):
        func_name = func.id
    elif isinstance(func, ast.Attribute):
        # e.g. Person.extract(...) — use the attribute (method) name
        func_name = func.attr
    else:
        return None

    # Only keyword arguments are supported (Gemma 3 format uses kwargs exclusively)
    if inner.args:
        logger.warning(
            "Gemma 3 tool parser: positional args in call to %r are ignored; "
            "only keyword arguments are extracted.",
            func_name,
        )

    kwargs: dict[str, Any] = {}
    for kw in inner.keywords:
        if kw.arg is None:
            # **kwargs unpacking — skip
            continue
        try:
            kwargs[kw.arg] = _ast_node_to_python(kw.value)
        except ValueError as exc:
            logger.warning(
                "Gemma 3 tool parser: could not convert kwarg %r for %r: %s",
                kw.arg,
                func_name,
                exc,
            )
            return None

    return {
        "id": generate_tool_id(),
        "name": func_name,
        "arguments": json.dumps(kwargs),
    }


@ToolParserManager.register_module("gemma3")
class Gemma3ToolParser(ToolParser):
    """
    Tool call parser for Gemma 3 / MedGemma models.

    Parses the training-time ``tool_code`` fence format:

        ```tool_code
        print(FunctionName(kwarg1=value1, kwarg2="value2"))
        ```

    One or more ``print(...)`` calls may appear in a single fence block.
    Arguments are extracted via ``ast.parse`` (never ``eval``/``exec``).

    Used when ``--enable-auto-tool-choice --tool-call-parser gemma3`` are set.
    """

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 3 / MedGemma model response.

        Args:
            model_output: The full text returned by the model.
            request: Optional request context (unused by this parser).

        Returns:
            ``ExtractedToolCallInformation`` with parsed tool calls, or
            ``tools_called=False`` when no ``tool_code`` fence is present or
            when the fence body cannot be parsed.
        """
        cleaned = self.strip_think_tags(model_output)

        m = _FENCE_RE.search(cleaned)
        if not m:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_before = cleaned[: m.start()].strip() or None
        fence_body = m.group(1)

        if len(fence_body) > _MAX_FENCE_LEN:
            logger.warning(
                "Gemma 3 tool parser: fence body exceeds %d bytes; skipping.",
                _MAX_FENCE_LEN,
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tree = ast.parse(fence_body, mode="exec")
        except SyntaxError as exc:
            logger.warning(
                "Gemma 3 tool parser: SyntaxError parsing tool_code block: %s", exc
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[dict[str, Any]] = []
        for stmt in tree.body:
            tc = _parse_print_call(stmt)
            if tc is not None:
                tool_calls.append(tc)

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_before,
            )
        else:
            # Fence found but no valid print(Func(...)) calls extracted
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract tool calls from streaming Gemma 3 model output.

        The ``tool_code`` fence is only parseable once it is fully closed, so
        we buffer silently until the closing ``` token arrives, then delegate
        to ``extract_tool_calls`` for the full parse.

        Args:
            previous_text: Text before this delta.
            current_text: Complete text accumulated so far.
            delta_text: New text in this chunk.
            previous_token_ids: Token IDs before this delta (unused).
            current_token_ids: All token IDs so far (unused).
            delta_token_ids: New token IDs in this chunk (unused).
            request: Optional request context (unused).

        Returns:
            Delta message dict with ``content`` or ``tool_calls``, or ``None``
            to signal buffering.
        """
        has_fence_start = "```tool_code" in current_text

        if not has_fence_start:
            # No fence started yet — pass content through normally
            return {"content": delta_text}

        # Check whether the closing fence has just arrived in this delta
        closing_arrived = "```" in delta_text and _FENCE_RE.search(current_text)

        if closing_arrived:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        # Inside fence but not yet closed — buffer (return None)
        return None


# ---------------------------------------------------------------------------
# Inline self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = Gemma3ToolParser()

    # --- Test 1: Single block, single call ---
    out1 = '```tool_code\nprint(GetWeather(city="Paris", units="metric"))\n```'
    r1 = parser.extract_tool_calls(out1)
    assert r1.tools_called, "Test 1 failed: tools_called should be True"
    assert len(r1.tool_calls) == 1, f"Test 1 failed: expected 1 call, got {len(r1.tool_calls)}"
    assert r1.tool_calls[0]["name"] == "GetWeather", f"Test 1 failed: name={r1.tool_calls[0]['name']}"
    args1 = json.loads(r1.tool_calls[0]["arguments"])
    assert args1 == {"city": "Paris", "units": "metric"}, f"Test 1 failed: args={args1}"
    print("Test 1 passed: single block, single call")

    # --- Test 2: Single block, multiple calls ---
    out2 = (
        "```tool_code\n"
        'print(Search(query="Claude"))\n'
        'print(Lookup(id=42))\n'
        "```"
    )
    r2 = parser.extract_tool_calls(out2)
    assert r2.tools_called, "Test 2 failed: tools_called should be True"
    assert len(r2.tool_calls) == 2, f"Test 2 failed: expected 2 calls, got {len(r2.tool_calls)}"
    assert r2.tool_calls[0]["name"] == "Search"
    assert r2.tool_calls[1]["name"] == "Lookup"
    assert json.loads(r2.tool_calls[1]["arguments"])["id"] == 42
    print("Test 2 passed: single block, multiple calls")

    # --- Test 3: Various arg types ---
    out3 = (
        "```tool_code\n"
        "print(TypeTest("
        'name="Alice", age=30, score=9.5, active=True, deleted=False, '
        "nothing=None, tags=[\"a\", \"b\"], meta={\"k\": 1}"
        "))\n"
        "```"
    )
    r3 = parser.extract_tool_calls(out3)
    assert r3.tools_called, "Test 3 failed: tools_called should be True"
    args3 = json.loads(r3.tool_calls[0]["arguments"])
    assert args3["name"] == "Alice", f"Test 3 str: {args3['name']}"
    assert args3["age"] == 30, f"Test 3 int: {args3['age']}"
    assert abs(args3["score"] - 9.5) < 1e-9, f"Test 3 float: {args3['score']}"
    assert args3["active"] is True, f"Test 3 True: {args3['active']}"
    assert args3["deleted"] is False, f"Test 3 False: {args3['deleted']}"
    assert args3["nothing"] is None, f"Test 3 None: {args3['nothing']}"
    assert args3["tags"] == ["a", "b"], f"Test 3 list: {args3['tags']}"
    assert args3["meta"] == {"k": 1}, f"Test 3 dict: {args3['meta']}"
    print("Test 3 passed: various arg types (str, int, float, bool, None, list, dict)")

    # --- Test 4: No tool_code fence (plain content) ---
    out4 = "The answer is 42."
    r4 = parser.extract_tool_calls(out4)
    assert not r4.tools_called, "Test 4 failed: tools_called should be False"
    assert r4.content == out4, f"Test 4 failed: content={r4.content!r}"
    print("Test 4 passed: no tool_code fence -> tools_called=False")

    # --- Test 5: Malformed Python in fence ---
    out5 = "```tool_code\nprint(BadSyntax(x=\n```"
    r5 = parser.extract_tool_calls(out5)
    assert not r5.tools_called, "Test 5 failed: tools_called should be False"
    print("Test 5 passed: malformed Python -> tools_called=False, no crash")

    # --- Test 6: MedGemma-style dotted call (Person.extract) ---
    out6 = '```tool_code\nprint(Person.extract(text="John Doe, 45 years old"))\n```'
    r6 = parser.extract_tool_calls(out6)
    assert r6.tools_called, "Test 6 failed: tools_called should be True"
    assert r6.tool_calls[0]["name"] == "extract", f"Test 6 name: {r6.tool_calls[0]['name']}"
    assert json.loads(r6.tool_calls[0]["arguments"])["text"] == "John Doe, 45 years old"
    print("Test 6 passed: dotted call (Person.extract) -> attribute name extracted")

    print("\nAll tests passed.")
