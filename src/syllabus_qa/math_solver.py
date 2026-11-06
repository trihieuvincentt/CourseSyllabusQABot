"""Deterministic math helper for chat input.

Math questions are handled before retrieval. A syllabus vector index is useful
for course content, but it is a poor calculator and can cause a model to
hallucinate when a user sends an expression such as 2x + 3 = 7. This module
supports a limited, auditable grammar instead of executing Python or asking an
LLM to perform the calculation.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable


class MathInputError(ValueError):
    """Raised when an input is not in the supported math grammar."""


@dataclass(frozen=True)
class MathResult:
    """A user-facing result and a machine-readable operation label."""

    answer: str
    operation: str


_ALLOWED_NAMES: dict[str, float] = {"pi": math.pi, "e": math.e}
_ALLOWED_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "abs": abs,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_VARIABLE_RE = re.compile(r"\b[a-zA-Z]\b")
_MATH_CHARS_RE = re.compile(r"^[0-9a-zA-Z_\s+\-*/%^().,=]+$")


def _normalise(expression: str) -> str:
    """Normalise calculator notation while retaining a strict character set."""

    value = expression.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    value = value.replace("^", "**")
    # Accept common school notation: 2x, 2(x + 1), and (x + 1)(x - 1).
    value = re.sub(r"(?<=[0-9.)])\s*(?=[a-zA-Z(])", "*", value)
    value = re.sub(r"(?<=[a-zA-Z)])\s*(?=\d)", "*", value)
    return value


def _parse(expression: str, *, variable: str | None = None) -> ast.Expression:
    if not expression or len(expression) > 512:
        raise MathInputError("Biểu thức rỗng hoặc quá dài.")
    if not _MATH_CHARS_RE.fullmatch(expression.replace("**", "^")):
        raise MathInputError("Biểu thức chứa ký tự chưa được hỗ trợ.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise MathInputError("Không đọc được biểu thức toán học.") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if variable is not None and node.id == variable:
                continue
            if node.id not in _ALLOWED_NAMES and node.id not in _ALLOWED_FUNCTIONS:
                raise MathInputError(f"Chỉ hỗ trợ biến {variable!r} trong phương trình.")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise MathInputError("Hàm toán học chưa được hỗ trợ.")
            if node.keywords or len(node.args) != 1:
                raise MathInputError("Hàm toán học chỉ nhận một đối số.")
        elif isinstance(
            node,
            (
                ast.Expression,
                ast.Constant,
                ast.BinOp,
                ast.UnaryOp,
                ast.Name,
                ast.Call,
                ast.Load,
                ast.operator,
                ast.unaryop,
            ),
        ):
            continue
        else:
            raise MathInputError("Cấu trúc biểu thức chưa được hỗ trợ.")
    return tree


def _evaluate_node(node: ast.AST, variables: dict[str, float] | None = None) -> float:
    variables = variables or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, variables))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, variables)
        right = _evaluate_node(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise MathInputError("Số mũ quá lớn.")
        return float(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return float(_ALLOWED_FUNCTIONS[node.func.id](_evaluate_node(node.args[0], variables)))
    raise MathInputError("Không thể tính biểu thức này.")


def _polynomial(node: ast.AST, variable: str) -> tuple[float, float, float]:
    """Return coefficients (a, b, c) for a*x² + b*x + c."""

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return 0.0, 0.0, float(node.value)
    if isinstance(node, ast.Name) and node.id == variable:
        return 0.0, 1.0, 0.0
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        a, b, c = _polynomial(node.operand, variable)
        sign = 1.0 if isinstance(node.op, ast.UAdd) else -1.0
        return sign * a, sign * b, sign * c
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _polynomial(node.left, variable)
        right = _polynomial(node.right, variable)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
            return left[0] + sign * right[0], left[1] + sign * right[1], left[2] + sign * right[2]
        if isinstance(node.op, ast.Mult):
            if left[0] and right[0]:
                raise MathInputError("Chỉ hỗ trợ phương trình bậc hai trở xuống.")
            a = left[0] * right[2] + left[1] * right[1] + left[2] * right[0]
            b = left[1] * right[2] + left[2] * right[1]
            c = left[2] * right[2]
            return a, b, c
        if isinstance(node.op, ast.Div):
            if right[0] or right[1] or right[2] == 0:
                raise MathInputError("Mẫu số phải là hằng số khác 0.")
            return left[0] / right[2], left[1] / right[2], left[2] / right[2]
        if isinstance(node.op, ast.Pow):
            if right[0] or right[1] or right[2] not in (0, 1, 2):
                raise MathInputError("Chỉ hỗ trợ số mũ 0, 1 hoặc 2.")
            exponent = int(right[2])
            if exponent == 0:
                return 0.0, 0.0, 1.0
            if exponent == 1:
                return left
            if left[0]:
                raise MathInputError("Phương trình vượt quá bậc hai.")
            return left[1] ** 2, 2 * left[1] * left[2], left[2] ** 2
    raise MathInputError("Phương trình có thành phần chưa được hỗ trợ.")


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"


def _solve_equation(equation: str) -> MathResult:
    left, right = (_normalise(part) for part in equation.split("=", 1))
    variables = sorted(set(_VARIABLE_RE.findall(equation)) - {"e"})
    if len(variables) != 1:
        raise MathInputError("Phương trình phải có đúng một biến (ví dụ x).")
    variable = variables[0]
    left_tree = _parse(left, variable=variable)
    right_tree = _parse(right, variable=variable)
    a1, b1, c1 = _polynomial(left_tree.body, variable)
    a2, b2, c2 = _polynomial(right_tree.body, variable)
    a, b, c = a1 - a2, b1 - b2, c1 - c2
    if math.isclose(a, 0, abs_tol=1e-12) and math.isclose(b, 0, abs_tol=1e-12):
        answer = "Phương trình có vô số nghiệm." if math.isclose(c, 0, abs_tol=1e-12) else "Phương trình vô nghiệm."
        return MathResult(answer, "equation")
    if math.isclose(a, 0, abs_tol=1e-12):
        root = -c / b
        return MathResult(f"{variable} = {_format_number(root)}", "equation")
    discriminant = b * b - 4 * a * c
    if discriminant < -1e-12:
        return MathResult(f"Phương trình không có nghiệm thực (Δ = {_format_number(discriminant)}).", "equation")
    if math.isclose(discriminant, 0, abs_tol=1e-12):
        root = -b / (2 * a)
        return MathResult(f"{variable} = {_format_number(root)} (nghiệm kép)", "equation")
    root_a = (-b + math.sqrt(discriminant)) / (2 * a)
    root_b = (-b - math.sqrt(discriminant)) / (2 * a)
    roots = sorted((root_a, root_b))
    return MathResult(
        f"{variable} = {_format_number(roots[0])} hoặc {variable} = {_format_number(roots[1])}",
        "equation",
    )


def looks_like_math(text: str) -> bool:
    """Return true only for a high-confidence expression/equation input."""

    value = text.strip()
    if not value or len(value) > 512 or not _MATH_CHARS_RE.fullmatch(value.replace("^", "**")):
        return False
    has_operator = bool(re.search(r"[+\-*/%^=]", value)) or bool(
        re.search(r"(?:sqrt|sin|cos|tan|log|exp|abs)\s*\(", value)
    )
    has_number = bool(re.search(r"\d", value))
    identifiers = re.findall(r"[a-zA-Z]+", value)
    known_identifiers = set(_ALLOWED_NAMES) | set(_ALLOWED_FUNCTIONS)
    only_math_identifiers = all(identifier in known_identifiers or len(identifier) == 1 for identifier in identifiers)
    return has_number and has_operator and only_math_identifiers


def solve_math(text: str) -> MathResult:
    """Evaluate an arithmetic expression or solve a linear/quadratic equation."""

    if not looks_like_math(text):
        raise MathInputError("Đây không phải là biểu thức toán học được hỗ trợ.")
    expression = _normalise(text)
    if expression.count("=") == 1:
        return _solve_equation(expression)
    tree = _parse(expression)
    try:
        value = _evaluate_node(tree.body)
    except (ArithmeticError, ValueError, OverflowError) as exc:
        raise MathInputError("Không thể tính biểu thức này.") from exc
    if not math.isfinite(value):
        raise MathInputError("Kết quả không hữu hạn.")
    return MathResult(f"Kết quả: {_format_number(value)}", "expression")
