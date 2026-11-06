import pytest

from syllabus_qa.math_solver import MathInputError, looks_like_math, solve_math


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3 * 4", "Kết quả: 14"), ("sqrt(16)", "Kết quả: 4"), ("2^3", "Kết quả: 8")],
)
def test_solves_arithmetic_expression(expression: str, expected: str) -> None:
    assert solve_math(expression).answer == expected


def test_solves_linear_equation_with_implicit_multiplication() -> None:
    assert solve_math("2x + 3 = 7").answer == "x = 2"


def test_solves_quadratic_equation() -> None:
    assert solve_math("x^2 - 5x + 6 = 0").answer == "x = 2 hoặc x = 3"


def test_does_not_treat_normal_question_as_math() -> None:
    assert not looks_like_math("Nội dung môn học là gì?")


def test_rejects_python_syntax() -> None:
    with pytest.raises(MathInputError):
        solve_math("__import__(1)")
