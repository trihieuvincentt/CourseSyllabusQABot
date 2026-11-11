"""Formula-aware handling for math content in chunks and user questions.

The pipeline in this repository treats every chunk as ordinary prose. That
breaks math in three places:

* BM25 tokenisation drops operators such as ``=``, ``*`` and ``/``;
* embeddings blur the structure of an expression;
* retrieved chunks carry no metadata saying "this chunk defines a formula".

This module keeps the *features of the calculation* at both ends of the
pipeline:

* build time - scan chunks, extract formulas, canonicalise them while keeping
  the verbatim ``raw`` text, attach metadata tags, and serialise a JSON index;
* query time - detect formulas in a question, build a search string that
  preserves operators (spelled out so BM25/embeddings can use them), match
  formulas structurally against the index, and safely evaluate purely numeric
  expressions without ``eval``.

Nothing here mutates the source data: the JSON index is a derived artefact and
every formula still points back to its ``source_chunk_id`` and page range.
"""
from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Encoding repair (the corpus was produced by several extractors)
# ---------------------------------------------------------------------------

_MOJIBAKE_RUN = re.compile(
    "[\u0080-\u00ff\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6"
    "\u02dc\u2013\u2014\u2018\u2019\u201a\u201c\u201d\u201e\u2020\u2021"
    "\u2022\u2026\u2030\u2039\u203a\u20ac\u2122]{2,}"
)


def restore_mojibake(text: str) -> str:
    """Repair UTF-8 text that was mistakenly decoded as Windows-1252.

    Only runs of characters that can be re-encoded as CP1252 bytes are
    touched, and only when the bytes form valid UTF-8 again. Legitimate
    typographic quotes are therefore left alone.
    """

    def repair(match: re.Match[str]) -> str:
        chunk = match.group(0)
        try:
            candidate = chunk.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return chunk
        return candidate

    return _MOJIBAKE_RUN.sub(repair, text)


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ", "0123456789n")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₙ", "0123456789n")

_UNICODE_MATH = {
    "×": "*",
    "⋅": "*",
    "·": "*",
    "∙": "*",
    "∗": "*",
    "✕": "*",
    "÷": "/",
    "⁄": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "≅": "==",
    "≡": "==",
    "±": "+-",
    "∓": "-+",
    "√": "sqrt",
    "∑": "sum",
    "Σ": "sum",
    "∏": "prod",
    "∫": "integral",
    "∞": "inf",
    "∈": "in",
    "∉": "not in",
    "⊂": "subset",
    "⊆": "subset",
    "∪": "union",
    "∩": "intersect",
    "∅": "emptyset",
    "→": "->",
    "←": "<-",
    "⇒": "=>",
    "⇐": "<=",
    "∧": "and",
    "∨": "or",
    "¬": "not",
    "∀": "forall",
    "∃": "exists",
    "∂": "partial",
    "∇": "nabla",
    "∘": "compose",
}

_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "ς": "sigma", "σ": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta", "Ε": "Epsilon",
    "Ζ": "Zeta", "Η": "Eta", "Θ": "Theta", "Ι": "Iota", "Κ": "Kappa",
    "Λ": "Lambda", "Μ": "Mu", "Ν": "Nu", "Ξ": "Xi", "Ο": "Omicron",
    "Π": "Pi", "Ρ": "Rho", "Σ": "Sigma", "Τ": "Tau", "Υ": "Upsilon",
    "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega",
}

_LATEX_TEXT_COMMANDS = ("text", "mathrm", "mathbf", "mathit", "mathcal", "mathbb", "operatorname")
_LATEX_SIMPLE = {
    "cdot": "*", "times": "*", "ast": "*", "div": "/", "pm": "+-",
    "mp": "-+", "leq": "<=", "le": "<=", "geq": ">=", "ge": ">=",
    "neq": "!=", "ne": "!=", "approx": "~=", "equiv": "==",
    "rightarrow": "->", "to": "->", "leftarrow": "<-", "infty": "inf",
    "partial": "partial", "nabla": "nabla", "sum": "sum", "prod": "prod",
    "int": "integral", "log": "log", "ln": "ln", "exp": "exp",
    "max": "max", "min": "min", "argmax": "argmax", "argmin": "argmin",
    "sin": "sin", "cos": "cos", "tan": "tan", "tanh": "tanh",
    "sqrt": "sqrt", "quad": " ", "qquad": " ", "%": "%",
}
_LATEX_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
    "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho", "sigma": "sigma",
    "tau": "tau", "upsilon": "upsilon", "phi": "phi", "chi": "chi",
    "psi": "psi", "omega": "omega",
}

_OPERATOR_NAMES = {
    "=": "equals",
    "==": "equals",
    "!=": "not-equal",
    "<": "less-than",
    ">": "greater-than",
    "<=": "less-equal",
    ">=": "greater-equal",
    "~=": "approx-equal",
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "^": "power",
    "%": "percent",
    "±": "plus-minus",
    "√": "sqrt",
    "sum": "sum",
    "prod": "product",
    "integral": "integral",
    "->": "maps-to",
    "∈": "in",
}

_SPELLED_OPERATORS = {
    "=": "equals",
    "+": "plus",
    "-": "minus",
    "*": "multiply",
    "/": "divide",
    "^": "power",
    "<": "less than",
    ">": "greater than",
    "<=": "less than or equal",
    ">=": "greater than or equal",
    "!=": "not equal",
    "~=": "approximately equal",
    "±": "plus or minus",
    "√": "square root",
}


def _expand_scripts(text: str) -> str:
    def superscript(match: re.Match[str]) -> str:
        return "^{" + match.group(0).translate(_SUPERSCRIPTS) + "}"

    def subscript(match: re.Match[str]) -> str:
        return "_{" + match.group(0).translate(_SUBSCRIPTS) + "}"

    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+", superscript, text)
    return re.sub(r"[₀₁₂₃₄₅₆₇₈₉ₙ]+", subscript, text)


def _latex_to_canonical(body: str) -> str:
    """Best-effort LaTeX -> linear notation, preserving the operation order."""
    text = body
    text = re.sub(r"\\(?:left|right|[bB]igg?)\b", "", text)
    text = re.sub(r"\\(?:quad|qquad|,|;|!| )", " ", text)
    for command in _LATEX_TEXT_COMMANDS:
        text = re.sub(rf"\\{command}\s*\{{([^{{}}]*)\}}", r"\1", text)
    # \frac{a}{b} -> (a)/(b); repeat to handle one level of nesting.
    for _ in range(2):
        text = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"root(\2,\1)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    # Sub/superscript braces: x_{i} -> x_i, x^{i+1} -> x^(i+1)
    text = re.sub(
        r"[_^]\{([^{}]*)\}",
        lambda m: ("_" if m.group(0)[0] == "_" else "^")
        + (m.group(1) if re.fullmatch(r"[\w.]+", m.group(1)) else f"({m.group(1)})"),
        text,
    )
    # Known commands, including \log_{2} -> log_2.
    text = re.sub(
        r"\\([A-Za-z]+)",
        lambda m: _LATEX_SIMPLE.get(m.group(1).lower(), _LATEX_GREEK.get(m.group(1).lower(), m.group(1))),
        text,
    )
    text = text.replace("\\{", "{").replace("\\}", "}").replace("\\\\", " ")
    return text


def _map_symbols(text: str) -> str:
    for symbol, replacement in _UNICODE_MATH.items():
        text = text.replace(symbol, f" {replacement} " if replacement.isalpha() else replacement)
    for symbol, replacement in _GREEK.items():
        text = text.replace(symbol, f" {replacement} ")
    return text


def canonicalize_math(text: str) -> str:
    """Normalise a formula without losing its structure.

    The result is a plain-text expression such as ``F1 = (2 * Precision *
    Recall) / (Precision + Recall)``. Unit superscripts and subscripts are kept
    as ``^(...)`` / ``_(...)`` so exponents are not silently flattened.
    Compound terms such as ``TF-IDF`` or ``t-SNE`` keep their hyphen instead of
    turning it into a subtraction.
    """
    value = _expand_scripts(text)
    value = unicodedata.normalize("NFKC", value)
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        placeholder = f"{len(protected)}"
        protected[placeholder] = match.group(0)
        return placeholder

    # Compound terms (TF-IDF, t-SNE, LSTM-CRF) and increments (i++) are words,
    # not chains of subtraction/addition operators.
    value = re.sub(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+", protect, value)
    value = re.sub(r"[A-Za-z][A-Za-z0-9]*\+\+", protect, value)
    value = _map_symbols(value)
    value = re.sub(r"\^\{([^{}]*)\}", r"^\1", value)
    value = re.sub(r"_\{([^{}]*)\}", r"_\1", value)
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[“”„‟]", '"', value)
    value = re.sub(r"[‘’‚‛]", "'", value)
    value = re.sub(r"\s+", " ", value).strip()
    multi_ops: dict[str, str] = {}

    def protect_operator(match: re.Match[str]) -> str:
        placeholder = f"{len(multi_ops)}"
        multi_ops[placeholder] = match.group(1)
        return f" {placeholder} "

    value = re.sub(r"\s*(<=|>=|!=|~=|==|->|<-)\s*", protect_operator, value)
    # Exponents stay attached ("x^2"); every other operator is spaced out.
    value = re.sub(r"\s*([=<>+\-*/])\s*", r" \1 ", value)
    value = re.sub(r"\s+([)\]}])", r"\1", value)
    value = re.sub(r"([(\[{])\s+", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value)
    for placeholder, operator in multi_ops.items():
        value = value.replace(placeholder, operator)
    for placeholder, original in protected.items():
        value = value.replace(placeholder, original)
    return value.strip(" .,;:")


def spell_operators(canonical: str) -> str:
    """Add operator words so BM25 and embeddings can see the operation."""
    pattern = re.compile(r"<=|>=|!=|~=|==|[=+\-*/^<>]|±|√")

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return f" {_SPELLED_OPERATORS.get(token, token)} "

    return re.sub(r"\s+", " ", pattern.sub(replace, canonical)).strip()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_FUNCTION_NAMES = (
    "log2", "log10", "log", "ln", "exp", "sqrt", "argmax", "argmin",
    "softmax", "sigmoid", "tanh", "relu", "cos", "sin", "tan", "abs",
    "min", "max", "round",
)

_KNOWN_TARGETS = (
    "f1 score", "f1", "precision", "recall", "accuracy", "auc", "mrr",
    "map", "rmse", "mape", "bleu", "meteor", "rouge", "perplexity",
    "cosine similarity", "cosine", "tf-idf", "tfidf", "idf", "tf",
    "softmax", "sigmoid", "cross-entropy", "jaccard", "euclidean distance",
    "similarity",
)

_TOKEN_RE = re.compile(
    r"""
      (?P<func>\b(?:log2|log10|log|ln|exp|sqrt|argmax|argmin|softmax|sigmoid|
                    tanh|relu|cos|sin|tan|abs|min|max|round)\b(?=\s*\())
    | (?P<frac>[¼-¾⅐-⅞])
    | (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?)
    | (?P<ident>[A-Za-z][A-Za-z0-9_']*(?:_[A-Za-z0-9]+)?)
    | (?P<greek>[\u0370-\u03ff\u1f00-\u1fff]+)
    | (?P<op>==|!=|<=|>=|=>|->|<-|[=<>+\-*/^%~×÷·⋅∗−–—≤≥≠≈≡±∑Σ∏∫√∈∉→←⇒∧∨¬∂∇])
    | (?P<paren>[()\[\]{}])
    | (?P<bar>[\|])
    | (?P<comma>,)
    """,
    re.VERBOSE,
)

_MATH_IDENTS = {
    "precision", "recall", "accuracy", "similarity", "score", "weight",
    "weights", "loss", "gradient", "probability", "logit", "logits",
    "entropy", "tf", "idf", "tfidf", "theta", "alpha", "beta", "lambda",
    "mu", "sigma", "vector", "matrix", "document", "query", "label",
    # Word-analogy operands (King - Man + Woman ~ Queen).
    "king", "queen", "man", "woman",
}
_STRONG_OPS = {"=", "*", "/", "^", "±", "√", "sum", "prod", "integral", "<=", ">=", "!=", "~=", "==", "->"}
_ARITHMETIC_OPS = {"+", "-", "*", "/", "^"}
_VIETNAMESE_CONTINUATION = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_SYMBOL_TO_ASCII = {
    "×": "*", "⋅": "*", "·": "*", "∙": "*", "∗": "*", "✕": "*",
    "÷": "/", "⁄": "/", "−": "-", "–": "-", "—": "-",
    "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~=", "≅": "==", "≡": "==",
    "±": "±", "∑": "sum", "Σ": "sum", "∏": "prod", "∫": "integral",
    "√": "√", "→": "->", "←": "<-", "⇒": "=>", "∧": "and", "∨": "or",
    "¬": "not", "∂": "partial", "∇": "nabla", "∈": "in", "∉": "not in",
}
_DATE_LIKE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$"
)


def _is_math_ident(value: str) -> bool:
    lowered = value.casefold()
    if len(value) == 1:
        return True
    if re.fullmatch(r"[A-Za-z]\d+", value):
        return True
    if re.fullmatch(r"[A-Za-z]{1,3}\d+", value):
        return True
    if value.isupper() and 2 <= len(value) <= 5:
        return True
    # PDF extractions of subscripts (AiBi) keep their capital letters.
    if len(value) <= 6 and sum(character.isupper() for character in value) >= 2:
        return True
    return lowered in _MATH_IDENTS or lowered in _FUNCTION_NAMES


def _qualifies(values: list[str], kinds: list[str]) -> bool:
    ops = [
        _SYMBOL_TO_ASCII.get(value, value)
        for value, kind in zip(values, kinds)
        if kind == "op"
    ]
    if not ops:
        return False
    joined = "".join(values)
    if _DATE_LIKE.fullmatch(joined.replace(" ", "")):
        return False
    numbers = sum(kind in {"num", "frac"} for kind in kinds)
    idents = sum(kind in {"ident", "greek", "func"} for kind in kinds)
    has_call = any(
        kind == "func" and index + 1 < len(kinds) and values[index + 1] == "("
        for index, kind in enumerate(kinds)
    )
    # ISBNs, phone numbers and page ranges only chain "-" tokens; every real
    # formula needs at least one strong operator.
    if not any(op in _STRONG_OPS for op in ops):
        return False
    # Two or more "=" in one span (similarity = cos θ = A · B) is a formula.
    if ops.count("=") >= 2 and idents >= 2:
        return True
    if "=" in ops and numbers + idents >= 2 and any(op in _ARITHMETIC_OPS for op in ops):
        return True
    if has_call:
        return True
    numerals = [
        value for value, kind in zip(values, kinds) if kind in {"num", "frac"}
    ]
    if set(ops) <= {"/", "-"} and numbers == 0 and "(" not in values:
        return False  # "NLP/ML", "BoW/TF-IDF" - abbreviations, not formulas.
    if numerals and all(
        re.fullmatch(r"(?:19|20)\d{2}", value) for value in numerals
    ) and not any(op in {"=", "*", "^", "√", "sum", "prod", "integral", "->"} for op in ops):
        return False  # "(NLPBA/BioNLP) (2004)" - a name with a year.
    if any(
        op in {"*", "/", "^", "±", "√", "sum", "prod", "integral", "->"} for op in ops
    ) and (numbers >= 1 or idents >= 2):
        return True
    if any(op in {"<=", ">=", "!=", "~=", "==", "<", ">"} for op in ops) and (
        numbers >= 2 or ("~=" in ops and idents >= 3)
    ):
        return True
    return bool(
        sum(op in _ARITHMETIC_OPS for op in ops) >= 2
        and any(op in {"*", "/", "^"} for op in ops)
        and numbers + idents >= 3
    )


def _tokenize(text: str) -> list[tuple[int, int, str, str]]:
    return [
        (match.start(), match.end(), match.lastgroup or "", match.group(0))
        for match in _TOKEN_RE.finditer(text)
    ]


def _extract_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, raw) formula spans found in plain text."""
    spans: list[tuple[int, int, str]] = []
    tokens = _tokenize(text)
    run: list[tuple[int, int, str, str]] = []
    previous_end = -1

    def flush() -> None:
        nonlocal run
        if not run:
            return
        values = [value for _start, _end, _kind, value in run]
        kinds = [kind for _start, _end, kind, _value in run]
        # Trim leading/trailing operators so spans start and end on operands.
        while run and kinds[0] not in {"num", "frac", "ident", "func", "greek", "paren", "bar"}:
            run.pop(0)
            values.pop(0)
            kinds.pop(0)
        while run and kinds[-1] not in {"num", "frac", "ident", "func", "greek", "paren", "bar"}:
            run.pop()
            values.pop()
            kinds.pop()
        if not run:
            run = []
            return
        start, end = run[0][0], run[-1][1]
        # A single Latin letter that continues into a Vietnamese word with
        # diacritics ("bằng", "gì") is prose, not a variable.
        while (
            run
            and kinds[-1] == "ident"
            and len(values[-1]) == 1
            and end < len(text)
            and text[end] in _VIETNAMESE_CONTINUATION
        ):
            run.pop()
            values.pop()
            kinds.pop()
            end = run[-1][1] if run else start
        if not run:
            run = []
            return
        raw = text[start:end].strip()
        if len(raw) > 400:
            raw = raw[:400].rstrip()
            end = start + len(raw)
        if raw and _qualifies(values, kinds):
            spans.append((start, end, raw))
        run = []

    for start, end, kind, value in tokens:
        mathy = kind in {"num", "frac", "op", "paren", "bar", "comma", "greek", "func"} or (
            kind == "ident" and _is_math_ident(value)
        )
        if mathy:
            gap = text[previous_end:start] if previous_end >= 0 else ""
            if run and gap.strip():
                flush()
            run.append((start, end, kind, value))
        elif run:
            flush()
        previous_end = end
    flush()
    return spans


_LATEX_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)


def detect_formulas(text: str) -> list[dict[str, Any]]:
    """Detect LaTeX and plain-text formulas, preserving their raw text."""
    if not text or not text.strip():
        return []
    restored = restore_mojibake(text)
    found: list[dict[str, Any]] = []
    claimed: list[tuple[int, int]] = []

    for match in _LATEX_SPAN.finditer(restored):
        body = next(group for group in match.groups() if group is not None)
        display = match.group(0).startswith("$$") or match.group(0).startswith("\\[")
        # Currency pairs ("$10,000 ... $") are not LaTeX spans, and a real
        # inline formula is short and contains at least one math feature.
        body_text = body.strip()
        similar = any(
            match.start() < end and match.end() > start for start, end in claimed
        )
        if similar or len(body_text) > 200:
            continue
        if not re.search(r"[\\^_=<>]|\d\s*[+\-*/]\s*\d|[×÷≤≥≈√∑]", body_text):
            continue
        claimed.append((match.start(), match.end()))
        found.append(
            {
                "raw": match.group(0).strip(),
                "raw_context": _context(restored, match.start(), match.end()),
                "canonical": canonicalize_math(_latex_to_canonical(body)),
                "formula_type": "latex_display" if display else "latex_inline",
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )

    for start, end, raw in _extract_spans(restored):
        if any(start < c_end and end > c_start for c_start, c_end in claimed):
            continue
        found.append(
            {
                "raw": raw,
                "raw_context": _context(restored, start, end),
                "canonical": canonicalize_math(raw),
                "formula_type": "plain",
                "char_start": start,
                "char_end": end,
            }
        )

    for formula in found:
        _complete_formula(formula, restored)
    return [formula for formula in found if formula["canonical"]]


def _context(text: str, start: int, end: int, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:start] + text[start:end] + text[end:right]
    return re.sub(r"\s+", " ", snippet).strip()


def _split_sentences(text: str) -> list[tuple[int, str]]:
    sentences: list[tuple[int, str]] = []
    cursor = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        sentences.append((cursor, text[cursor:match.start()]))
        cursor = match.end()
    sentences.append((cursor, text[cursor:]))
    return sentences


def _find_target(prefix: str, canonical: str) -> str | None:
    lhs = canonical.split("=", 1)[0].strip() if "=" in canonical else ""
    lhs_match = re.fullmatch(r"[A-Za-z][A-Za-z0-9_\- ]{0,30}", lhs)
    if lhs_match and re.search(r"[A-Za-z]", lhs):
        candidate = lhs.strip()
        if candidate.casefold() in _KNOWN_TARGETS or candidate.isupper() or re.fullmatch(r"[A-Za-z]\d+", candidate):
            return candidate

    defined = re.search(
        r"([A-Za-z][A-Za-z0-9 ()\-']{0,40}?)\s+(?:is|are|can be)\s+"
        r"(?:defined|calculated|computed|given|measured|evaluated)\s+(?:as|by)\b",
        prefix,
        re.IGNORECASE,
    )
    if defined:
        candidate = defined.group(1).strip()
        if candidate and candidate.casefold() not in {"it", "this", "that", "which"}:
            return candidate

    lowered = prefix.casefold()
    best: str | None = None
    best_position = -1
    for name in _KNOWN_TARGETS:
        position = lowered.rfind(name)
        if position > best_position:
            best_position = position
            best = name
    return best


_ALLOWED_FUNCTIONS = set(_FUNCTION_NAMES) | {"sqrt", "sum", "prod", "integral", "root"}


def _complete_formula(formula: dict[str, Any], source: str) -> None:
    canonical = formula["canonical"]
    start = formula["char_start"]
    sentence_start = 0
    for offset, sentence in _split_sentences(source):
        if offset <= start < offset + len(sentence):
            sentence_start = offset
            break
    prefix = source[sentence_start:start]
    target = _find_target(prefix, canonical)
    formula["target"] = target
    formula["variables"] = _unique(
        match.group(0)
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9_']*(?:_[A-Za-z0-9]+)?", canonical)
        if match.group(0).casefold() not in _FUNCTION_NAMES
        and match.group(0).casefold() not in _GREEK.values()
    )
    formula["functions"] = _unique(
        match.group(0).casefold()
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9_]*", canonical)
        if match.group(0).casefold() in _ALLOWED_FUNCTIONS
    )
    formula["operators"] = _operators(canonical)
    # Numbers that directly follow a letter belong to identifiers (log2, F1).
    formula["numbers"] = _unique(
        match.group(0)
        for match in re.finditer(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", canonical)
    )
    formula["formula_kind"] = _classify(canonical, target, formula["operators"])
    formula["tokens"] = _retrieval_tokens(canonical)
    formula["signature"] = _signature(formula)
    computed = evaluate_formula(formula)
    if computed is not None:
        formula["computed"] = computed
    formula["metadata_tags"] = _formula_tags(formula, source)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _operators(canonical: str) -> list[str]:
    # canonicalize_math isolates binary operators with spaces, so splitting on
    # whitespace avoids counting hyphens inside compound terms (TF-IDF).
    operators = [
        _OPERATOR_NAMES[token]
        for token in canonical.split()
        if token in _OPERATOR_NAMES
    ]
    for word in ("sqrt", "sum", "prod", "integral"):
        if re.search(rf"\b{word}\b", canonical):
            operators.append(_OPERATOR_NAMES.get(word, word))
    if "^" in canonical:
        operators.append("power")
    return _unique(operators)


def _classify(canonical: str, target: str | None, operators: list[str]) -> str:
    if "equals" in operators:
        left = canonical.split("=", 1)[0].strip()
        if _is_arithmetic_expression(left):
            return "calculation"
        if target or re.match(r"^[A-Za-z]", left):
            return "definition"
        return "equation"
    if target and "(" in canonical:
        return "definition"
    return "expression"


def _retrieval_tokens(canonical: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_']*|\d+(?:\.\d+)?", canonical)
    return _unique(token for token in tokens if token.casefold() not in {"sqrt", "sum", "prod", "integral", "root"})


def _signature(formula: dict[str, Any]) -> str:
    parts = [
        f"kind={formula['formula_kind']}",
        "ops=" + ",".join(formula["operators"]),
        "fns=" + ",".join(sorted(name.casefold() for name in formula["functions"])),
        "vars=" + ",".join(sorted(name.casefold() for name in formula["variables"])),
        "nums=" + ",".join(formula["numbers"]),
    ]
    return "|".join(parts)


_DOMAIN_KEYWORDS = {
    "evaluation-metric": (
        "precision", "recall", "f1", "accuracy", "auc", "mrr", "rmse",
        "mape", "bleu", "meteor", "rouge", "perplexity", "tp", "fp", "tn", "fn",
    ),
    "text-representation": (
        "tf", "idf", "tfidf", "tf-idf", "vector", "cosine", "similarity",
        "embedding", "bag", "n-gram", "ngram", "one-hot", "vsm", "matrix",
    ),
    "neural-network": (
        "softmax", "sigmoid", "tanh", "relu", "gradient", "loss",
        "cross-entropy", "entropy", "neural", "layer", "weight", "bias", "logit",
    ),
}


def _domain(formula: dict[str, Any]) -> str:
    haystack = " ".join(
        str(formula.get(key) or "")
        for key in ("raw", "target", "raw_context")
    ).casefold()

    def hit(keyword: str) -> bool:
        return re.search(
            rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", haystack
        ) is not None

    scores = {
        domain: sum(hit(keyword) for keyword in keywords)
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=lambda domain: (scores[domain], domain))
    return best if scores[best] else "general-math"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _formula_tags(formula: dict[str, Any], source: str) -> list[str]:
    tags = {
        "math:formula",
        f"formula-type:{formula['formula_type']}",
        f"formula-kind:{formula['formula_kind']}",
        f"domain:{_domain(formula)}",
    }
    if formula.get("target"):
        tags.add(f"formula-target:{_slug(str(formula['target']))}")
    for operator in formula["operators"]:
        tags.add(f"operator:{operator}")
    for function in formula["functions"]:
        tags.add(f"function:{function.casefold()}")
    formula_text = f"{formula['raw']} {formula.get('raw_context', '')}"
    if re.search(r"[×÷·⋅∗−–—≤≥≠≈≡±∑Σ∏∫√∈∉→←⇒∧∨¬∂∇]", formula_text):
        tags.add("unicode-math")
    if formula["formula_type"].startswith("latex"):
        tags.add("has-latex")
    if "\ufffd" in source:
        tags.add("quality:contains-replacement-char")
    if _MOJIBAKE_RUN.search(source):
        tags.add("quality:mojibake-detected")
    if len(formula["raw"]) >= 400:
        tags.add("quality:span-truncated")
    if "computed" in formula:
        tags.add("evaluable:arithmetic")
    return sorted(tags)


# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------

_SAFE_FUNCTIONS = {
    "log2": math.log2,
    "log10": math.log10,
    "log": math.log,
    "ln": math.log,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_POWER = 32


def evaluate_arithmetic(expression: str) -> float | None:
    """Evaluate a numeric expression with an AST whitelist (never ``eval``)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
                raise ValueError("exponent too large")
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            else:
                result = left ** right
            if not math.isfinite(result) or abs(result) >= 1e12:
                raise ValueError("result out of range")
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords or node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError("unsupported call")
            return float(_SAFE_FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args)))
        if isinstance(node, ast.Name) and node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError("unsupported expression")

    try:
        value = evaluate(tree)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _is_arithmetic_expression(expression: str) -> bool:
    if not expression or not re.search(r"\d", expression):
        return False
    without_functions = re.sub(
        r"\b(?:log2|log10|log|ln|exp|sqrt|abs|round|min|max|pow)\b", "", expression
    )
    if re.search(r"[A-Za-z]", without_functions):
        return False
    return evaluate_arithmetic(expression.replace("^", "**")) is not None


def evaluate_formula(formula: dict[str, Any]) -> dict[str, Any] | None:
    canonical = formula["canonical"]
    candidates = [canonical]
    if "=" in canonical:
        left, right = (part.strip() for part in canonical.split("=", 1))
        # For "1/3 = 0.33" or "log2(4/3) = 0.4114" the interesting side is the
        # left one; a bare numeric right-hand side is just the stated result.
        candidates = [left] if re.fullmatch(r"\d+(?:\.\d+)?", right) else [right, left]
    for expression in candidates:
        if not _is_arithmetic_expression(expression):
            continue
        value = evaluate_arithmetic(expression.replace("^", "**"))
        if value is None:
            continue
        return {
            "expression": canonicalize_math(expression),
            "value": round(value, 6),
            "rounded": _format_number(value),
        }
    return None


def _format_number(value: float) -> str:
    if abs(value) >= 1e-4:
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Formula index (build time)
# ---------------------------------------------------------------------------

_CODE_FUNCTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in _FUNCTION_NAMES) + r")\s*\(",
    re.IGNORECASE,
)


def chunk_formula_text(chunk: dict[str, Any]) -> str:
    """Return the text used for formula extraction for one chunk."""
    chunk_type = chunk.get("chunk_type")
    content = chunk.get("content") or {}
    if chunk_type in {"text", "toc"}:
        return str(content.get("text") or "")
    if chunk_type == "table":
        parts = [str(content.get("caption") or ""), str(content.get("context") or "")]
        for row in content.get("rows") or []:
            parts.append("; ".join(f"{key}: {value}" for key, value in row.items()))
        return "\n".join(part for part in parts if part)
    return ""


def build_formula_index(
    chunks: list[dict[str, Any]], source_name: str = "unified.chunks.jsonl"
) -> dict[str, Any]:
    """Extract formulas from chunks and return a JSON-serialisable index."""
    formulas: list[dict[str, Any]] = []
    chunks_with_formulas: set[str] = set()
    type_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    tag_vocabulary: dict[str, set[str]] = {}

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        chunk_type = str(chunk.get("chunk_type") or "unknown")
        text = chunk_formula_text(chunk)
        records: list[dict[str, Any]] = []

        if text:
            seen_canonical: set[str] = set()
            detected = [
                formula
                for formula in detect_formulas(text)
                if not (
                    formula["canonical"] in seen_canonical
                    or seen_canonical.add(formula["canonical"])
                )
            ]
            for sequence, formula in enumerate(detected, start=1):
                record = {
                    "formula_id": f"{chunk_id}#f{sequence:02d}",
                    "source_chunk_id": chunk_id,
                    "document_id": chunk.get("document_id"),
                    "chunk_type": chunk_type,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "raw": formula["raw"],
                    "canonical": formula["canonical"],
                    "raw_context": formula["raw_context"],
                    "formula_type": formula["formula_type"],
                    "formula_kind": formula["formula_kind"],
                    "target": formula.get("target"),
                    "variables": formula["variables"],
                    "functions": formula["functions"],
                    "operators": formula["operators"],
                    "numbers": formula["numbers"],
                    "tokens": formula["tokens"],
                    "signature": formula["signature"],
                    "metadata_tags": formula["metadata_tags"],
                }
                if "computed" in formula:
                    record["computed"] = formula["computed"]
                record["embed_text"] = _formula_embed_text(record)
                records.append(record)

        code = str((chunk.get("content") or {}).get("code") or "")
        if chunk_type == "code" and code:
            functions = _unique(
                match.group(1).casefold() for match in _CODE_FUNCTION_RE.finditer(code)
            )
            if functions:
                record = {
                    "formula_id": f"{chunk_id}#f01",
                    "source_chunk_id": chunk_id,
                    "document_id": chunk.get("document_id"),
                    "chunk_type": chunk_type,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "raw": ", ".join(functions),
                    "canonical": ", ".join(f"{name}(...)" for name in functions),
                    "raw_context": str((chunk.get("content") or {}).get("symbol") or ""),
                    "formula_type": "code_function",
                    "formula_kind": "function_reference",
                    "target": None,
                    "variables": [],
                    "functions": functions,
                    "operators": [],
                    "numbers": [],
                    "tokens": functions,
                    "signature": "kind=function_reference|fns=" + ",".join(sorted(functions)),
                    "metadata_tags": sorted(
                        {
                            "math:code-function",
                            "formula-type:code_function",
                            "formula-kind:function_reference",
                            "domain:code",
                        }
                        | {f"function:{name}" for name in functions}
                        | {"source-chunk-type:code"}
                    ),
                    "embed_text": f"Code functions: {', '.join(functions)}",
                }
                records.append(record)

        if records:
            chunks_with_formulas.add(chunk_id)
            formulas.extend(records)

    for formula in formulas:
        type_counts[formula["formula_type"]] = type_counts.get(formula["formula_type"], 0) + 1
        kind_counts[formula["formula_kind"]] = kind_counts.get(formula["formula_kind"], 0) + 1
        domain = next(
            (tag.split(":", 1)[1] for tag in formula["metadata_tags"] if tag.startswith("domain:")),
            "unclassified",
        )
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        for tag in formula["metadata_tags"]:
            if ":" in tag:
                group, value = tag.split(":", 1)
                tag_vocabulary.setdefault(group, set()).add(value)

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "syllabus-qa formula index",
        "source_file": source_name,
        "document_id": next(
            (str(chunk.get("document_id")) for chunk in chunks if chunk.get("document_id")),
            None,
        ),
        "summary": {
            "chunks_scanned": len(chunks),
            "chunks_with_formulas": len(chunks_with_formulas),
            "formula_count": len(formulas),
            "by_formula_type": dict(sorted(type_counts.items())),
            "by_formula_kind": dict(sorted(kind_counts.items())),
            "by_domain": dict(sorted(domain_counts.items())),
        },
        "tag_vocabulary": {
            group: sorted(values) for group, values in sorted(tag_vocabulary.items())
        },
        "formulas": formulas,
    }


def _formula_embed_text(record: dict[str, Any]) -> str:
    parts = []
    if record.get("target"):
        parts.append(f"Formula for {record['target']}")
    parts.append(record["canonical"])
    parts.append(spell_operators(record["canonical"]))
    if record.get("variables"):
        parts.append("Variables: " + ", ".join(record["variables"]))
    if record.get("functions"):
        parts.append("Functions: " + ", ".join(record["functions"]))
    return ". ".join(part for part in parts if part)


def load_or_build_index(
    chunks_path: Path, index_path: Path
) -> dict[str, Any]:
    """Load the persisted JSON index, or rebuild it from the chunk file."""
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as source:
            index = json.load(source)
        if isinstance(index, dict) and isinstance(index.get("formulas"), list):
            return index
    with chunks_path.open(encoding="utf-8") as source:
        chunks = [json.loads(line) for line in source if line.strip()]
    return build_formula_index(chunks, source_name=chunks_path.name)


# ---------------------------------------------------------------------------
# Query analysis (query time)
# ---------------------------------------------------------------------------

_ALIASES = {
    "công thức": "formula",
    "cong thuc": "formula",
    "định nghĩa": "definition",
    "dinh nghia": "definition",
    "tính": "calculate",
    "tinh": "calculate",
    "độ chính xác": "precision accuracy",
    "do chinh xac": "precision accuracy",
    "triệu hồi": "recall",
    "trieu hoi": "recall",
    "điểm f1": "f1 score",
    "diem f1": "f1 score",
    "trọng số": "weight",
    "trong so": "weight",
    "tần suất": "frequency",
    "tan suat": "frequency",
    "nghịch đảo": "inverse",
    "nghich dao": "inverse",
    "tương đồng": "similarity",
    "tuong dong": "similarity",
}


def expand_aliases(question: str) -> str:
    lowered = question.casefold()
    additions = [
        english for vietnamese, english in _ALIASES.items() if vietnamese in lowered
    ]
    return " ".join(dict.fromkeys(additions))


def analyze_question(question: str) -> dict[str, Any]:
    """Analyse a user question and build a formula-preserving search string."""
    original = question.strip()
    restored = restore_mojibake(original)
    formulas = detect_formulas(restored)
    detected = bool(formulas)

    tags: set[str] = set()
    retrieval_terms: list[str] = []
    computed: list[dict[str, Any]] = []
    search_parts = [original]

    alias_text = expand_aliases(restored)
    if alias_text:
        search_parts.append(alias_text)

    for formula in formulas:
        canonical = formula["canonical"]
        search_parts.append(canonical)
        search_parts.append(spell_operators(canonical))
        if formula.get("target"):
            search_parts.append(str(formula["target"]))
        tags.update(formula["metadata_tags"])
        retrieval_terms.extend(formula["tokens"])
        retrieval_terms.extend(
            _SPELLED_OPERATORS.get(operator, operator)
            for operator in re.findall(r"<=|>=|!=|~=|[=+\-*/^<>]", canonical)
        )
        if "computed" in formula:
            computed.append(
                {
                    "expression": formula["computed"]["expression"],
                    "value": formula["computed"]["value"],
                    "rounded": formula["computed"]["rounded"],
                }
            )

    if detected:
        tags.add("math:query")

    return {
        "schema_version": SCHEMA_VERSION,
        "question": original,
        "detected": detected,
        "formula_count": len(formulas),
        "formulas": formulas,
        "computed": computed,
        "search_text": "\n".join(part for part in search_parts if part),
        "retrieval_terms": sorted(set(retrieval_terms)),
        "metadata_tags": sorted(tags),
        "notes": (
            ["Đã khôi phục lỗi mã hoá UTF-8/Windows-1252 trước khi phân tích."]
            if restored != original
            else []
        ),
    }


# ---------------------------------------------------------------------------
# Matching (query time)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")


def match_formulas(
    formulas: list[dict[str, Any]], analysis: dict[str, Any], top_n: int = 5
) -> list[dict[str, Any]]:
    """Rank indexed formulas against a formula-aware query analysis."""
    query_formulas = analysis.get("formulas") or []
    query_words = set(_WORD_RE.findall(str(analysis.get("question", "")).casefold()))

    q_vars: set[str] = set()
    q_functions: set[str] = set()
    q_operators: set[str] = set()
    q_tokens: set[str] = set()
    q_targets: set[str] = set()
    for formula in query_formulas:
        q_vars |= {value.casefold() for value in formula.get("variables", [])}
        q_functions |= {value.casefold() for value in formula.get("functions", [])}
        q_operators |= set(formula.get("operators", []))
        q_tokens |= {value.casefold() for value in formula.get("tokens", [])}
        if formula.get("target"):
            q_targets.add(str(formula["target"]).casefold())
    computed_values = [item["value"] for item in analysis.get("computed", [])]

    matches: list[dict[str, Any]] = []
    for formula in formulas:
        score = 0.0
        reasons: list[str] = []
        target = str(formula.get("target") or "").casefold()
        f_vars = {value.casefold() for value in formula.get("variables", [])}
        f_functions = {value.casefold() for value in formula.get("functions", [])}
        f_operators = set(formula.get("operators", []))
        f_tokens = {value.casefold() for value in formula.get("tokens", [])}

        if target and (target in q_targets or target in query_words):
            score += 6.0
            reasons.append("target-mentioned")
        elif target and any(word in query_words for word in target.split()):
            score += 3.0
            reasons.append("target-word-mentioned")

        if q_vars and f_vars:
            overlap = len(q_vars & f_vars) / len(q_vars | f_vars)
            if overlap:
                score += 4.0 * overlap
                reasons.append(f"variable-overlap:{overlap:.2f}")
        if q_functions & f_functions:
            score += 3.0
            reasons.append("function-overlap")
        if q_operators and f_operators:
            overlap = len(q_operators & f_operators) / len(q_operators | f_operators)
            if overlap:
                score += 2.0 * overlap
                reasons.append(f"operator-overlap:{overlap:.2f}")
        if q_tokens and f_tokens:
            overlap = len(q_tokens & f_tokens) / len(q_tokens | f_tokens)
            if overlap:
                score += 2.0 * overlap
                reasons.append(f"token-overlap:{overlap:.2f}")

        mentioned = {value for value in f_vars if value in query_words and len(value) > 2}
        if mentioned and not q_vars:
            score += 2.0 * len(mentioned) / max(1, len(f_vars))
            reasons.append("variables-mentioned")

        for value in computed_values:
            if any(
                _numbers_close(value, number) for number in formula.get("numbers", [])
            ):
                score += 5.0
                reasons.append("computed-value-matches")
                break

        if score > 0:
            matches.append(
                {
                    "score": round(score, 4),
                    "match_reasons": reasons,
                    "formula": formula,
                }
            )

    matches.sort(key=lambda item: (-item["score"], item["formula"]["formula_id"]))
    return matches[:top_n]


def _numbers_close(query_value: float, number: str) -> bool:
    try:
        value = float(number)
    except ValueError:
        return False
    if abs(value) < 1e-12:
        return abs(query_value) < 1e-12
    return abs(value - query_value) <= max(1e-3, 0.01 * abs(value))


def formulas_by_chunk(
    formulas: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for formula in formulas:
        grouped.setdefault(str(formula.get("source_chunk_id")), []).append(formula)
    return grouped
