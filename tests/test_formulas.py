import json
from pathlib import Path

import pytest

from syllabus_qa.retrieval.formulas import (
    analyze_question,
    build_formula_index,
    canonicalize_math,
    detect_formulas,
    evaluate_arithmetic,
    match_formulas,
    restore_mojibake,
    spell_operators,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_CHUNKS = ROOT / "data" / "qdrant" / "practical_nlp.unified.chunks.jsonl"

F1_SENTENCE = (
    "F1 is defined as (2 × Precision × Recall) / (Precision + Recall)."
)


def test_restore_mojibake_repairs_cp1252_mangled_symbols() -> None:
    mojibake = "F1 is defined as (2 Ã— Precision) and itâ€™s exact."
    fixed = restore_mojibake(mojibake)
    assert "×" in fixed
    assert "’" in fixed
    # Legitimate typographic quotes are left untouched.
    assert restore_mojibake("we’ll stay") == "we’ll stay"


def test_canonicalize_preserves_operators_and_exponents() -> None:
    assert canonicalize_math("2 × Precision × Recall") == "2 * Precision * Recall"
    assert canonicalize_math("x ≤ y") == "x <= y"
    assert canonicalize_math("x^{2} + y^{2}") == "x^2 + y^2"
    assert canonicalize_math("x² + y²") == "x^2 + y^2"
    assert "sqrt" in canonicalize_math("√(x + 1)")
    # Hyphenated terms are not turned into subtraction.
    assert canonicalize_math("TF-IDF score = TF * IDF") == "TF-IDF score = TF * IDF"


def test_spell_operators_exposes_math_to_tokenizers() -> None:
    spelled = spell_operators("(2 * Precision) / (Precision + Recall)")
    assert "multiply" in spelled
    assert "divide" in spelled
    assert "plus" in spelled


def test_detect_f1_definition_keeps_structure_and_tags() -> None:
    (formula,) = detect_formulas(F1_SENTENCE)

    assert formula["canonical"] == "(2 * Precision * Recall) / (Precision + Recall)"
    assert formula["target"] == "F1"
    assert formula["formula_kind"] == "definition"
    assert formula["formula_type"] == "plain"
    assert set(formula["operators"]) >= {"multiply", "divide", "add"}
    assert set(formula["variables"]) >= {"Precision", "Recall"}
    assert "formula-target:f1" in formula["metadata_tags"]
    assert "domain:evaluation-metric" in formula["metadata_tags"]
    assert "unicode-math" in formula["metadata_tags"]


def test_detect_log2_expression_computes_value_safely() -> None:
    (formula,) = detect_formulas("We compute log2(4/3) = 0.4114 for the IDF.")

    assert formula["functions"] == ["log2"]
    assert formula["canonical"] == "log2(4 / 3) = 0.4114"
    assert formula["computed"]["value"] == pytest.approx(0.415037, abs=1e-6)
    assert "evaluable:arithmetic" in formula["metadata_tags"]


def test_detect_word_analogy_expression() -> None:
    (formula,) = detect_formulas("King – Man + Woman ≈ Queen")
    assert formula["canonical"] == "King - Man + Woman ~= Queen"
    assert "approx-equal" in formula["operators"]


@pytest.mark.parametrize(
    "text",
    [
        "ISBN 978-1-492-05405-4.",
        "Call 800-998-9938 for details.",
        "See pages 122-123, 327-329.",
        "state-of-the-art models from chapter 3-4.",
        "NLP/ML systems use BoW/TF-IDF.",
        "(NLPBA/BioNLP) (2004) released a corpus.",
    ],
)
def test_prose_and_ranges_are_not_formulas(text: str) -> None:
    assert detect_formulas(text) == []


def test_detect_latex_inline_formula() -> None:
    (formula,) = detect_formulas(r"In LaTeX: $F_1 = \frac{2PR}{P+R}$ means F1.")
    assert formula["formula_type"] == "latex_inline"
    assert formula["canonical"] == "F_1 = (2PR) / (P + R)"
    assert "has-latex" in formula["metadata_tags"]


def test_detect_currency_pair_is_not_latex() -> None:
    text = "Costs $10,000 and €1000 for this tokenizer, while $ budgets vary."
    assert all(formula["formula_type"] != "latex_inline" for formula in detect_formulas(text))


def test_evaluate_arithmetic_rejects_unsafe_input() -> None:
    assert evaluate_arithmetic("sqrt(4)") == 2.0
    assert evaluate_arithmetic("__import__('os')") is None
    assert evaluate_arithmetic("2 ** 64") is None
    assert evaluate_arithmetic("1 / 0") is None


def test_analyze_vietnamese_question_preserves_formula_and_computes() -> None:
    analysis = analyze_question("Tính log2(4/3) bằng bao nhiêu?")

    assert analysis["detected"] is True
    assert analysis["formula_count"] == 1
    formula = analysis["formulas"][0]
    assert formula["raw"] == "log2(4/3)"
    assert formula["canonical"] == "log2(4 / 3)"
    assert analysis["computed"][0]["rounded"] == "0.415037"
    assert "log2(4 / 3)" in analysis["search_text"]
    assert "math:query" in analysis["metadata_tags"]
    json.dumps(analysis, ensure_ascii=False)  # JSON-serialisable


def test_analyze_question_without_formula_is_not_augmented() -> None:
    analysis = analyze_question("Công thức F1 score là gì?")
    assert analysis["detected"] is False
    assert analysis["formulas"] == []
    assert analysis["search_text"].startswith("Công thức F1 score là gì?")


def _fixture_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "doc:text:0001",
            "document_id": "doc",
            "chunk_type": "text",
            "page_start": 10,
            "page_end": 10,
            "content": {"text": F1_SENTENCE + " " + F1_SENTENCE},
        },
        {
            "chunk_id": "doc:table1:full",
            "document_id": "doc",
            "chunk_type": "table",
            "page_start": 12,
            "page_end": 12,
            "content": {
                "table_id": "table1",
                "caption": "Table 1. Cosine similarity",
                "context": "Similarity of two vectors",
                "rows": [{"Formula": "similarity = cos θ = A · B / (|A| |B|)"}],
            },
        },
        {
            "chunk_id": "doc:code:softmax",
            "document_id": "doc",
            "chunk_type": "code",
            "page_start": 20,
            "page_end": 20,
            "content": {
                "symbol": "softmax_layer",
                "language": "python",
                "code": "probs = softmax(logits)",
            },
        },
    ]


def test_build_formula_index_is_json_ready_and_deduplicated() -> None:
    index = build_formula_index(_fixture_chunks(), source_name="fixture.jsonl")

    assert index["schema_version"] == "1.0"
    assert index["summary"]["chunks_scanned"] == 3
    assert index["summary"]["formula_count"] >= 2
    ids = [formula["formula_id"] for formula in index["formulas"]]
    assert len(ids) == len(set(ids))
    # The repeated F1 sentence inside one chunk is stored once.
    f1_records = [
        formula
        for formula in index["formulas"]
        if (formula.get("target") or "").casefold() == "f1"
    ]
    assert len(f1_records) == 1
    assert f1_records[0]["source_chunk_id"] == "doc:text:0001"
    assert f1_records[0]["page_start"] == 10
    assert "tag_vocabulary" in index
    json.dumps(index, ensure_ascii=False)


def test_match_formulas_ranks_the_defining_chunk_first() -> None:
    index = build_formula_index(_fixture_chunks())
    analysis = analyze_question("F1 = 2 × Precision × Recall / (Precision + Recall) là gì?")

    matches = match_formulas(index["formulas"], analysis, top_n=3)
    assert matches
    top = matches[0]
    assert top["formula"]["target"] == "F1"
    assert top["formula"]["source_chunk_id"] == "doc:text:0001"
    assert top["score"] >= 6
    assert "target-mentioned" in top["match_reasons"]


def test_match_formulas_by_name_without_symbols() -> None:
    index = build_formula_index(_fixture_chunks())
    analysis = analyze_question("Công thức F1 score là gì?")

    matches = match_formulas(index["formulas"], analysis, top_n=3)
    assert matches
    assert matches[0]["formula"]["target"] == "F1"
    assert "target-mentioned" in matches[0]["match_reasons"]


@pytest.mark.skipif(not REAL_CHUNKS.is_file(), reason="unified chunks are not present")
def test_real_corpus_index_preserves_provenance() -> None:
    with REAL_CHUNKS.open(encoding="utf-8") as source:
        chunks = [json.loads(line) for line in source if line.strip()]
    index = build_formula_index(chunks, source_name=REAL_CHUNKS.name)
    formulas = index["formulas"]

    assert formulas
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    assert all(formula["source_chunk_id"] in chunk_ids for formula in formulas)
    f1_formulas = [
        formula
        for formula in formulas
        if (formula.get("target") or "").casefold() == "f1"
    ]
    assert any(
        formula["source_chunk_id"]
        == "practical-natural-language-processing:table2-2:p99-p100:full"
        for formula in f1_formulas
    )
    json.dumps(index, ensure_ascii=False)
