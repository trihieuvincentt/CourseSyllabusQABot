"""Analyse user questions that contain math formulas and write JSON output.

This script runs fully offline (no Gemini/Qdrant/Jina keys needed). For each
question it:

1. detects and canonicalises the formula(s) while keeping their raw text;
2. safely evaluates purely numeric expressions with an AST whitelist;
3. matches the query against the formula index (structural + lexical score);
4. attaches metadata tags and source-chunk provenance;
5. writes everything to a JSON file.

Usage (from the repository root):

    python scripts/query_formula.py --question "Tính log2(4/3) bằng bao nhiêu?" \
        --question "F1 = (2 × Precision × Recall) / (Precision + Recall) nghĩa là gì?" \
        --output data/processed/formula_query_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from syllabus_qa.retrieval.formulas import (
    analyze_question,
    load_or_build_index,
    match_formulas,
)

DEFAULT_SOURCE = ROOT / "data" / "qdrant" / "practical_nlp.unified.chunks.jsonl"
DEFAULT_INDEX = ROOT / "data" / "processed" / "practical_nlp.formulas.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "formula_query_results.json"
DEFAULT_QUESTIONS = (
    "Tính log2(4/3) bằng bao nhiêu?",
    "F1 = (2 × Precision × Recall) / (Precision + Recall) nghĩa là gì?",
    "Công thức F1 score là gì?",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", action="append", help="Question to analyse; may repeat.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    questions = args.question or list(DEFAULT_QUESTIONS)
    if not args.source.is_file():
        print(f"Source JSONL not found: {args.source}", file=sys.stderr)
        return 2

    index = load_or_build_index(args.source, args.index)
    formulas = index["formulas"]

    with args.source.open(encoding="utf-8") as stream:
        chunks = {
            row["chunk_id"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }

    results = []
    for question in questions:
        analysis = analyze_question(question)
        matches = match_formulas(formulas, analysis, top_n=args.top_n)
        for match in matches:
            chunk = chunks.get(match["formula"]["source_chunk_id"], {})
            start, end = chunk.get("page_start"), chunk.get("page_end")
            match["source_citation"] = (
                f"PDF trang {start}" if start == end else f"PDF trang {start}–{end}"
            )
            match["source_chunk_type"] = chunk.get("chunk_type")
        results.append(
            {
                "question": question,
                "formula_analysis": analysis,
                "formula_matches": matches,
            }
        )

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "index_source": str(args.index.name),
        "index_formula_count": len(formulas),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")

    for result in results:
        analysis = result["formula_analysis"]
        print(f"Question: {result['question']}")
        print(f"  detected: {analysis['detected']}, formulas: {analysis['formula_count']}")
        for computed in analysis["computed"]:
            print(f"  computed: {computed['expression']} = {computed['rounded']}")
        for match in result["formula_matches"]:
            formula = match["formula"]
            print(
                f"  match {match['score']:.2f}: {formula['formula_id']} "
                f"({formula.get('target') or formula['formula_kind']}) "
                f"-> {match['source_citation']} | {', '.join(match['match_reasons'][:3])}"
            )
    print(f"JSON written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
