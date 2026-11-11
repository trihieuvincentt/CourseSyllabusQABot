"""Build the formula index (JSON) from a unified chunk JSONL file.

Usage (from the repository root):

    python scripts/build_formula_index.py
    python scripts/build_formula_index.py --source data/qdrant/practical_nlp.unified.chunks.jsonl \
        --output data/processed/practical_nlp.formulas.json

The index is a derived artefact: chunks and embed_text are never modified.
Every formula record keeps its verbatim ``raw`` text plus ``source_chunk_id``
and page provenance so answers can still be cited back to the PDF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from syllabus_qa.retrieval.formulas import build_formula_index

DEFAULT_SOURCE = ROOT / "data" / "qdrant" / "practical_nlp.unified.chunks.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "practical_nlp.formulas.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        print(f"Source JSONL not found: {args.source}", file=sys.stderr)
        return 2

    with args.source.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    index = build_formula_index(chunks, source_name=args.source.name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(index, target, ensure_ascii=False, indent=2)
        target.write("\n")

    summary = index["summary"]
    print(f"Chunks scanned:          {summary['chunks_scanned']}")
    print(f"Chunks with formulas:    {summary['chunks_with_formulas']}")
    print(f"Formulas extracted:      {summary['formula_count']}")
    print(f"By formula type:         {summary['by_formula_type']}")
    print(f"By formula kind:         {summary['by_formula_kind']}")
    print(f"By domain:               {summary['by_domain']}")
    print(f"Index written to:        {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
