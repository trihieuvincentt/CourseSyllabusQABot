# Course Syllabus Q&A Bot

A Python application for question answering over course syllabi with source citations.

## Development setup

Requirements: Python 3.12+ and `uv`.

```powershell
uv sync
uv run pytest -q
uv run ruff check .
uv run uvicorn syllabus_qa.main:app --reload --app-dir src
```

Health check: `http://127.0.0.1:8000/health`.

Secrets belong in a local `.env` file and must not be committed. Do not upload source documents or send them to external services without confirming permission for that use.

## Math formula handling

Questions may contain math (LaTeX or plain expressions such as
`log2(4/3)` or `F1 = (2 × Precision × Recall) / (Precision + Recall)`). The
hybrid retriever canonicalises the formula without losing its operators, adds a
formula rank list to the RRF fusion, annotates sources with formula metadata
tags, and can safely evaluate purely numeric expressions. See
[docs/formula_handling.md](docs/formula_handling.md) for the full design,
metadata schema, and JSON output formats.

```powershell
# Build the formula index (JSON) from the unified chunks
python scripts/build_formula_index.py

# Analyse formula questions offline and write a JSON report
python scripts/query_formula.py --question "Tính log2(4/3) bằng bao nhiêu?"
```

`ui_server.py` additionally exposes `POST /api/formula/analyze` for formula-only
analysis (no LLM or vector search involved).
