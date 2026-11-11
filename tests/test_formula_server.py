import json
import sys
from pathlib import Path

from syllabus_qa.retrieval.formulas import analyze_question

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_server import compact_formula_analysis, get_formula_index


def test_compact_analysis_is_json_serialisable() -> None:
    analysis = analyze_question("Tính log2(4/3) bằng bao nhiêu?")
    compact = compact_formula_analysis(analysis)

    assert compact["detected"] is True
    assert compact["formula_count"] == 1
    assert compact["computed"][0]["rounded"] == "0.415037"
    assert "canonical" in compact["formulas"][0]
    json.dumps(compact, ensure_ascii=False)


def test_formula_index_loads_with_f1_record() -> None:
    index = get_formula_index()

    assert index.get("formulas")
    assert any(
        (formula.get("target") or "").casefold() == "f1"
        for formula in index["formulas"]
    )
