# Xử lý công thức toán trong pipeline RAG (formula-aware retrieval)

> Tài liệu này giải thích cách hệ thống xử lý khi người dùng nhập một công thức
> toán (LaTeX hoặc biểu thức thường) và cách dữ liệu/chunk được giữ nguyên đặc
> trưng phép tính, gắn thẻ metadata, cùng định dạng file JSON đầu ra.

## 1. Vấn đề

Pipeline hiện tại (dense Gemini + BM25 + RRF + Jina rerank) coi mọi chunk là
văn xuôi. Điều này làm hỏng toán học ở ba điểm:

1. **Tokenizer của BM25** (`[\w]+`) bỏ mất `=`, `*`, `/`, `^`, `×`… nên câu hỏi
   chứa công thức gần như không khớp từ khoá với chunk chứa công thức.
2. **Embedding** làm mờ cấu trúc: `(2 × Precision × Recall) / (Precision + Recall)`
   và một câu văn mô tả cùng công thức có thể cho vector khác nhau.
3. **Chunk nguồn không có metadata** cho biết nó định nghĩa/tính toán công thức
   nào, nên không thể lọc, xếp hạng hay hiển thị "công thức này ở trang nào".

Ngoài ra dữ liệu thật có các dạng công thức rất khác nhau:

- công thức trong bảng: `Word: dog; TF score: ⅓ = 0.33; IDF score: log2(4/3) = 0.4114`;
- công thức định nghĩa trong câu: `F1 is defined as (2 × Precision × Recall) / (Precision + Recall)`;
- ký hiệu Unicode: `×`, `÷`, `≤`, `≈`, `∑`, `√`, `θ`, `⅓`;
- công thức bị PDF bóp méo: `similarity = cos θ = A · BA 2 B 2 = ∑i = 1 n AiBi …`;
- công thức LaTeX do người dùng gõ: `$F_1 = \frac{2PR}{P+R}$`;
- mã giả/code có công thức: `probs = softmax(logits)`.

## 2. Nguyên tắc thiết kế

- **Không sửa dữ liệu gốc.** `data/qdrant/practical_nlp.unified.chunks.jsonl`
  và các file chunk/text/table/code giữ nguyên từng ký tự. Mọi thứ mới là
  artefact dẫn xuất (`data/processed/*.json`).
- **Giữ hai bản của công thức:**
  - `raw`: nguyên văn trong nguồn (kể cả `×`, `∑`, `⅓`, lỗi PDF);
  - `canonical`: dạng chuẩn hoá có thể so khớp/tìm kiếm, giữ nguyên cấu trúc
    phép tính (`F1 = (2 * Precision * Recall) / (Precision + Recall)`).
- **Không dùng `eval`.** Biểu thức số học thuần được tính bằng `ast` + whitelist
  hàm (`log2`, `log10`, `sqrt`, `exp`, …), giới hạn số mũ và kết quả.
- **Metadata có thể kiểm chứng:** mọi công thức trỏ về `source_chunk_id`,
  `document_id`, `page_start/page_end` để citation vẫn hợp lệ.
- **Truy hồi lai vẫn là trung tâm:** công thức chỉ thêm một danh sách xếp hạng
  thứ ba vào RRF, không thay thế dense/BM25.

## 3. Build time — trích xuất và gắn thẻ

Module: `src/syllabus_qa/retrieval/formulas.py`
Script: `scripts/build_formula_index.py`

### 3.1 Quy trình

```text
chunk JSONL
  → chunk_formula_text(chunk)          # text/toc: content.text; table: caption+context+rows; code: tách riêng
  → detect_formulas(text)              # LaTeX + biểu thức thường
  → canonicalize_math(raw)             # Unicode/LaTeX → dạng tuyến tính giữ cấu trúc
  → trích target/variables/functions/operators/numbers
  → evaluate_formula()                 # chỉ khi là số học thuần
  → metadata_tags
  → JSON index (không đụng file nguồn)
```

### 3.2 Chuẩn hoá (canonicalisation)

| Nguồn | Canonical | Ghi chú |
|---|---|---|
| `2 × Precision` | `2 * Precision` | `×`, `·`, `⋅` → `*` |
| `10 ÷ 4` | `10 / 4` | `÷` → `/` |
| `x ≤ y`, `a ≈ b` | `x <= y`, `a ~= b` | giữ được toán tử so sánh |
| `x²`, `x^{2}` | `x^2` | số mũ không bị làm phẳng thành `x2` |
| `√(x+1)` | `sqrt(x + 1)` | |
| `\frac{TP}{TP+FN}` | `(TP) / (TP + FN)` | giữ thứ tự phép chia |
| `TF-IDF`, `t-SNE`, `ICD-10-CM` | giữ nguyên | gạch nối không bị hiểu là phép trừ |
| `∑i = 1 n AiBi` | `sum i = 1 n AiBi` | ký hiệu được "đánh vần" để BM25/embedding đọc được |
| `⅓ = 0.33` | `1 / 3 = 0.33` | NFKC + fraction slash |

Ngoài `canonical`, hàm `spell_operators()` tạo chuỗi bổ trợ cho tìm kiếm:

```text
(2 * Precision) / (Precision + Recall)
→ (2 multiply Precision) divide (Precision plus Recall)
```

### 3.3 Chống nhận diện sai (false positive)

Một span chỉ được coi là công thức khi có **toán tử mạnh** (`=`, `*`, `/`, `^`,
`√`, `∑`, `≤`, `≈`, `→`…). Các trường hợp sau bị loại:

- ISBN/số điện thoại/dải trang: `978-1-492-05405-4`, `800-998-9938`, `122-123, 327-329`;
- từ ghép: `state-of-the-art`, `NLP/ML`, `BoW/TF-IDF`, `ICD-10-CM`;
- tên kèm năm: `(NLPBA/BioNLP) (2004)`;
- cặp `$` là tiền tệ chứ không phải LaTeX: `$10,000 … $`;
- chữ tiếng Việt bị cắt thành biến: `log2(4/3) b` (từ "bằng") được cắt đuôi.

### 3.4 Thẻ metadata

Mỗi công thức có danh sách `metadata_tags` dạng `nhóm:giá_trị`:

| Nhóm | Ví dụ | Ý nghĩa |
|---|---|---|
| `math` | `math:formula`, `math:query`, `math:code-function` | đánh dấu có toán |
| `formula-type` | `plain`, `latex_inline`, `latex_display`, `code_function` | nguồn biểu diễn |
| `formula-kind` | `definition`, `calculation`, `expression`, `function_reference` | loại công thức |
| `formula-target` | `f1`, `tf-idf`, `similarity` | đại lượng được định nghĩa |
| `domain` | `evaluation-metric`, `text-representation`, `neural-network`, `general-math`, `code` | lĩnh vực |
| `operator` | `equals`, `multiply`, `divide`, `add`, `power`, `sum` | toán tử có mặt |
| `function` | `log2`, `softmax`, `sigmoid` | hàm toán học |
| `unicode-math` / `has-latex` | — | dấu hiệu biểu diễn |
| `evaluable:arithmetic` | — | có thể tính ra số |
| `quality:*` | `contains-replacement-char`, `mojibake-detected`, `span-truncated` | cảnh báo chất lượng |
| `source-chunk-type:*` | `text`, `table`, `code` | loại chunk nguồn |

`tag_vocabulary` trong JSON tổng hợp toàn bộ giá trị đã gặp, thuận tiện cho UI
hiển thị bộ lọc.

### 3.5 Kết quả trên dữ liệu hiện tại

```text
practical_nlp.unified.chunks.jsonl → 637 chunks
  chunks_with_formulas : 11
  formula_count        : 22
  by_formula_kind      : calculation 10, definition 6, expression 5, function_reference 1
  by_domain            : text-representation 17, evaluation-metric 2, general-math 2, code 1
```

Ví dụ bản ghi:

```json
{
  "formula_id": "practical-natural-language-processing:table2-2:p99-p100:full#f01",
  "source_chunk_id": "practical-natural-language-processing:table2-2:p99-p100:full",
  "chunk_type": "table",
  "page_start": 99,
  "page_end": 100,
  "raw": "(2 × Precision × Recall) / (Precision + Recall)",
  "canonical": "(2 * Precision * Recall) / (Precision + Recall)",
  "formula_type": "plain",
  "formula_kind": "definition",
  "target": "F1",
  "variables": ["Precision", "Recall"],
  "functions": [],
  "operators": ["multiply", "divide", "add"],
  "numbers": ["2"],
  "signature": "kind=definition|ops=multiply,divide,add|fns=|vars=precision,recall|nums=2",
  "metadata_tags": ["domain:evaluation-metric", "formula-kind:definition",
                    "formula-target:f1", "math:formula", "operator:add",
                    "operator:divide", "operator:multiply", "unicode-math"],
  "embed_text": "Formula for F1. (2 * Precision * Recall) / (Precision + Recall). ..."
}
```

## 4. Query time — người dùng nhập công thức

Module dùng chung `analyze_question(question)` cho cả CLI và server.

### 4.1 Phân tích câu hỏi

```text
"Tính log2(4/3) bằng bao nhiêu?"
→ detected: true
→ formulas[0]:
    raw: "log2(4/3)"
    canonical: "log2(4 / 3)"
    functions: ["log2"]
    operators: ["divide"]
    numbers: ["4", "3"]
→ computed: { expression: "log2(4 / 3)", value: 0.415037, rounded: "0.415037" }
→ search_text:
    Tính log2(4/3) bằng bao nhiêu?
    calculate
    log2(4 / 3)
    log2(4 divide 3)
```

`search_text` được dùng cho **cả** embedding và BM25. Nhờ đó:

- dense thấy cả câu hỏi tự nhiên lẫn công thức chuẩn hoá;
- BM25 thấy các từ `divide`, `multiply`, `plus`… và các định danh `log2`, `Precision`;
- không có công thức nào bị mất toán tử.

### 4.2 So khớp cấu trúc

`match_formulas()` cho điểm từng công thức trong index:

| Tín hiệu | Điểm |
|---|---|
| `target` xuất hiện trong câu hỏi (`f1`, `cosine similarity`…) | +6 |
| trùng biến (Jaccard) | +4 × Jaccard |
| trùng hàm | +3 |
| trùng toán tử (Jaccard) | +2 × Jaccard |
| trùng token | +2 × Jaccard |
| giá trị vừa tính khớp một con số trong công thức (±1%) | +5 |

Ví dụ `F1 = (2 × Precision × Recall) / (Precision + Recall)`:

```text
11.67  table2-2 (F1)   target-mentioned, variable-overlap:0.67, operator-overlap:0.75
 8.29  text:0282 (f1)  target-mentioned, variable-overlap:0.25, operator-overlap:0.50
```

Câu hỏi chỉ nêu tên (`"Công thức F1 score là gì?"`) vẫn khớp nhờ
`target-mentioned` — đây là lý do formula matching chạy kể cả khi câu hỏi
không chứa ký hiệu toán.

### 4.3 Tích hợp vào RRF (server)

`ui_server.py` thêm danh sách xếp hạng thứ ba:

```text
RRF(chunk) = 1/(k + dense_rank) + 1/(k + bm25_rank) + 1/(k + formula_rank)
```

- chỉ công thức có `score >= FORMULA_MIN_SCORE (4.0)` mới vào `formula_rank`;
- khi câu hỏi nêu tên công thức nhưng không có ký hiệu, `embed_text` của các
  công thức khớp được nối vào query cho dense/BM25;
- kết quả trả về mỗi source có thêm:
  - `formula_ids`: các công thức nằm trong chunk đó;
  - `formula_tags`: hợp các metadata tag của các công thức đó;
- `retrieval_trace` có thêm `formula_detected`, `formula_count`,
  `formula_candidates`, `formula_match_ids`, `formula_search_used`;
- prompt sinh câu trả lời được bổ sung quy tắc: giữ nguyên toán tử/thứ tự phép
  tính, không diễn giải công thức thành lời mà bỏ mất công thức; giá trị
  `computed` là phép tính của người dùng, không phải trích từ sách.

Hai endpoint:

| Endpoint | Chức năng |
|---|---|
| `POST /api/chat` | trả lời đầy đủ; JSON có thêm `formula_analysis`, `formula_matches` |
| `POST /api/formula/analyze` | chỉ phân tích công thức + so khớp index (không gọi LLM/Qdrant/Jina) |

## 5. File JSON đầu ra

### 5.1 Index công thức — `data/processed/practical_nlp.formulas.json`

```json
{
  "schema_version": "1.0",
  "generator": "syllabus-qa formula index",
  "source_file": "practical_nlp.unified.chunks.jsonl",
  "document_id": "practical-natural-language-processing",
  "summary": {
    "chunks_scanned": 637,
    "chunks_with_formulas": 11,
    "formula_count": 22,
    "by_formula_type": { "code_function": 1, "plain": 21 },
    "by_formula_kind": { "calculation": 10, "definition": 6,
                         "expression": 5, "function_reference": 1 },
    "by_domain": { "code": 1, "evaluation-metric": 2,
                   "general-math": 2, "text-representation": 17 }
  },
  "tag_vocabulary": {
    "domain": ["code", "evaluation-metric", "general-math", "text-representation"],
    "operator": ["add", "divide", "equals", "multiply", "power", "sum", "..."],
    "...": ["..."]
  },
  "formulas": [ /* các bản ghi như mục 3.5 */ ]
}
```

### 5.2 Kết quả truy vấn — `data/processed/formula_query_results.json`

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-11-11T12:00:00+00:00",
  "index_source": "practical_nlp.formulas.json",
  "index_formula_count": 22,
  "results": [
    {
      "question": "Tính log2(4/3) bằng bao nhiêu?",
      "formula_analysis": {
        "detected": true,
        "formula_count": 1,
        "formulas": [
          {
            "raw": "log2(4/3)",
            "canonical": "log2(4 / 3)",
            "formula_type": "plain",
            "formula_kind": "expression",
            "target": null,
            "operators": ["divide"],
            "variables": [],
            "functions": ["log2"],
            "metadata_tags": ["function:log2", "math:formula", "operator:divide", "..."],
            "computed": { "expression": "log2(4 / 3)",
                          "value": 0.415037, "rounded": "0.415037" }
          }
        ],
        "computed": [ /* ... */ ],
        "search_text": "...",
        "metadata_tags": [ /* ... */ ]
      },
      "formula_matches": [
        {
          "score": 10.5,
          "match_reasons": ["function-overlap", "operator-overlap:0.50",
                            "token-overlap:0.75"],
          "formula": { /* bản ghi index đầy đủ */ },
          "source_citation": "PDF trang 121",
          "source_chunk_type": "table"
        }
      ]
    }
  ]
}
```

## 6. Cách chạy

```powershell
# 1) Sinh index công thức (JSON) từ chunk hiện có
python scripts/build_formula_index.py

# 2) Phân tích câu hỏi có công thức và xuất JSON (không cần API key)
python scripts/query_formula.py `
  --question "Tính log2(4/3) bằng bao nhiêu?" `
  --question "F1 = (2 × Precision × Recall) / (Precision + Recall) nghĩa là gì?" `
  --output data/processed/formula_query_results.json

# 3) Chạy server chat (dùng index tự động, rebuild trong bộ nhớ nếu thiếu file)
python ui_server.py --port 8000

# 4) Kiểm thử
uv run pytest -q
```

Nếu file index thiếu, `ui_server.py` và `scripts/query_formula.py` tự dựng lại
từ `data/qdrant/practical_nlp.unified.chunks.jsonl` mà không ghi đè dữ liệu.

## 7. Kiểm thử

`tests/test_formulas.py` bao phủ:

- sửa mojibake UTF-8/CP1252;
- chuẩn hoá Unicode/LaTeX (kể cả `TF-IDF` không thành phép trừ);
- nhận diện F1, `log2(4/3)`, word analogy, LaTeX inline;
- loại ISBN/số điện thoại/dải trang/từ ghép/tiền tệ;
- tính số học an toàn (`__import__`, lũy thừa lớn, chia 0 → từ chối);
- phân tích câu hỏi tiếng Việt (không dính chữ "bằng" vào công thức);
- dựng index JSON, khử trùng lặp, giữ provenance;
- so khớp theo công thức và theo tên;
- smoke test trên dữ liệu thật (F1 phải trỏ đúng `table2-2`, trang 99–100).

## 8. Hạn chế và hướng mở rộng

- Công thức bị PDF bóp méo (mất dấu trị tuyệt đối, chỉ số dưới dính nhau) chỉ
  được giữ nguyên `raw` + chuẩn hoá một phần; không tự "đoán" lại ký tự gốc.
  Các chunk như vậy được gắn `quality:*` để người dùng biết.
- Chưa OCR công thức trong ảnh/PDF scan; chưa parse MathML.
- Code chunk chỉ được gắn `function_reference` (tên hàm toán học), không trích
  toàn bộ biểu thức trong code để tránh nhiễu.
- Nếu muốn tìm kiếm vector riêng cho công thức, có thể ingest các bản ghi
  `embed_text` trong index thành point Qdrant với `chunk_type = "formula"` và
  `relations.parent_id = source_chunk_id`; hiện tại hệ thống dùng index trực
  tiếp nên không cần bước này.
- Từ điển alias Việt–Anh trong `formulas.py` chỉ phục vụ các thuật ngữ phổ
  biến; có thể mở rộng khi thêm tài liệu mới.
