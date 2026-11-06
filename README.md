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

## Xử lý công thức toán

Input có dạng biểu thức hoặc phương trình được nhận diện trước bước retrieval
để không bị tìm nhầm trong dữ liệu syllabus. Máy chủ xử lý trực tiếp các phép
tính số, phương trình bậc nhất và bậc hai, ví dụ 2 + 3 * 4, sqrt(16),
2x + 3 = 7 và x^2 - 5x + 6 = 0. Bộ phân tích dùng AST với whitelist
toán tử và hàm, không chạy Python tùy ý và không gửi công thức sang LLM để
tính toán. Nếu input không thuộc grammar được hỗ trợ, máy chủ trả lỗi rõ ràng
để người dùng có thể viết lại công thức.
