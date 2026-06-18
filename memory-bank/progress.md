# Progress

## Đã Hoàn Thành
- **Khởi tạo dự án**: Multi-Agent Courtroom Simulation Framework.
- **Phase 1 ViLQA scaffold**: data loader, split, `DebateOrchestrator`, baselines, retrieval, memory, evaluator EM/F1.
- **P0/P1 pipeline**: LLM providers, judge fallback, semantic rerank, memory ablation, debate loop, ablation matrix script.
- **Phase 3 courtroom scaffold (items 10–13)**:
  - Agents theo vai trò pháp lý: prosecutor, defense, defendant, judge LJP.
  - Courtroom protocol 3 giai đoạn: opening → debate → judgment (closing + deliberation + ruling).
  - `CourtCase` schema + loaders + pilot case VN.
  - `LJPEvaluator` cho charge/article/sentence metrics.
  - Backward compat Phase 1 qua `DebateAgent` + `compat.py`.

## Kết Quả Thí Nghiệm
| Thí nghiệm | Mô tả | Model | Dataset | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| Unit tests offline | `python -m unittest discover -s tests` | Mock | Synthetic | **23 tests pass** | Bao gồm Phase 3 courtroom |
| Phase 3 courtroom smoke | `python -m src.main --run-courtroom --llm mock` | MockLLM | case_01_theft.json | Session hoàn tất | Smoke kỹ thuật |
| P1 debate smoke | mock batch debate | MockLLM | ALQAC | OK | Phase 1 không bị break |

## Đang Thực Hiện
- Validation courtroom pilot bằng LLM thật (Gemini/local).
- Mapping dataset SimuCourt/VLegal-Bench khi tải HF.

## Còn Lại
- [ ] Batch courtroom runner + metrics aggregation cho LJP.
- [ ] Citation validity / hallucination checker tích hợp (LegalCiteBench/LegalHalBench style).
- [ ] Human eval rubric wiring cho courtroom sessions.
- [ ] P1 Gemini validation và ablation matrix `--execute`.
- [ ] So sánh Phase 1 QA debate vs Phase 3 courtroom LJP trên cùng case adapter.

## Vấn Đề Đã Biết
- `load_simucourt` / `load_vlegal` phụ thuộc HF dataset schema; có thể cần chỉnh field mapping sau khi tải thật.
- `MockLLM` không phản ánh chất lượng LJP thực tế.
- Courtroom session gọi cả `render_ljp_verdict` (structured) và `render_verdict` (QA-style) — verdict QA là backward-compat view.
- Dataset loaders SimuCourt dùng `law-ai/SimuCourt` — cần xác nhận repo HF chính xác khi chạy production.
