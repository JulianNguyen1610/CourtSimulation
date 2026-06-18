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
| Unit tests offline | `python -m unittest discover -s tests` | Mock | Synthetic | **28 tests pass** | Bao gồm Phase 3 courtroom + answer postprocess |
| Phase 3 courtroom smoke | `python -m src.main --run-courtroom --llm mock` | MockLLM | case_01_theft.json | Session hoàn tất | Smoke kỹ thuật |
| P1 debate smoke | mock batch debate | MockLLM | ALQAC | OK | Phase 1 không bị break |
| P1 local direct validation | `--method direct --split validation --limit 0` | dolphin3:latest | ALQAC validation 53 | EM=0.2642, F1=0.6802 | Direct baseline thắng rõ |
| P1 local debate validation | `--method debate --split validation --limit 0 --rounds 1` | dolphin3:latest | ALQAC validation 53 | EM=0.0755, F1=0.3677 | Fallback thấp (0.0094); lỗi over-extraction/paraphrase |
| P1 local debate after first fix | `--method debate --split validation --limit 10 --rounds 1` | dolphin3:latest | ALQAC validation 10 | EM=0.1000, F1=0.4178 | Vẫn còn over-extraction; đã bổ sung prompt/postprocess lần 2 |
| P1 local both (postprocess đồng đều) | `--method both --split validation --limit 0` | dolphin3:latest | ALQAC validation 53 | direct EM=0.2075/F1=0.6572; debate EM=0.2075/F1=0.5460 | EM ngang, direct F1 cao hơn |
| **P1 local both (model lớn)** | `--method both --split validation --limit 0 --rounds 1` | qwen3.5:9b | ALQAC validation 53 | **debate EM=0.4717/F1=0.8106 > direct EM=0.0189/F1=0.4034** | Debate thắng lớn; direct sụp do `max_output_tokens=128` (config Linux cũ) cắt JSON |

## Đang Thực Hiện
- P1 error analysis cho Ollama debate trước khi chạy ablation matrix; chưa claim debate cải thiện so với direct.
- Validation courtroom pilot bằng LLM thật (Gemini/local).
- Mapping dataset SimuCourt/VLegal-Bench khi tải HF.

## Còn Lại
- [ ] Batch courtroom runner + metrics aggregation cho LJP.
- [ ] Citation validity / hallucination checker tích hợp (LegalCiteBench/LegalHalBench style).
- [ ] Human eval rubric wiring cho courtroom sessions.
- [ ] P1 Gemini validation và ablation matrix `--execute`.
- [ ] So sánh Phase 1 QA debate vs Phase 3 courtroom LJP trên cùng case adapter.

## Vấn Đề Đã Biết
- P1 local debate trên ViLQA có xu hướng trả cả câu điều luật hoặc paraphrase tiếng Anh thay vì span ngắn; prompt và `shorten_legal_answer()` đã được siết nhưng cần re-run validation.
- `load_simucourt` / `load_vlegal` phụ thuộc HF dataset schema; có thể cần chỉnh field mapping sau khi tải thật.
- `MockLLM` không phản ánh chất lượng LJP thực tế.
- Courtroom session gọi cả `render_ljp_verdict` (structured) và `render_verdict` (QA-style) — verdict QA là backward-compat view.
- Dataset loaders SimuCourt dùng `law-ai/SimuCourt` — cần xác nhận repo HF chính xác khi chạy production.
