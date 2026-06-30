# Research Project — Agent Rules

Đây là dự án nghiên cứu **Multi-Agent Courtroom Simulation Framework** — mô phỏng phiên tòa đa tác tử để giải quyết bài toán Legal Judgment Prediction (LJP) và Legal QA bằng tiếng Việt.

## Tổng Quan Kiến Trúc

```
Phase 1: ViLQA/ALQAC Legal QA Debate (proponent ↔ opponent → judge verdict)
Phase 3: Courtroom LJP Simulation (prosecutor ↔ defense ↔ defendant → judge ruling)
```

Hai phase **backward compatible**: Phase 1 vẫn chạy qua `DebateOrchestrator`; Phase 3 chạy qua `CourtroomSession`.

## Cấu Trúc Thư Mục Quan Trọng

| Đường dẫn | Vai trò |
|-----------|---------|
| `src/agents/` | Tất cả agent: `debate_agent`, `judge_agent`, `prosecutor`, `defense`, `defendant`, `evaluator_agent` |
| `src/courtroom/` | `protocol.py` (turn order), `session.py` (lifecycle 3 giai đoạn) |
| `src/orchestrator.py` | `DebateOrchestrator` cho Phase 1 |
| `src/experiment_runner.py` | `BaselineBatchRunner` — chạy và lưu kết quả batch |
| `src/baselines.py` | `direct`, `cot`, `vanilla`, `extractive_qa`, `bm25_reader` |
| `src/llm.py` | `LLMClient` protocol, `MockLLM`, `OpenAILLM`, `GeminiLLM`, `LocalLLM` |
| `src/models.py` | Tất cả Pydantic models: `CaseProfile`, `CourtCase`, `AgentOutput`, `Verdict`, `LegalJudgment`, v.v. |
| `src/retrieval/legal_retriever.py` | BM25 + optional semantic rerank (`BAAI/bge-m3`) |
| `src/memory/memory_store.py` | Three-tier memory: `regulations`, `experiences`, `cases` |
| `src/evaluation/ljp_evaluator.py` | LJP metrics: charge/article accuracy, sentence MAE/RMSE/bucket |
| `src/utils/` | `prompt_compact.py`, `answer_postprocess.py` |
| `configs/default.yaml` | Config Phase 1 (LLM per role, retrieval, memory, debate) |
| `configs/courtroom.yaml` | Config Phase 3 (protocol phases, agent token limits) |
| `configs/prompts/` | Prompt templates Phase 1 |
| `configs/prompts/courtroom/` | Prompt templates Phase 3 |
| `data/ALQAC.csv` | Dataset Phase 1 — 530 cases, cột: `context`, `question`, `answer` |
| `data/processed/` | Courtroom JSON cases cho Phase 3 |
| `scripts/run_ablation_matrix.py` | Tạo hoặc chạy ablation matrix |
| `scripts/error_analysis.py` | Phân tích lỗi dự đoán |
| `tests/` | 28 unit tests — phải pass trước khi merge |
| `memory-bank/` | Context dự án (đọc để hiểu trạng thái hiện tại) |

## Quy Tắc Kỹ Thuật

### Python & Code Style
- Python **3.10+**: dùng `str | None`, `list[str]`, `from __future__ import annotations`.
- Dùng **Pydantic** cho tất cả data models; không dùng `dict` thuần cho dữ liệu có schema.
- Không hard-code API key — đọc từ env: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `LOCAL_LLM_API_KEY`.
- `MockLLM` chỉ dùng cho **unit test và CI offline**; kết quả nghiên cứu phải dùng LLM thật.
- Khi thêm provider LLM mới, implement `LLMClient` protocol trong `src/llm.py`.

### Prompt & Agent
- Tất cả prompt template đặt trong `configs/prompts/` (Phase 1) hoặc `configs/prompts/courtroom/` (Phase 3).
- Prompt dùng `str.format(**values)` — **không** dùng f-string trực tiếp.
- Agent output **phải** là JSON hợp lệ; luôn implement fallback parsing (`_loads_json_or_empty`, `_recover_json_field`).
- Khi JSON invalid: retry 1 lần với real LLM, ghi `fallback_count`, không raise exception.
- Không đưa `gold answer` vào `agent_view()` hay prompt — tránh label leakage.

### Config & Experiment
- Cấu hình LLM theo role trong `configs/default.yaml` (mục `llm.roles`).
- Override bằng env: `LLM_BACKEND`, `LLM_MODEL`, `LLM_TEMPERATURE`, `JUDGE_LLM_MODEL`, v.v.
- **Không** tune prompt/model/temperature trên **test split** — chỉ dùng `validation`.
- Retrieval index fit chỉ từ `train` cases.
- Báo cáo đầy đủ: model/provider/temperature theo method trong `metrics.json`.
- Theo dõi `fallback_rate`; fallback cao làm giảm giá trị kết luận về debate.

### Ollama (Local LLM)
- Model thinking (ví dụ `qwen3.5:9b`) cần `LOCAL_LLM_REASONING_EFFORT=none` để tránh content rỗng.
- Cap context window: `num_ctx=8192` trên Ollama server (không để mặc định 262144).
- `max_output_tokens` cho `direct`/`cot`: **384** (không để 128 — gây JSON bị cắt).

### Testing
- Chạy test trước khi thay đổi: `python -m unittest discover -s tests`
- Test mới cho feature mới; đặt trong `tests/test_<phase_or_feature>.py`.
- Không break backward compat Phase 1 khi thêm Phase 3 code.

## Quy Trình Chạy Experiment

### Phase 1 — ViLQA QA Debate
```powershell
# Smoke test (mock)
python -m src.main --run-batch --llm mock --method both --limit 2 --rounds 1

# Validation với Gemini
python -m src.main --run-batch --llm gemini --method both --split validation --limit 0 --rounds 1

# Validation với Ollama
python -m src.main --config configs/ollama.yaml --run-batch --llm local --local-model qwen3.5:9b --method both --split validation --limit 0 --rounds 1

# Ablation matrix (dry-run xem lệnh)
python scripts/run_ablation_matrix.py --dry-run
```

### Phase 3 — Courtroom LJP
```powershell
# Smoke test (mock)
python -m src.main --run-courtroom --llm mock

# Pilot case VN
python -m src.main --run-courtroom --courtroom-case data/processed/case_01_theft.json --llm gemini
```

## Ablation Matrix

| Biến | Giá trị |
|------|---------|
| Retrieval | `off`, `bm25_only`, `bm25_rerank` |
| Memory | `off`, `read_only`, `read_update` |
| Debate rounds | `1`, `3`, `5` |
| Judge | `off` (vanilla), `on` (debate) |
| Roles | `proponent-opponent` (P1), `prosecutor-defense` (P3) |

## Trạng Thái Hiện Tại (đọc `memory-bank/` để cập nhật)

- **Phase 1**: ✅ Hoàn thiện — DebateOrchestrator, baselines, retrieval, memory, evaluation.
- **Phase 3**: ✅ Scaffold hoàn thiện — CourtroomSession, Protocol, 4 agent roles, LJPEvaluator.
- **Kết quả tốt nhất**: `qwen3.5:9b` debate EM=0.4717/F1=0.8106 vs direct EM=0.0189/F1=0.4034.
- **Còn lại**: Batch courtroom runner, citation validity checker, Gemini ablation matrix, so sánh P1 vs P3 trên cùng case.

## Nguyên Tắc Nghiên Cứu

1. Không claim cải thiện nếu chưa có metric trên **cùng split** và **cùng evaluation code**.
2. Khi báo cáo kết quả, luôn kèm: model name, provider, temperature, split, limit, rounds.
3. LLM-as-judge rubric (legal accuracy, argument quality, logical consistency) **không** nhận gold answer.
4. `shorten_legal_answer()` áp dụng đồng đều cho tất cả methods (direct, cot, vanilla, debate) để tránh bias.
5. Đọc `memory-bank/progress.md` trước khi bắt đầu task mới để tránh làm lại việc đã xong.
