# Multi-Agent Courtroom Simulation — Tình Trạng & Mô Tả Dự Án

> Cập nhật: 2026-06-17  
> Liên quan: [Phân tích kỹ thuật](./advanced-techniques-analysis.md) · [Checklist triển khai](./implementation-checklist.md)

---

## 1. Tóm tắt dự án

**Multi-Agent Courtroom Simulation Framework** là dự án nghiên cứu AI/NLP tập trung vào:

- Giả lập tranh tụng pháp lý đa tác tử (Proponent/Opponent hoặc Prosecutor/Defense/Defendant/Judge).
- Dự đoán câu trả lời pháp lý ngắn (ViLQA/ALQAC extractive QA) ở Phase 1.
- Mở rộng sang **Legal Judgment Prediction (LJP)** với protocol phiên tòa có cấu trúc ở Phase 3.

Ngôn ngữ ưu tiên: **tiếng Việt** (ALQAC, UTS_VLC, pilot case hình sự VN). Hỗ trợ mở rộng EN qua SimuCourt/VLegal-Bench.

### Câu hỏi nghiên cứu chính

1. Debate đa tác tử có cải thiện độ chính xác legal QA/LJP so với single-agent (Direct, CoT) không?
2. Retrieval (BM25 + rerank) và memory 3 tầng có giúp grounding bằng chứng/pháp luật không?
3. Protocol phiên tòa có cấu trúc có tạo lập luận khách quan và giảm hallucination hơn vanilla debate không?

---

## 2. Kiến trúc tổng thể

### 2.1 Pipeline Phase 1 (ViLQA QA Debate)

```text
Input (context + question)
    ↓
Retrieve legal evidence     [BM25 ± semantic rerank ± UTS_VLC]
    ↓
Retrieve past memory        [regulations / experiences / cases]
    ↓
Debate loop (n rounds)
    Proponent → Opponent → Judge (belief update)
    [optional: judge question, early stop, closing statements]
    ↓
Judge verdict (JSON)
    ↓
Evaluator                   [EM/F1 + optional LLM rubric]
    ↓
Update memory               [reflection, mode read_update]
```

### 2.2 Pipeline Phase 3 (Courtroom LJP)

```text
CourtCase (facts, evidence, testimonies, ground_truth)
    ↓
Opening: Judge → Prosecutor indictment → Defendant testimony → Defense opening
    ↓
Debate (n rounds): Prosecutor ↔ Defense [± Judge question]
    ↓
Closing: Prosecutor → Defense
    ↓
Judgment: Judge deliberation → LJP verdict (charge/article/sentence)
    ↓
LJPEvaluator
```

### 2.3 Sơ đồ module

```mermaid
flowchart TD
    CLI[src/main.py] --> P1[BaselineBatchRunner]
    CLI --> Court[CourtroomSession]
    P1 --> Orch[DebateOrchestrator]
    Court --> Proto[CourtroomProtocol]
    Orch --> Ret[LegalRetriever + Reranker]
    Orch --> Mem[MemoryStore]
    Orch --> Agents[DebateAgent / JudgeAgent]
    Proto --> LegalAgents[Prosecutor / Defense / Defendant / Judge]
    P1 --> Eval[ViLQAEvaluator]
    Court --> LJPEval[LJPEvaluator]
```

---

## 3. Cấu trúc repository

| Thư mục / file | Vai trò |
|---|---|
| `src/main.py` | CLI: smoke test, batch P1, courtroom pilot |
| `src/models.py` | Schema Pydantic: CaseProfile, CourtCase, DebateResult, LegalJudgment, … |
| `src/data_loader.py` | ALQAC CSV, court JSON, SimuCourt, VLegal-Bench |
| `src/llm.py` | LLMClient, MockLLM, GeminiLLM, OpenAILLM, LocalLLM, factory theo role |
| `src/orchestrator.py` | DebateOrchestrator Phase 1 |
| `src/experiment_runner.py` | Batch runner, baselines, ablation config |
| `src/baselines.py` | Direct, CoT, Vanilla, extractive QA, BM25+reader |
| `src/retrieval/` | BM25 retriever, semantic reranker, UTS_VLC loader |
| `src/memory/memory_store.py` | Memory 3 tầng, reflection, dedup |
| `src/agents/` | DebateAgent, JudgeAgent, EvaluatorAgent, prosecutor/defense/defendant |
| `src/courtroom/` | Protocol + Session Phase 3 |
| `src/evaluation/` | ViLQAEvaluator, LJPEvaluator |
| `src/utils/` | answer_postprocess, prompt_compact |
| `configs/default.yaml` | Cấu hình P1: dataset, retrieval, memory, LLM roles |
| `configs/courtroom.yaml` | Cấu hình Phase 3 protocol |
| `configs/prompts/` | Prompt Phase 1 + `courtroom/` |
| `data/ALQAC.csv` | Dataset ViLQA chính Phase 1 |
| `data/processed/case_01_theft.json` | Pilot case trộm cắp VN |
| `scripts/run_ablation_matrix.py` | Sinh/chạy ma trận ablation P1 |
| `scripts/error_analysis.py` | Phân tích lỗi dự đoán |
| `tests/` | 31 unit tests (mock/offline) |
| `docs/` | Tài liệu dự án và kế hoạch thí nghiệm |
| `memory-bank/` | Ngữ cảnh dự án cho agent (brief, progress, patterns) |

---

## 4. Phases và phạm vi

| Phase | Mục tiêu | Task chính | Trạng thái |
|---|---|---|---|
| **Phase 1** | ViLQA extractive QA qua structured debate | Pipeline + baselines + ablation + test split | **Hoàn tất (M1)** |
| **Phase 2** | Memory & RAG nâng cao | UTS_VLC comparison (optional) | **Ablation done**; UTS_VLC chưa so sánh |
| **Phase 3** | Courtroom LJP simulation | Roles pháp lý, protocol 3 giai đoạn, LJP metrics | **Scaffold xong**; chưa batch runner |
| **Phase 4** | Đánh giá & an toàn | Citation check, hallucination, human eval, LegalSim red-team | **Chưa bắt đầu** |

---

## 5. Các phần đã đạt được

### 5.1 Hạ tầng & reproducibility

- [x] Loader ViLQA/ALQAC với split cố định (seed 42: train 424 / val 53 / test 53).
- [x] Schema Pydantic thống nhất cho toàn pipeline.
- [x] `CaseProfile.agent_view()` loại gold answer — chống data leakage.
- [x] LLM factory đa provider: `mock`, `gemini`, `openai`, `local` (Ollama-compatible).
- [x] Config theo role (direct, cot, proponent, opponent, judge, evaluator, prosecutor, …).
- [x] Batch artifacts: `predictions.csv`, `metrics.json`, `config.json` (không lưu API key).
- [x] **31 unit tests** pass offline (`python -m unittest discover -s tests`).

### 5.2 Phase 1 — Debate QA

- [x] `DebateAgent`: private strategy → public argument/rebuttal.
- [x] `JudgeAgent`: belief tracking, verdict JSON, fallback + retry, `fallback_rate` trong metrics.
- [x] `DebateOrchestrator`: n-round loop, closing statements, optional judge question, early stop theo confidence.
- [x] `LegalRetriever`: BM25 lexical trên train contexts.
- [x] `SemanticReranker`: BGE-m3 / lexical fallback.
- [x] UTS_VLC corpus loader (optional, `--include-uts-vlc`).
- [x] `MemoryStore`: 3 bucket, mode off/read_only/read_update, reflection prompt, dedup, embedding retrieval.
- [x] Baselines: Direct LLM, CoT, Vanilla Debate, extractive QA reader, BM25+reader.
- [x] `ViLQAEvaluator`: Exact Match + token F1.
- [x] `EvaluatorAgent`: rubric LLM-as-judge (legal_accuracy, argument_quality, logical_consistency).
- [x] `answer_postprocess.py`: rút span legal ngắn, áp dụng đồng đều mọi method.
- [x] Ablation matrix script + plan (`docs/experiments/p1_ablation_plan.md`).

### 5.3 Phase 3 — Courtroom LJP

- [x] `CourtCase` schema: facts, evidence, testimonies, judgment ground truth.
- [x] Agents: `ProsecutorAgent`, `DefenseAgent`, `DefendantAgent`, `JudgeAgent` (LJP methods).
- [x] `CourtroomProtocol`: opening → debate → closing → deliberation → ruling.
- [x] `CourtroomSession`: lifecycle đầy đủ, early stop, phase logging.
- [x] Pilot case: `data/processed/case_01_theft.json` (tội trộm cắp, BLHS VN).
- [x] `LJPEvaluator`: charge/article accuracy, sentence metrics, citation hooks.
- [x] Loaders: `load_court_case_json`, `load_simucourt`, `load_vlegal` (cần verify HF schema).
- [x] CLI: `--run-courtroom`, `--courtroom-case`, `--courtroom-config`.
- [x] Backward compat Phase 1 qua `src/agents/compat.py`.

### 5.4 Tích hợp LLM thật

- [x] Gemini API (`google-genai` SDK) — smoke test thành công với `gemini-2.5-flash`.
- [x] Local Ollama (qwen3.5:9b, dolphin3) — validation batch đã chạy.
- [x] Role-specific temperature và max_output_tokens trong YAML.

---

## 6. Kết quả thí nghiệm đã ghi nhận

| Thí nghiệm | Model | Split | EM | F1 | Ghi chú |
|---|---|---|---:|---:|---|
| **Vanilla debate** | qwen3.5:9b | val 53 | **0.679** | **0.940** | SOTA val |
| Structured optimized | qwen3.5:9b | val 53 | 0.585 | 0.854 | retrieval=off |
| Structured r=1 baseline | qwen3.5:9b | val 53 | 0.491 | 0.812 | bm25 default |
| Direct | qwen3.5:9b | val 53 | 0.245 | 0.663 | fair config |
| CoT | qwen3.5:9b | val 53 | 0.472 | 0.861 | |
| **Vanilla debate** | qwen3.5:9b | **test 53** | **0.377** | **0.771** | B.5.8 one-shot |
| Structured optimized | qwen3.5:9b | test 53 | 0.321 | 0.696 | retrieval=off, memory=RO |
| Ablation matrix | qwen3.5:9b | val 53 | — | — | 6 variants, run 20260626T104936Z |
| Courtroom smoke | MockLLM | case_01 | — | — | Phase 3 kỹ thuật OK |

**Claim hợp lệ:** vanilla > structured > direct (val); debate > direct (+100% EM val). Báo cáo val/test gap trên test (−30 pp EM vanilla).

Chi tiết: [`docs/experiments/results-summary.md`](./experiments/results-summary.md), [`p1_ablation_summary.csv`](./experiments/p1_ablation_summary.csv).

---

## 7. Đang thực hiện

- Phase 3 courtroom pilot LLM thật (D.3.8, D.4.5).
- Mapping SimuCourt / VLegal-Bench (planned).

---

## 8. Chưa hoàn thành

- [ ] Batch courtroom runner + metrics aggregation cho LJP (D.6).
- [ ] Similar-case retrieval (Courtroom-LLM style).
- [ ] Citation validity / hallucination checker (E.1).
- [ ] Human evaluation rubric (E.2).
- [ ] Case study transcript appendix (F.2.8).
- [x] Ablation matrix P1 — done 2026-06-26.
- [x] Test split one-shot — done 2026-06-27.
- [x] Báo cáo results-summary + ablation-analysis.

---

## 9. Vấn đề đã biết

| Vấn đề | Ảnh hưởng | Hướng xử lý |
|---|---|---|
| Debate trả span dài / paraphrase EN | EM/F1 thấp với model nhỏ | Prompt + `shorten_legal_answer()` (đã siết; cần re-run) |
| Direct bị cắt JSON khi `max_output_tokens` quá thấp | Direct baseline không công bằng | Bump direct/cot lên 384 tokens; JSON recovery |
| Gemini free tier quota / key leaked | API 429/403 | Key mới; dùng flash-lite hoặc local |
| SimuCourt HF schema có thể khác | Loader fail hoặc mapping sai | Verify sau download |
| MockLLM không phản ánh chất lượng LLM | Smoke only | Luôn dùng LLM thật cho metric claims |
| Courtroom dual verdict (LJP + QA view) | Confusion khi đọc output | Document rõ; QA view là backward-compat |

---

## 10. Lệnh vận hành thường dùng

### Smoke test dữ liệu

```powershell
python -m src.main --dataset data/ALQAC.csv
```

### Debate 1 case (Gemini)

```powershell
$env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')
python -m src.main --run-debate --llm gemini --gemini-model gemini-2.5-flash --case-index 0 --rounds 1
```

### Batch validation Phase 1

```powershell
python -m src.main --run-batch --split validation --method both --limit 0 --rounds 1 --llm local
```

### Courtroom pilot (mock)

```powershell
python -m src.main --run-courtroom --llm mock --courtroom-case data/processed/case_01_theft.json
```

### Unit tests

```powershell
python -m unittest discover -s tests -v
```

### Ablation dry-run

```powershell
python scripts/run_ablation_matrix.py --llm mock --limit 5
```

---

## 11. Dataset & tài nguyên liên quan

Chi tiết dataset công khai: [`docs/legal-datasets-collection.md`](./legal-datasets-collection.md).

| Dataset | Vai trò trong dự án |
|---|---|
| `data/ALQAC.csv` | Phase 1 chính — ViLQA extractive QA |
| `VietnamAIHub/UTS_VLC` | External legal corpus cho RAG |
| `data/processed/case_01_theft.json` | Pilot LJP VN |
| SimuCourt / AgentsCourt | Benchmark courtroom (planned) |
| VLegal-Bench | 22 task pháp lý VN (planned) |
| SynthLaw / MASER | Kịch bản tổng hợp (planned) |

---

## 12. Related work đã review

Các paper trong `document/` đã được tóm tắt và map vào thiết kế:

1. AgenticSimLaw — debate explainability, private/public split  
2. AgentsCourt — RAG + court simulation + Agent-as-Judge  
3. Courtroom-LLM — similar-case, structured multi-LLM  
4. AgentCourt Adversarial Evolvable — knowledge evolution 3-tier  
5. MASER — multi-agent legal interaction driver  
6. LegalSim — procedural exploit / AI safety  

Phân tích chi tiết: [`docs/advanced-techniques-analysis.md`](./advanced-techniques-analysis.md).

---

## 13. Tiêu chí hoàn thành theo phase

### Phase 1 — Done khi

- Debate vs baselines có metric validation reproducible (cùng config, cùng code eval).
- Ablation ≥ 6 biến chính đã chạy và ghi `p1_ablation_summary.csv`.
- Error analysis phân loại ≥ 80% lỗi debate.
- Test split chạy **một lần** sau khi chốt hyperparameter trên validation.

### Phase 3 — Done khi

- ≥ 1 courtroom session LLM thật hoàn chỉnh 3 giai đoạn trên pilot VN.
- LJP metrics trên pilot + ≥ 1 sample external dataset.
- Batch runner courtroom tương đương `BaselineBatchRunner`.

---

## 14. Liên kết tài liệu

| File | Nội dung |
|---|---|
| [`docs/baseline-phase1.md`](./baseline-phase1.md) | Ghi chú baseline Phase 1 ban đầu |
| [`docs/experiments/p1_ablation_plan.md`](./experiments/p1_ablation_plan.md) | Kế hoạch ablation P1 |
| [`docs/advanced-techniques-analysis.md`](./advanced-techniques-analysis.md) | Phân tích kỹ thuật từ papers |
| [`docs/implementation-checklist.md`](./implementation-checklist.md) | Checklist triển khai chi tiết |
| [`memory-bank/progress.md`](../memory-bank/progress.md) | Tiến độ cập nhật cho agent |
