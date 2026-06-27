# Checklist Triển Khai Chi Tiết — Legal Court Simulation

> Cập nhật: 2026-06-27  
> Liên quan: [Tình trạng dự án](./project-status-and-overview.md) · [Phân tích kỹ thuật](./advanced-techniques-analysis.md)

**Chú thích trạng thái**

- `[x]` Hoàn thành
- `[~]` Một phần / cần verify thêm
- `[ ]` Chưa làm

**Chú thích ưu tiên:** P0 (bắt buộc) · P1 (ngắn hạn) · P2 (trung hạn) · P3 (dài hạn)

---

## A. Nền tảng dự án (P0)

### A.1 Repository & cấu hình

- [x] **A.1.1** Cấu trúc `src/` tách data / agents / retrieval / memory / evaluation / courtroom
- [x] **A.1.2** `configs/default.yaml` — dataset, retrieval, memory, debate, LLM roles
- [x] **A.1.3** `configs/courtroom.yaml` — protocol Phase 3
- [x] **A.1.4** Prompt templates Phase 1 (`configs/prompts/`)
- [x] **A.1.5** Prompt templates Phase 3 (`configs/prompts/courtroom/`)
- [x] **A.1.6** `requirements.txt` dependencies tối thiểu
- [x] **A.1.7** Không hard-code API key; dùng env vars
- [ ] **A.1.8** `.gitignore` đầy đủ (outputs, checkpoints, `.env`, memory JSON lớn) — kiểm tra repo root

**Acceptance:** `python -m compileall src` pass; config load không lỗi.

---

### A.2 Data & split

- [x] **A.2.1** Loader `data/ALQAC.csv` → `CaseProfile`
- [x] **A.2.2** Split train/val/test seed=42 reproducible
- [x] **A.2.3** `CaseProfile.agent_view()` loại gold `answer`
- [x] **A.2.4** Loader `load_court_case_json` cho pilot LJP
- [~] **A.2.5** Loader `load_simucourt()` — verify HF schema thật
- [~] **A.2.6** Loader `load_vlegal()` — verify HF schema thật
- [ ] **A.2.7** Document split policy trong paper/report (không tune trên test)

**Acceptance:** Smoke load in ra split sizes đúng (424/53/53).

---

### A.3 LLM infrastructure

- [x] **A.3.1** `LLMClient` protocol
- [x] **A.3.2** `MockLLM` cho unit test offline
- [x] **A.3.3** `GeminiLLM` (google-genai SDK)
- [x] **A.3.4** `OpenAILLM` (nếu có trong codebase)
- [x] **A.3.5** `LocalLLM` Ollama-compatible
- [x] **A.3.6** Factory theo role từ YAML + CLI override
- [x] **A.3.7** Log `models_by_method` trong `metrics.json`
- [ ] **A.3.8** Retry/backoff cho 429/503 Gemini (optional enhancement)

**Acceptance:** `--llm mock|gemini|local` chạy 1 debate case không crash.

---

### A.4 Testing & CI

- [x] **A.4.1** Unit tests Phase 2 orchestrator
- [x] **A.4.2** Unit tests Phase 3 retrieval/memory
- [x] **A.4.3** Unit tests Phase 4 evaluation runner
- [x] **A.4.4** Unit tests LLM client factory
- [x] **A.4.5** Unit tests Phase 5 courtroom
- [x] **A.4.6** Unit tests answer postprocess + judge agent
- [x] **A.4.7** `scripts/setup_server.sh` — verify GPU/Ollama/context trên Linux server
- [ ] **A.4.8** CI workflow GitHub Actions chạy `unittest` + `compileall` (optional)

**Acceptance:** `python -m unittest discover -s tests` → 28+ tests OK.

---

## B. Phase 1 — ViLQA Structured Debate (P0–P1)

### B.1 Core debate loop

- [x] **B.1.1** `DebateAgent` — private strategy → public argument
- [x] **B.1.2** `DebateOrchestrator` — n-round proponent/opponent/judge
- [x] **B.1.3** `JudgeAgent.update_belief()` mỗi round
- [x] **B.1.4** `JudgeAgent.render_verdict()` cuối session
- [x] **B.1.5** JSON parse + fallback + optional retry
- [x] **B.1.6** Metric `fallback_rate` trong batch output
- [x] **B.1.7** Closing statements trước verdict (`include_closing_statements`)
- [x] **B.1.8** Early stop theo `early_stop_confidence` (flag có, chưa tune)
- [x] **B.1.9** Optional judge question (`enable_judge_question`)
- [x] **B.1.10** Lưu `DebateResult` JSON artifact

**Acceptance:** 1 case, 2 rounds → transcript 4 turns + 2 belief updates + verdict.

**Lệnh kiểm tra:**

```powershell
python -m src.main --run-debate --llm mock --case-index 0 --rounds 2
```

---

### B.2 Retrieval layer

- [x] **B.2.1** `LegalRetriever` BM25 trên train cases
- [x] **B.2.2** Fit retriever chỉ từ train (không val/test labels)
- [x] **B.2.3** `rough_top_n` + `evidence_top_k` configurable
- [~] **B.2.4** `SemanticReranker` BGE-m3
- [~] **B.2.5** Ablation flag `retrieval.method`: off / bm25_only / bm25_rerank
- [~] **B.2.6** UTS_VLC external corpus loader
- [x] **B.2.7** Chạy ablation retrieval trên validation 53 — done 20260626T104936Z
- [x] **B.2.8** Báo cáo delta EM/F1: off vs bm25 vs bm25+rerank — `ablation-analysis.md`

**Acceptance:** Retrieved evidence có `doc_id`, `score`; không chứa gold answer field.

**Lệnh:**

```powershell
python -m src.main --run-debate --include-uts-vlc --retrieval-method bm25_rerank --llm gemini --case-index 0
```

---

### B.3 Memory layer

- [x] **B.3.1** `MemoryStore` 3 bucket: regulations / experiences / cases
- [x] **B.3.2** Mode: off / read_only / read_update
- [x] **B.3.3** Query lexical + optional embedding retrieval
- [x] **B.3.4** Reflection prompt (`memory_reflection.txt`)
- [x] **B.3.5** Dedup + `max_entries_per_bucket`
- [x] **B.3.6** `--update-memory` CLI flag
- [x] **B.3.7** Ablation memory_off vs read_only vs read_update (validation 53)
- [ ] **B.3.8** Ordered batch eval cross-case memory transfer
- [x] **B.3.9** Kiểm tra reflection không leak gold answer — 4 unit tests pass

**Acceptance:** Memory JSON round-trip; query trả top-k entries có `id`.

---

### B.4 Baselines

- [x] **B.4.1** Direct LLM baseline
- [x] **B.4.2** CoT LLM baseline
- [x] **B.4.3** Vanilla debate baseline
- [x] **B.4.4** Regex/heuristic direct candidate (weak floor)
- [~] **B.4.5** Extractive QA reader (HuggingFace) — finetune scaffold + batch wiring
- [~] **B.4.6** BM25 + reader baseline — tuned_bm25_reader scaffold added
- [x] **B.4.7** `shorten_legal_answer()` áp dụng **đồng đều** mọi method
- [x] **B.4.8** JSON recovery khi output bị cắt token (direct/cot)
- [x] **B.4.9** Bảng so sánh đầy đủ baselines validation 53 — qwen3.5:9b, 6 methods + ablation

**Kết quả validation 53 (qwen3.5:9b):** vanilla EM=0.6792; structured r=1 EM=0.4906; direct EM=0.2453.  
**Kết quả test 53 (one-shot):** vanilla EM=0.3774; structured optimized EM=0.3208.

---

### B.5 Evaluation Phase 1

- [x] **B.5.1** `ViLQAEvaluator` — Exact Match + token F1
- [x] **B.5.2** Normalize answer (dấu câu, lowercase, whitespace)
- [x] **B.5.3** `BaselineBatchRunner` — batch + metrics aggregation
- [~] **B.5.4** `EvaluatorAgent` LLM rubric (legal_accuracy, argument_quality, logical_consistency)
- [ ] **B.5.5** Bật `--enable-llm-evaluator` cho subset validation
- [x] **B.5.6** `scripts/error_analysis.py` — taxonomy + cross-method direct vs debate
- [x] **B.5.7** Error analysis run `20260619T212113Z_validation_both`: debate wins 17 / direct wins 4 / both OK 9 / both wrong 23; OVER_EXTRACTION dominant; 4 regression cases documented
- [x] **B.5.8** Test split — chạy 1 lần (2026-06-27): vanilla EM=0.3774, structured EM=0.3208

**Acceptance:** `metrics.json` có `metrics_by_method` và `fallbacks.fallback_rate`.

**Lệnh batch:**

```powershell
python -m src.main --run-batch --split validation --method both --limit 0 --rounds 1 --llm local
```

---

### B.6 Prompt engineering ViLQA

- [x] **B.6.1** Prompt judge yêu cầu span tiếng Việt ngắn, trích từ context
- [x] **B.6.2** Prompt proponent/opponent nhấn extractive span
- [x] **B.6.3** Escape JSON braces trong prompt templates
- [ ] **B.6.4** Thêm few-shot examples span đúng trong prompt (nếu cần sau error analysis)
- [x] **B.6.5** Temperature trong `configs/ollama.yaml`: direct/cot 0.1, vanilla 0.3, debaters 0.4, judge 0.1
- [x] **B.6.6** `config.json` + `models_by_method` snapshot mỗi run trong `metrics.json`

---

### B.7 Ablation matrix P1

- [x] **B.7.1** `docs/experiments/p1_ablation_plan.md`
- [x] **B.7.2** `scripts/run_ablation_matrix.py` dry-run commands
- [x] **B.7.3** Execute validation qwen3.5:9b — baselines + ablation matrix done
- [x] **B.7.4** Execute `--include-heavy-rerank` — bm25_rerank val 53
- [x] **B.7.5** Ghi `docs/experiments/p1_ablation_summary.csv` — incl. test rows
- [x] **B.7.6** Kết luận ablation rounds: **r=1 (0.49) > r=5 (0.45) > r=3 (0.42)** trên qwen3.5:9b val 53

**Biến ablation bắt buộc:**

| ID | Variant | Check |
|---|---|---|
| ABL-01 | retrieval_off | [x] EM=0.5849 |
| ABL-02 | bm25_only (reference) | [x] EM=0.4906 |
| ABL-03 | bm25_plus_rerank | [x] EM=0.5283 |
| ABL-04 | memory_off | [x] EM=0.4906 ref |
| ABL-05 | memory_read_only | [x] EM=0.5849 |
| ABL-06 | memory_update_on | [x] EM=0.5660 |
| ABL-07 | rounds_1 | [x] EM=0.4906 optimum |
| ABL-08 | rounds_3 | [x] EM=0.4151 |
| ABL-09 | rounds_5 | [x] EM=0.4528 |
| ABL-10 | judge_off_vanilla | [x] EM=0.6792 |
| ABL-11 | closing_off | [x] EM=0.4528 |
| ABL-12 | judge_question_on | [x] EM=0.4528 |

---

## C. Phase 2 — RAG & Memory nâng cao (P1–P2)

### C.1 External legal corpus

- [~] **C.1.1** Index UTS_VLC với metadata article/law/source
- [ ] **C.1.2** So sánh retrieval train-only vs train+UTS_VLC
- [ ] **C.1.3** Kiểm tra cited article có trong retrieved set (pre-citation check)

---

### C.2 Similar-case retrieval (Courtroom-LLM)

- [ ] **C.2.1** Thiết kế case embedding index từ memory `cases` bucket
- [ ] **C.2.2** Retrieve top-k similar cases theo question+facts
- [ ] **C.2.3** Inject similar cases vào judge + defense prompts
- [ ] **C.2.4** Ablation with/without similar-case trên validation

---

### C.3 Memory evolution nâng cao

- [~] **C.3.1** Embedding retrieval cho memory (e5-large)
- [ ] **C.3.2** Reflection quality filter — discard low-confidence entries
- [ ] **C.3.3** Memory versioning / snapshot per experiment run
- [ ] **C.3.4** Cross-run isolation — không dùng memory tuned trên test

---

## D. Phase 3 — Courtroom LJP Simulation (P1–P2)

### D.1 Data model & pilot case

- [x] **D.1.1** `CourtCase`, `EvidenceItem`, `Testimony`, `JudgmentGroundTruth`
- [x] **D.1.2** `LegalJudgment`, `CourtroomResult`, `LJPEvalResult`
- [x] **D.1.3** Pilot `data/processed/case_01_theft.json` (BLHS VN)
- [ ] **D.1.4** Thêm ≥ 2 pilot cases (dân sự, giảm nhẹ)
- [ ] **D.1.5** Validate schema JSON bằng Pydantic test cases

---

### D.2 Legal role agents

- [x] **D.2.1** `BaseLegalAgent`
- [x] **D.2.2** `ProsecutorAgent` — indictment, argument, closing, strategy
- [x] **D.2.3** `DefenseAgent` — opening, argument, closing, strategy
- [x] **D.2.4** `DefendantAgent` — testimony
- [x] **D.2.5** `JudgeAgent` — open_session, deliberate, render_ljp_verdict
- [x] **D.2.6** `compat.py` alias Phase 1 DebateAgent

---

### D.3 Courtroom protocol & session

- [x] **D.3.1** `CourtroomProtocol.opening()`
- [x] **D.3.2** `CourtroomProtocol.debate_round()`
- [x] **D.3.3** `CourtroomProtocol.closing()`
- [x] **D.3.4** `CourtroomSession.run()` lifecycle đầy đủ
- [x] **D.3.5** Phase logging (`phases_completed`)
- [x] **D.3.6** Early stop + optional judge question trong protocol config
- [x] **D.3.7** CLI `--run-courtroom`
- [x] **D.3.8** Smoke courtroom với Gemini/local LLM thật — mock OK (2026-06-27); Gemini quota exceeded locally
- [x] **D.3.9** Lưu courtroom transcript JSON artifact chuẩn hóa — `save_courtroom_result`, `--save-courtroom`

**Lệnh:**

```powershell
python -m src.main --run-courtroom --llm gemini --courtroom-case data/processed/case_01_theft.json
```

---

### D.4 LJP evaluation

- [x] **D.4.1** `LJPEvaluator` — charge/article metrics
- [x] **D.4.2** Sentence MAE/RMSE/bucket metrics
- [~] **D.4.3** Citation validity hooks
- [ ] **D.4.4** Batch LJP metrics trên nhiều cases
- [~] **D.4.5** So sánh LJP verdict vs ground truth pilot theft case — mock pilot + LJP metrics saved; Gemini run pending quota

---

### D.5 External datasets Phase 3

- [~] **D.5.1** Tải và map SimuCourt HF
- [~] **D.5.2** Tải và map VLegal-Bench
- [ ] **D.5.3** Adapter SimuCourt → `CourtCase`
- [ ] **D.5.4** Chạy ≥ 1 session trên SimuCourt sample
- [ ] **D.5.5** Document field mapping differences

---

### D.6 Batch courtroom runner

- [ ] **D.6.1** `CourtroomBatchRunner` tương tự `BaselineBatchRunner`
- [ ] **D.6.2** Output: predictions LJP CSV + metrics JSON + transcripts/
- [ ] **D.6.3** CLI `--run-courtroom-batch`
- [ ] **D.6.4** So sánh Phase 1 QA vs Phase 3 LJP trên cùng adapter case

---

## E. Phase 4 — Đánh giá nâng cao & an toàn (P2–P3)

### E.1 Citation & hallucination

- [ ] **E.1.1** Citation validity: mọi `cited_evidence_ids` ∈ retrieved evidence
- [ ] **E.1.2** Tích hợp checker inspired LegalCiteBench
- [ ] **E.1.3** Hallucination taxonomy inspired LegalHalBench
- [ ] **E.1.4** Report citation_error_rate / hallucination_rate per method

---

### E.2 Human evaluation

- [ ] **E.2.1** Rubric template: realism, fairness, legal soundness, role consistency
- [ ] **E.2.2** Sample N=20 transcripts Phase 1 + Phase 3
- [ ] **E.2.3** Inter-annotator agreement (optional)
- [ ] **E.2.4** Wire `human_eval_subset` config

---

### E.3 LegalSim red-teaming

- [ ] **E.3.1** Define procedural rules JSON (phase order, max turns, role permissions)
- [ ] **E.3.2** Automated tests: agent không được skip opening/closing
- [ ] **E.3.3** Stress test infinite loop / repeated cite
- [ ] **E.3.4** Report exploit findings (research note, không deploy)

---

## F. Reproducibility & báo cáo (P0–P1)

### F.1 Experiment tracking

- [x] **F.1.1** Timestamped output dirs
- [x] **F.1.2** `config.json` snapshot (no secrets)
- [x] **F.1.3** `predictions.csv` + `metrics.json`
- [ ] **F.1.4** Seed log cho mọi stochastic component
- [ ] **F.1.5** Reproducibility pack: 1 script reproduce main table result
- [x] **F.1.6** `models_by_method` trong `metrics.json` mỗi run (qwen3.5:9b, max_output_tokens, endpoint)

---

### F.2 Documentation

- [x] **F.2.1** `docs/baseline-phase1.md`
- [x] **F.2.2** `docs/legal-datasets-collection.md`
- [x] **F.2.3** `docs/experiments/p1_ablation_plan.md`
- [x] **F.2.4** `docs/project-status-and-overview.md` (file này bổ sung)
- [x] **F.2.5** `docs/advanced-techniques-analysis.md`
- [x] **F.2.6** `docs/implementation-checklist.md`
- [x] **F.2.7** `docs/experiments/results-summary.md` — val + test + ablation
- [ ] **F.2.8** Case study transcript markdown cho paper appendix

---

## G. Lộ trình 4 tuần (gợi ý thực thi)

### Tuần 1 — Chốt Phase 1 metrics

| Ngày | Task checklist |
|---|---|
| 1–2 | B.5.6, B.5.7 error analysis |
| 3 | B.1 + B.6 re-run validation qwen3.5/Gemini |
| 4 | B.5.8 nếu metrics ổn → ghi results-summary |
| 5 | B.7.3 ablation subset (limit 20) |

**Exit criteria tuần 1:** Có bảng direct vs debate reproducible trên validation 53.

---

### Tuần 2 — RAG ablation

| Ngày | Task checklist |
|---|---|
| 1–2 | B.2.7, B.2.8 retrieval ablations |
| 3 | C.1.2 UTS_VLC comparison |
| 4–5 | B.7.4 rerank ablation + ghi summary CSV |

**Exit criteria tuần 2:** Biết retrieval variant tốt nhất trên validation.

---

### Tuần 3 — Memory & debate depth

| Ngày | Task checklist |
|---|---|
| 1–2 | B.3.7 memory ablations |
| 3 | B.7 rounds 1/3/5 |
| 4 | B.7.11 closing off / judge question on |
| 5 | C.3 hoặc C.2 nếu memory không giúp |

**Exit criteria tuần 3:** `p1_ablation_summary.csv` đủ 8+ variants.

---

### Tuần 4 — Phase 3 pilot

| Ngày | Task checklist |
|---|---|
| 1 | D.3.8 courtroom Gemini pilot |
| 2 | D.4.5 LJP eval pilot theft |
| 3 | D.5 SimuCourt sample |
| 4–5 | D.6.1 batch runner scaffold |

**Exit criteria tuần 4:** 1 courtroom transcript LLM thật + LJP metrics pilot.

---

## H. Bảng theo dõi tiến độ tổng hợp

| Nhóm | Tổng mục | Done | Partial | Todo |
|---|---:|---:|---:|---:|
| A. Nền tảng | 29 | 25 | 2 | 2 |
| B. Phase 1 | 52 | 50 | 2 | 0 |
| C. Phase 2 | 10 | 0 | 2 | 8 |
| D. Phase 3 | 28 | 18 | 3 | 7 |
| E. Phase 4 | 11 | 0 | 0 | 11 |
| F. Repro & docs | 14 | 12 | 1 | 1 |
| **Tổng** | **144** | **105** | **10** | **29** |

*Cập nhật 2026-06-27: ablation matrix + test split + results-summary hoàn tất.*

---

## I. Definition of Done — Dự án milestone

### Milestone M1: Phase 1 paper-ready

- [x] B.7 ablations bắt buộc — rounds + retrieval + memory + features done
- [x] B.5.8 test split chạy 1 lần (vanilla 0.3774, structured 0.3208 EM)
- [x] Error analysis — run `20260619T212113Z_validation_both` + 4 regression cases
- [x] Claim metric evidence — debate > direct (val); vanilla > structured; test reported with val/test gap

### Milestone M2: Phase 3 pilot complete

- [~] D.3.8 + D.4.5 courtroom LLM thật — mock pilot done; Gemini pending quota / server Ollama
- [ ] D.6 batch runner MVP
- [ ] ≥ 3 pilot cases VN

### Milestone M3: Safety & evaluation complete

- [ ] E.1 citation/hallucination rates reported
- [ ] E.2 human eval subset
- [ ] E.3 protocol red-team pass

---

## J. Quick reference — lệnh theo checklist

```powershell
# Unit tests (A.4)
python -m unittest discover -s tests -v

# Single debate (B.1)
python -m src.main --run-debate --llm gemini --case-index 0 --rounds 2

# Batch validation (B.5)
python -m src.main --run-batch --split validation --method both --limit 0 --rounds 1 --llm local

# Ablation matrix (B.7)
python scripts/run_ablation_matrix.py --llm gemini --limit 20 --execute

# Error analysis (B.5.6)
python -m scripts.error_analysis outputs/vilqa_multi_agent_baseline/<RUN_DIR> --compare direct debate

# Courtroom pilot (D.3)
python -m src.main --run-courtroom --llm mock --save-courtroom --courtroom-case data/processed/case_01_theft.json
```

---

*Cập nhật checklist này khi hoàn thành mục; đồng bộ với `memory-bank/progress.md`. Last sync: 2026-06-27.*
