# Progress

## Đã Hoàn Thành
- **Khởi tạo dự án**: Multi-Agent Courtroom Simulation Framework.
- **Phase 1 ViLQA scaffold**: data loader, split, `DebateOrchestrator`, baselines, retrieval, memory, evaluator EM/F1.
- **P0/P1 pipeline**: LLM providers, judge fallback, semantic rerank, memory ablation, debate loop, ablation matrix script.
- **Tài liệu dự án (2026-06-17)**:
 - `docs/project-status-and-overview.md` — tình trạng, kiến trúc, kết quả thí nghiệm.
 - `docs/advanced-techniques-analysis.md` — phân tích kỹ thuật từ 6 papers.
 - `docs/implementation-checklist.md` — checklist triển khai chi tiết (~144 mục).
- **Phase 3 courtroom scaffold (items 10–13)**:
 - Agents theo vai trò pháp lý: prosecutor, defense, defendant, judge LJP.
 - Courtroom protocol 3 giai đoạn: opening → debate → judgment (closing + deliberation + ruling).
 - `CourtCase` schema + loaders + pilot case VN.
 - `LJPEvaluator` cho charge/article/sentence metrics.
 - Backward compat Phase 1 qua `DebateAgent` + `compat.py`.
- **Tooling (2026-06-20)**:
 - `scripts/error_analysis.py` — taxonomy lỗi legal QA debate, cross-method direct vs debate.
 - `scripts/setup_server.sh` — verify GPU/Ollama/context/deps trên Linux server.
- **P1 validation qwen3.5:9b (server spark-063e, 2026-06-19/26)**:
 - Run chính `both` validation 53: structured debate r=1 > direct (fair config).
 - Ablation rounds 1/3/5 hoàn tất: **r=1 optimum**.
 - **Toàn bộ 6 baselines validation 53 hoàn tất** (direct, cot, vanilla, structured debate r=1/3/5, extractive_qa, bm25_reader).
 - **Ablation matrix 6 variants** hoàn tất (retrieval_off, bm25_rerank, memory_read_only/update, closing_off, judge_question_on).
 - Error analysis trên run `20260619T212113Z_validation_both`.
 - Phân tích 4 case debate regression (prefix / over-extract / partial list).
 - Python 3.10 compatibility fix (`datetime.UTC` → `timezone.utc`).
- **Ablation matrix P1 scaffold (2026-06-26)**:
  - `scripts/run_ablation_matrix.py` cập nhật: 10 variants, r=1 base, closing/judge_question flags, Python 3.10 compat fix.
  - `scripts/run_p1_ablations.sh` script bash chạy tuần tự trên server.
  - Unit test smoke OK cho tất cả ablation flags.
- **B.3.9 Memory leak fix (2026-06-26)**:
  - Bug: `_append_default_memories` lưu gold answer trong `context_excerpt` / `text` mà không qua `_sanitize_entry`.
  - Fix: chạy `_sanitize_entry()` trên mọi default entry; thêm `"prediction"` vào sanitize keys.
  - 4 new unit tests `MemoryLeakPreventionTest` — 35 total tests pass.

- **P1 test split one-shot (2026-06-27)**:
 - Vanilla test 53: EM=0.3774, F1=0.7712
 - Structured optimized test 53: EM=0.3208, F1=0.6957 (retrieval=off, memory=read_only)
 - Metrics: `docs/experiments/test_metrics/`
- **Fine-tuned reader (2026-06-27, spark-063e)**:
 - Train: `checkpoints/legal_qa_reader/best_model` (XLM-R, 5 epochs, train-only)
 - Val 53: `finetuned_reader` EM=0.5849, F1=0.7610 (+22.6 pp vs extractive_qa)
 - Val 53: `tuned_bm25_reader` EM=0.5283, F1=0.7023 (BM25 −5.7 pp vs reader-only)
 - Artifacts: `docs/experiments/reader_metrics/`
- **Courtroom pilot scaffold (2026-06-27)**:
 - `save_courtroom_result()` + CLI `--save-courtroom`
 - Mock pilot `case_01_theft.json`: 10 turns, LJP metrics saved
 - Finetuned reader batch wiring (`finetuned_reader`, `tuned_bm25_reader`)
 - `src/reader/finetune_reader.py`, `scripts/train_reader.py`, `src/retrieval/tuned_rag.py`
 - 36 unit tests pass

## Kết Quả Thí Nghiệm

### Validation 53 (qwen3.5:9b, spark-063e)

| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | Vanilla debate | **0.6792** | **0.9401** |
| 2 | Structured, retrieval=off | 0.5849 | 0.8535 |
| 2 | **Finetuned reader** | **0.5849** | 0.7610 |
| 3 | Structured, memory=read_only | 0.5849 | 0.8269 |
| 4 | Structured, memory=read_update | 0.5660 | 0.8478 |
| 5 | Tuned BM25 + finetuned reader | 0.5283 | 0.7023 |
| 5 | Structured, retrieval=rerank | 0.5283 | 0.8096 |
| 6 | Structured r=1 (baseline) | 0.4906 | 0.8124 |
| 7 | CoT | 0.4717 | 0.8610 |
| 8 | Structured, closing=off | 0.4528 | 0.7806 |
| 9 | Structured r=5 | 0.4528 | 0.8048 |
| 10 | Structured r=3 | 0.4151 | 0.7633 |
| 11 | Extractive QA reader (generic) | 0.3585 | 0.6413 |
| 12 | Direct | 0.2453 | 0.6634 |
| 13 | BM25 + reader (generic) | 0.1887 | 0.4557 |

### Test 53 one-shot (2026-06-27, B.5.8)

| Method | EM | F1 | Config |
|---|---:|---:|---|
| Vanilla | 0.3774 | 0.7712 | default |
| Structured (optimized) | 0.3208 | 0.6957 | retrieval=off, memory=read_only, r=1 |

Val→test EM drop: vanilla −30.2 pp; structured −26.4 pp.

### Ablation rounds — structured debate

| Rounds | EM | F1 | Kết luận |
|---:|---:|---:|---|
| 1 | 0.4906 | 0.8124 | Optimum |
| 3 | 0.4151 | 0.7633 | Belief drift |
| 5 | 0.4528 | 0.8048 | Vẫn thua r=1 |

### Error analysis — validation (structured vs direct)

| Metric | direct | structured debate |
|---|---:|---:|
| EM | 0.2453 (13/53) | 0.4906 (26/53) |
| OVER_EXTRACTION | 67.5% | 55.6% |

**Regression cases (direct đúng → debate sai):** vilqa-236 (prefix), vilqa-125/36 (over-extract), vilqa-499 (list).

### Runs tham chiếu (không claim chính)

| Thí nghiệm | Kết quả |
|---|---|
| dolphin3 direct | EM=0.2642 |
| qwen broken max_tokens=128 | INVALID |

### Runs kỹ thuật

| Thí nghiệm | Kết quả |
|---|---|
| Unit tests | 35 pass |
| Phase 3 courtroom smoke | MockLLM OK |

## Đang Thực Hiện

- Courtroom pilot LLM thật trên server (D.3.8)

## Còn Lại

- [x] Test split one-shot (B.5.8)
- [x] Ablation matrix 6 variants + baselines val 53
- [x] `docs/experiments/results-summary.md`
- [x] Courtroom mock pilot + artifact save (D.3.9)
- [x] Fine-tune reader checkpoint + validation eval (B.4.5/B.4.6)
- [~] Phase 3 pilot LLM thật + LJP eval (D.3.8, D.4.5) — mock done
- [ ] Fix postprocess: prefix "Sau", list-answer (vilqa-499)
- [ ] Batch courtroom runner (D.6.1)
- [ ] Human eval rubric (E.2)

## Vấn Đề Đã Biết
- **Vanilla debate (0.68) vượt structured debate (0.49)**: Single-prompt self-debate hiệu quả hơn multi-turn structured debate trên model 9B.
- Strict EM nhạy prefix ("Sau 01 tháng" vs "01 tháng") — cân nhắc relaxed EM cho analysis.
- Structured debate over-extract vẫn 55.6% lỗi; direct 67.5%.
- Rounds>1 trên qwen3.5:9b không cải thiện EM; r=1 là default cho structured debate.
- Câu trả lời dạng danh sách (vilqa-499) debate chọn 1 mục thay vì full list.
- BM25 retrieval làm **giảm** QA performance (extractive_qa 0.36 → BM25+reader 0.19; finetuned 0.58 → tuned_bm25 0.53).
- **Finetuned reader** EM=0.5849 — best non-LLM baseline; +22.6 pp vs generic extractive_qa.
- dolphin3 không phù hợp debate baseline.
- **FIXED**: `_append_default_memories` leak gold answer — đã thêm `_sanitize_entry()` call.

## Metric Tracking — Bảng chính (validation 53, qwen3.5:9b fair config)

| Method | EM | F1 | Run ID |
|---:|---:|---|
| **vanilla debate** | **0.6792** | **0.9401** | server 2026-06-26 |
| **finetuned_reader** | **0.5849** | **0.7610** | server 2026-06-27 |
| structured debate r=1 | 0.4906 | 0.8124 | 20260619T212113Z_validation_both |
| tuned_bm25_reader | 0.5283 | 0.7023 | server 2026-06-27 |
| cot | 0.4717 | 0.8610 | server 2026-06-26 |
| structured debate r=5 | 0.4528 | 0.8048 | server 2026-06-19/20 |
| structured debate r=3 | 0.4151 | 0.7633 | server 2026-06-19 |
| extractive_qa | 0.3585 | 0.6413 | server 2026-06-26 |
| direct | 0.2453 | 0.6634 | 20260619T212113Z_validation_both |
| bm25_reader | 0.1887 | 0.4557 | server 2026-06-26 |

## Metric Tracking — Test 53 one-shot (2026-06-27, B.5.8)

| Method | EM | F1 | Fallback | Config |
|---:|---:|---:|---:|---|
| **vanilla** | **0.3774** | **0.7712** | 0 | default |
| structured (optimized) | 0.3208 | 0.6957 | 1.9% | retrieval=off, memory=read_only, r=1 |

Val→test EM drop: vanilla −30.2 pp; structured −26.4 pp (test harder / val-tuned config).
