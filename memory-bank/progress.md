# Progress

## Đã Hoàn Thành

### Nền tảng & Phase 1 scaffold
- Multi-Agent Courtroom Simulation Framework; ViLQA/ALQAC pipeline
- DebateOrchestrator + **JudgeMediatedOrchestrator** (judge điều phối — config chính)
- LLM providers, retrieval, memory, baselines, EM/F1 evaluator, postprocess v2
- Fine-tuned reader (checkpoints/legal_qa_reader/best_model)
- Phase 3 courtroom scaffold (mock pilot OK)

### Thí nghiệm validation 53 (qwen3.5:9b)
- Toàn bộ baselines + ablation matrix (retrieval/memory/rounds/features)
- Vanilla r=1 retrieval=off: EM 0.7358 (rerun postprocess 2026-06-29)
- Orchestrator ablation (2026-06-29): fixed 0.6038 → judge_mediated v0 0.6792 (+7.5 pp)
- **Prompt v1 (2026-06-30):** judge_mediated v0 0.6792 → **v1 0.7358** (+5.66 pp)
- **Chốt judge_mediated v1 làm config chính** (code + docs)

### Thí nghiệm test 53 (one-shot)
| Method | EM | F1 | Ngày |
|---|---:|---:|---|
| **Judge-mediated v1** | **0.4528** | **0.7335** | 2026-07-01 |
| Judge-mediated v0 | 0.3962 | 0.6915 | 2026-06-30 |
| Vanilla | 0.3774 | 0.7712 | 2026-06-27 |
| Fixed debate | 0.3208 | 0.6957 | 2026-06-27 |

**v1 delta trên test:** EM +5.66 pp, F1 +4.20 pp (giống delta val → generalize tốt).
**Val→test gap v1:** −28.3 pp (giống v0 → không overfit val).

### Error analysis
**Val v0:** structured vs direct; val rerun postprocess; orchestrator head-to-head (+4 EM)
**Test v0:** OVER_EXTRACTION 59%, 10 near-miss prefix
**Test v1 vs v0:**

| Error Type | v0 | v1 | Delta |
|---|---:|---:|---|
| OVER_EXTRACTION | 19 (59.4%) | 10 (34.5%) | **−9 (−47%)** ✅ |
| PARTIAL_SPAN | 3 (9.4%) | 9 (31.0%) | +6 |
| OTHER | 8 (25.0%) | 8 (27.6%) | 0 |
| UNKNOWN_ANSWER | 2 (6.2%) | 2 (6.9%) | 0 |
| **Total errors** | 32 | 29 | **−3** ✅ |

**V1 impact:** OVER_EXTRACTION giảm 47%, net +3 correct. Tradeoff: PARTIAL_SPAN tăng (length cap aggressive).

### Prompt v1 (2026-06-30 21:21, validated 2026-07-01)
4 file prompt patch, instruction-level constraints (không tune trên val/test):
- judge_verdict.txt: length cap ≤15 từ, binary rule, number+unit only
- judge_belief.txt: length cap 15 từ, never expand shorter span
- proponent_argument.txt: shortest extractive span rule
- opponent_rebuttal.txt: shortest extractive span rule
- **Val v1:** EM +5.66 pp, F1 +1.16 pp
- **Test v1:** EM +5.66 pp, F1 +4.20 pp
- **Δ tổng quát hóa** — không overfit val.

### Tooling
- scripts/run_orchestrator_ablation.py, scripts/run_test_judge_mediated.sh/.ps1
- scripts/compare_orchestrator_predictions.py
- scripts/error_analysis.py

## Kết Quả Thí Nghiệm — Bảng tổng hợp

### Validation 53

| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | **Judge-mediated v1** | **0.7358** | 0.8756 |
| 1 | Vanilla r=1, retrieval=off | **0.7358** | 0.9295 |
| 3 | Vanilla r=3 bm25 | 0.6792 | 0.9401 |
| 3 | Judge-mediated v0 | 0.6792 | 0.8640 |
| 5 | Fixed debate optimized | 0.6038 | 0.8412 |
| 6 | Finetuned reader | 0.5849 | 0.7610 |

### Test 53 (one-shot)

| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | **Judge-mediated v1** | **0.4528** | 0.7335 |
| 2 | Judge-mediated v0 | 0.3962 | 0.6915 |
| 3 | Vanilla | 0.3774 | 0.7712 |
| 4 | Fixed debate | 0.3208 | 0.6957 |

**Headline:** judge_mediated v1 best trên cả val lẫn test cho EM. F1 vanilla vẫn cao hơn (0.7712 vs 0.7335) do vanilla không cắt context.

## Đang Thực Hiện

- Paper/report: tables, limitations, appendix case studies

## Còn Lại (Phase 1 paper-ready)

- [x] Judge-mediated orchestrator + ablation
- [x] Test one-shot judge_mediated v0
- [x] Error analysis val + test v0
- [x] Prompt hardening v1 (length cap + binary rule + shortest span)
- [x] Validate v1 trên val 53 (EM 0.7358)
- [x] Validate v1 trên test 53 (EM 0.4528)
- [x] Error analysis test v1 (OVER_EXTRACTION −47%)
- [ ] Appendix 3 case studies
- [ ] Limitations paragraph (val/test gap ~28 pp, strict EM, PARTIAL_SPAN tradeoff)
- [ ] Optional: postprocess prefix trên val only (Layer 1 domain-invariant chỉ)
- [ ] Phase 3 courtroom LLM thật (D.3.8)

## Vấn Đề Đã Biết

- Val→test gap ~28 pp EM (mọi LLM method); test n=53
- V1 tradeoff: length cap giảm OVER_EXTRACTION (−47%) nhưng tăng PARTIAL_SPAN (+6 cases)
- Strict EM nhạy prefix (phải, kể từ, dùng) — V1 prompt giảm near-miss
- Không fine-tune lại cần thiết cho orchestrator switch

## Metric Tracking

| Method | Val EM | Test EM | Artifact |
|---|---:|---:|---|
| **judge_mediated v1 (current)** | **0.7358** | **0.4528** | 	est_metrics/judge_mediated_v1_test_metrics.json |
| judge_mediated v0 | 0.6792 | 0.3962 | orchestrator_ablation/, 	est_metrics/ |
| vanilla r=1 retrieval=off | 0.7358 | 0.3774 | erun_20260629/, anilla_test_metrics.json |
| fixed debate optimized | 0.6038 | 0.3208 | erun_20260629/, debate_optimized_test_metrics.json |
| finetuned_reader | 0.5849 | — | eader_metrics/ |