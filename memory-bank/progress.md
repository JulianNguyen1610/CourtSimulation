# Progress

## Đã Hoàn Thành

### Nền tảng & Phase 1 scaffold
- Multi-Agent Courtroom Simulation Framework; ViLQA/ALQAC pipeline
- `DebateOrchestrator` + **`JudgeMediatedOrchestrator`** (judge điều phối — config chính)
- LLM providers, retrieval, memory, baselines, EM/F1 evaluator, postprocess v2
- Fine-tuned reader (`checkpoints/legal_qa_reader/best_model`)
- Phase 3 courtroom scaffold (mock pilot OK)

### Thí nghiệm validation 53 (qwen3.5:9b)
- Toàn bộ baselines + ablation matrix (retrieval/memory/rounds/features)
- Vanilla r=1 retrieval=off: **EM 0.7358** (rerun postprocess 2026-06-29)
- Orchestrator ablation (2026-06-29): fixed 0.6038 → **judge_mediated 0.6792** (+7.5 pp)
- **Chốt judge_mediated làm config chính** (code + docs)

### Thí nghiệm test 53 (one-shot)
| Method | EM | F1 | Ngày |
|---|---:|---:|---|
| **Judge-mediated** (primary) | **0.3962** | 0.6915 | 2026-06-30 |
| Vanilla | 0.3774 | 0.7712 | 2026-06-27 |
| Fixed debate | 0.3208 | 0.6957 | 2026-06-27 |

### Error analysis
- Val structured vs direct; val rerun postprocess
- **Orchestrator head-to-head** (val): 6 fixes, 2 regressions, net +4 EM
- **Test judge_mediated**: OVER_EXTRACTION 59%, 10 near-miss prefix

### Tooling
- `scripts/run_orchestrator_ablation.py`, `scripts/run_test_judge_mediated.sh/.ps1`
- `scripts/compare_orchestrator_predictions.py`
- `scripts/error_analysis.py`

## Kết Quả Thí Nghiệm — Bảng tổng hợp

### Validation 53

| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | Vanilla r=1, retrieval=off | **0.7358** | **0.9295** |
| 2 | **Judge-mediated (primary)** | **0.6792** | **0.8640** |
| 3 | Vanilla r=3 bm25 | 0.6792 | 0.9401 |
| 4 | Fixed debate optimized | 0.6038 | 0.8412 |
| 5 | Finetuned reader | 0.5849 | 0.7610 |

### Test 53 (one-shot)

| Method | EM | F1 |
|---|---:|---:|
| **Judge-mediated** | **0.3962** | 0.6915 |
| Vanilla | 0.3774 | 0.7712 |
| Fixed debate | 0.3208 | 0.6957 |

## Đang Thực Hiện

- Paper/report: tables, limitations, appendix case studies

## Còn Lại (Phase 1 paper-ready)

- [x] Judge-mediated orchestrator + ablation
- [x] Test one-shot judge_mediated
- [x] Error analysis val + test (primary method)
- [ ] Appendix 3 case studies
- [ ] Limitations paragraph (val/test gap ~28 pp, strict EM)
- [ ] Optional: postprocess prefix trên val only
- [ ] Phase 3 courtroom LLM thật (D.3.8)

## Vấn Đề Đã Biết

- Val→test gap ~28–36 pp EM (mọi LLM method); test n=53
- Judge-mediated: OVER_EXTRACTION vẫn dominant (val 41% errors, test 59%)
- Vanilla EM val cao hơn judge-mediated nhưng không có judge coordinator
- Strict EM nhạy prefix (`phải`, `kể từ`, `dùng`) — ~10 near-miss test
- Câu có/không (`vilqa-63`, `110`, `274`) — semantic inversion
- Không fine-tune lại cần thiết cho orchestrator switch

## Metric Tracking

| Method | Val EM | Test EM | Artifact |
|---|---:|---:|---|
| **judge_mediated (primary)** | **0.6792** | **0.3962** | `orchestrator_ablation/`, `test_metrics/judge_mediated_test_metrics.json` |
| vanilla r=1 retrieval=off | 0.7358 | 0.3774 | `rerun_20260629/`, `vanilla_test_metrics.json` |
| fixed debate optimized | 0.6038 | 0.3208 | `rerun_20260629/`, `debate_optimized_test_metrics.json` |
| finetuned_reader | 0.5849 | — | `reader_metrics/` |
