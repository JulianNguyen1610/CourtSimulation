# Active Context

## Đang Làm Gì
- **Đã triển khai judge-mediated orchestrator** (2 agent debate + 1 judge điều phối)
- **Đã hoàn tất rerun validation** + error analysis val (`docs/experiments/error_analysis_val_rerun_20260629.md`)
- Còn: ablation fixed vs judge_mediated, error analysis **test** split, appendix case studies

## Kiến trúc debate mới (2026-06-29)

```text
Judge → call_proponent / call_opponent / ask_question / request_closing / end_debate
         ↓
Proponent / Opponent tranh luận → belief update sau mỗi round đủ
         ↓
Closing (optional) → Verdict
```

- Config: `debate.orchestrator: judge_mediated` hoặc CLI `--orchestrator judge_mediated`
- Method log: `debate_judge_mediated` trong predictions khi dùng chế độ này

## Kết quả chính (ALQAC split seed=42)

### Validation 53

| Method | EM | F1 |
|---|---:|---:|
| **Vanilla r=1, retrieval=off** (paper + postprocess, rerun 2026-06-29) | **0.7358** | **0.9295** |
| Vanilla r=3, bm25_only (prior SOTA) | 0.6792 | 0.9401 |
| Structured, retrieval=off + memory=read_only (rerun 2026-06-29) | 0.6038 | 0.8412 |
| **Finetuned reader** | **0.5849** | **0.7610** |
| Structured r=1 (bm25) | 0.4906 | 0.8124 |

### Test 53 (one-shot, 2026-06-27)

| Method | EM | F1 |
|---|---:|---:|
| Vanilla | **0.3774** | **0.7712** |
| Structured optimized | 0.3208 | 0.6957 |

## Paper config (frozen)

```yaml
# Primary
method: vanilla
rounds: 1
retrieval: off

# Secondary
method: debate
retrieval: off
memory: read_only
rounds: 1
closing: on
```

## Bước Tiếp Theo (Phase 1)
1. Chạy ablation **fixed vs judge_mediated**: `bash scripts/run_orchestrator_ablation.sh --execute` (plan: `docs/experiments/orchestrator_ablation_plan.md`)
2. Error analysis **test** split
2. Appendix case studies (vilqa-236, vilqa-499, vilqa-331)
3. Sync `phase1-completion-plan.md` checklist
