# Active Context

## Quyết định kiến trúc (chốt 2026-06-29)

**Config chính dự án** = judge-mediated debate (`orchestrator: judge_mediated`).

```yaml
method: debate
orchestrator: judge_mediated
rounds: 1
retrieval: off
memory: read_only
closing: on
```

- Default: `configs/default.yaml`, `configs/ollama.yaml`, `BatchRunConfig`, CLI `--orchestrator` fallback
- Method log: `debate_judge_mediated` trong predictions
- `fixed` orchestrator giữ cho legacy ablation repro (`--orchestrator fixed`)

## Kiến trúc debate (primary)

```text
Judge → call_proponent / call_opponent / ask_question / request_closing / end_debate
         ↓
Proponent / Opponent → belief update sau mỗi round
         ↓
Closing (optional) → Verdict
```

## Kết quả validation 53 (ALQAC seed=42)

| Method | EM | F1 | Vai trò |
|---|---:|---:|---|
| Vanilla r=1, retrieval=off | 0.7358 | 0.9295 | Baseline so sánh (single-agent self-debate) |
| **Judge-mediated debate** (primary) | **0.6792** | **0.8640** | **Config chính dự án** |
| Fixed orchestrator debate | 0.6038 | 0.8412 | Legacy ablation baseline |
| Finetuned reader | 0.5849 | 0.7610 | Non-LLM floor |

Orchestrator ablation run: `outputs/orchestrator_ablation/20260629T123354Z/`

### Test 53 (one-shot 2026-06-27, fixed orchestrator only)

| Method | EM | F1 |
|---|---:|---:|
| Vanilla | 0.3774 | 0.7712 |
| Structured fixed | 0.3208 | 0.6957 |

**Chưa chạy test** cho judge_mediated.

## Đang làm / tiếp theo

1. ~~Error analysis fixed vs judge_mediated~~ → `docs/experiments/orchestrator_ablation/20260629T123354Z/error_analysis_head_to_head.md`
2. **Test one-shot** judge_mediated
3. Appendix case studies + sync `phase1-completion-plan.md`
