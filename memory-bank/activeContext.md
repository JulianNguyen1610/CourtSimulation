# Active Context

## Trạng thái hiện tại (2026-06-30)

Phase 1 **gần paper-ready**: config chính judge-mediated đã chốt; validation + test one-shot + error analysis orchestrator hoàn tất. Docs/memory bank đồng bộ.

## Config chính (frozen)

```yaml
method: debate
orchestrator: judge_mediated
rounds: 1
retrieval: off
memory: read_only
closing: on
```

- Code default: `configs/default.yaml`, `configs/ollama.yaml`, `BatchRunConfig`, CLI
- Predictions method: `debate_judge_mediated`
- Legacy repro: `--orchestrator fixed`

## Kiến trúc primary

```text
Judge → call_proponent / call_opponent / ask_question / request_closing / end_debate
         ↓
Proponent / Opponent → belief update
         ↓
Closing → Verdict (JSON) → postprocess
```

## Kết quả chính (ALQAC seed=42, qwen3.5:9b)

### Validation 53

| Method | EM | F1 | Vai trò |
|---|---:|---:|---|
| Vanilla r=1, retrieval=off | 0.7358 | 0.9295 | Baseline so sánh EM |
| **Judge-mediated** (primary) | **0.6792** | **0.8640** | **Config chính / claim kiến trúc** |
| Fixed orchestrator | 0.6038 | 0.8412 | Ablation legacy |
| Finetuned reader | 0.5849 | 0.7610 | Non-LLM floor |

Orchestrator ablation: `docs/experiments/orchestrator_ablation/20260629T123354Z/`

### Test 53 (one-shot, frozen)

| Method | EM | F1 | Fallback |
|---|---:|---:|---:|
| **Judge-mediated** (2026-06-30) | **0.3962** | **0.6915** | 0.94% |
| Vanilla (2026-06-27) | 0.3774 | 0.7712 | 0 |
| Fixed debate (2026-06-27) | 0.3208 | 0.6957 | 1.9% |

Val→test primary: 0.6792 → 0.3962 (−28.3 pp). **Best debate EM on test** (+7.5 pp vs fixed).

## Error analysis (đã xong)

| Artifact | Nội dung |
|---|---|
| `orchestrator_ablation/.../error_analysis_head_to_head.md` | Val: fixed vs judge_mediated (+4 hits) |
| `test_metrics/error_analysis_judge_mediated_test.md` | Test: 59% OVER_EXTRACTION |
| `error_analysis_val_rerun_20260629.md` | Val rerun postprocess |

## Bước tiếp theo

1. **Appendix case studies** (vilqa-236, vilqa-499, +1 head-to-head win)
2. **Paper tables + limitations** (val/test gap, strict EM, list answers)
3. Optional: postprocess prefix rules trên **validation** (`phải`, `dùng`, `kể từ`) — không tune test
4. Phase 3 courtroom pilot LLM thật (sau khi chốt Phase 1 report)
