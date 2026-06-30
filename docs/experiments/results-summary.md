# Results Summary — Legal QA Debate

Updated: 2026-06-30

## Validation 53 (qwen3.5:9b / fine-tuned reader, spark-063e)

| Method | EM | F1 | Fallback |
|---|---:|---:|---:|
| **vanilla r=1, retrieval=off** (paper + postprocess, rerun 2026-06-29) | **0.7358** | **0.9295** | 0 |
| **debate judge_mediated** (project primary, 2026-06-29) | **0.6792** | **0.8640** | — |
| vanilla r=3, bm25_only (prior SOTA) | 0.6792 | 0.9401 | 0 |
| structured fixed orchestrator, retrieval=off (rerun 2026-06-29) | 0.6038 | 0.8412 | 5.7% |
| **finetuned_reader** | **0.5849** | 0.7610 | 0 |
| structured, memory=read_only | 0.5849 | 0.8269 | 5.7% |
| tuned_bm25_reader | 0.5283 | 0.7023 | 0 |
| structured, retrieval=rerank | 0.5283 | 0.8096 | 1.9% |
| structured r=1 (baseline) | 0.4906 | 0.8124 | 4.7% |
| CoT | 0.4717 | 0.8610 | 0 |
| extractive_qa (generic SQuAD2) | 0.3585 | 0.6413 | 0 |
| direct | 0.2453 | 0.6634 | 0 |
| bm25_reader (generic) | 0.1887 | 0.4557 | 0 |

### Fine-tuned reader (2026-06-27)

Checkpoint: `checkpoints/legal_qa_reader/best_model` (deepset/xlm-roberta-base-squad2, 5 epochs, train split only).

| Method | EM | F1 | Δ EM vs extractive_qa |
|---|---:|---:|---:|
| **finetuned_reader** | **0.5849** | **0.7610** | **+22.6 pp** |
| tuned_bm25_reader | 0.5283 | 0.7023 | +17.0 pp |
| extractive_qa (reference) | 0.3585 | 0.6413 | — |
| bm25_reader (reference) | 0.1887 | 0.4557 | — |

**Takeaways:** Fine-tuning là non-LLM baseline mạnh nhất trên val. BM25 làm giảm EM reader và debate.

## Test 53 — one-shot (frozen config)

| Method | Config | EM | F1 | Fallback | Hits |
|---|---|---:|---:|---:|---:|
| **judge_mediated** (primary, 2026-06-30) | r=1, retrieval=off, memory=RO | **0.3962** | 0.6915 | 0.94% | 21/53 |
| vanilla (2026-06-27) | default | 0.3774 | 0.7712 | 0 | 20/53 |
| fixed debate (2026-06-27) | r=1, retrieval=off, memory=RO | 0.3208 | 0.6957 | 1.9% | 17/53 |

**Val → test gap (primary):**

| Method | Val EM | Test EM | Δ EM |
|---|---:|---:|---:|
| **judge_mediated** | 0.6792 | 0.3962 | −28.3 pp |
| vanilla r=1 retrieval=off | 0.7358 | 0.3774 | −35.8 pp |
| fixed debate optimized | 0.6038 | 0.3208 | −28.3 pp |

## Interpretation

1. **Judge-mediated** = config chính (val EM 0.6792, test EM 0.3962); best debate on test (+7.5 pp vs fixed).
2. **Vanilla** = baseline EM cao nhất val (0.7358) nhưng không phản ánh judge-coordinated multi-agent.
3. Val→test gap ~28–36 pp — báo cáo trong limitations; không tune trên test.
4. Lỗi chính test: OVER_EXTRACTION (59%); nhiều near-miss do prefix span.

## Orchestrator ablation (validation 53, 2026-06-29)

| Orchestrator | EM | F1 | Δ EM vs fixed | Hits |
|---|---:|---:|---:|---:|
| fixed | 0.6038 | 0.8135 | — | 32/53 |
| **judge_mediated** | **0.6792** | **0.8640** | **+7.5 pp** | **36/53** |

Run: `outputs/orchestrator_ablation/20260629T123354Z/`

## Project config (primary, 2026-06-29)

```yaml
# Primary — judge-coordinated multi-agent debate
method: debate
orchestrator: judge_mediated
rounds: 1
retrieval: off
memory: read_only
closing: on
# val EM=0.6792, F1=0.8640; test EM=0.3962, F1=0.6915

# Baseline comparison
method: vanilla
rounds: 1
retrieval: off
# val EM=0.7358

# Legacy ablation repro
method: debate
orchestrator: fixed

# Non-LLM reader (reporting)
method: finetuned_reader
checkpoint: checkpoints/legal_qa_reader/best_model
```

## Artifacts

- `docs/experiments/test_metrics/judge_mediated_test_metrics.json`
- `docs/experiments/test_metrics/error_analysis_judge_mediated_test.md`
- `docs/experiments/test_metrics/vanilla_test_metrics.json`
- `docs/experiments/test_metrics/debate_optimized_test_metrics.json`
- `docs/experiments/reader_metrics/finetuned_reader_val_metrics.json`
- `docs/experiments/reader_metrics/tuned_bm25_reader_val_metrics.json`
- Full history: `docs/experiments/p1_ablation_summary.csv`
- Orchestrator ablation + head-to-head: `docs/experiments/orchestrator_ablation/20260629T123354Z/`
- Test judge_mediated: `docs/experiments/test_metrics/judge_mediated_test/predictions.csv`

## Next Steps

- Appendix case studies + limitations paragraph
- Phase 3 courtroom pilot LLM thật (sau Phase 1 report)
- Optional: postprocess prefix rules (validate on val only)
