# Results Summary — Legal QA Debate

Updated: 2026-06-29

## Validation 53 (qwen3.5:9b / fine-tuned reader, spark-063e)

| Method | EM | F1 | Fallback |
|---|---:|---:|---:|
| **vanilla r=1, retrieval=off** (paper + postprocess, rerun 2026-06-29) | **0.7358** | **0.9295** | 0 |
| vanilla r=3, bm25_only (prior SOTA) | 0.6792 | 0.9401 | 0 |
| structured, retrieval=off (rerun 2026-06-29) | 0.6038 | 0.8412 | 5.7% |
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

**Takeaways:** Fine-tuning là **non-LLM baseline mạnh nhất** trên val 53 (EM bằng structured+retrieval_off, F1 thấp hơn). BM25 vẫn làm giảm EM (−5.7 pp vs reader-only), cùng pattern ablation retrieval trên debate.

## Test 53 — B.5.8 one-shot (2026-06-27)

| Method | Config | EM | F1 | Fallback | Hits |
|---|---|---:|---:|---:|---:|
| **vanilla** | default | **0.3774** | **0.7712** | 0 | 20/53 |
| structured (optimized) | retrieval=off, memory=read_only, r=1 | 0.3208 | 0.6957 | 1.9% | 17/53 |

**Val → test gap:**

| Method | Val EM | Test EM | Δ EM | Val F1 | Test F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla (paper config val) | 0.7358 | 0.3774 | −35.8 pp | 0.9295 | 0.7712 | −15.8 pp |
| vanilla (prior r=3 bm25 val) | 0.6792 | 0.3774 | −30.2 pp | 0.9401 | 0.7712 | −16.9 pp |
| structured (optimized) | 0.6038 | 0.3208 | −28.3 pp | 0.8412* | 0.6957 | −14.6 pp |

*Structured val F1 from retrieval_off + memory_read_only rerun (0.8412); prior retrieval_off run was 0.8535.

## Interpretation

1. **Vanilla paper config (r=1, retrieval=off + postprocess) đạt EM=0.7358 val** — vượt prior SOTA r=3 bm25 (+5.7 pp).
2. **Vanilla vẫn tốt nhất trên test** (EM 0.38 vs 0.32 structured) — khớp hướng validation.
3. **Fine-tuned reader** đạt EM=0.5849 trên val — vượt generic reader (+22.6 pp), nhưng thấp hơn cả structured rerun (0.6038) và vanilla (0.7358).
4. **BM25 + reader** vẫn giảm performance kể cả với fine-tuned checkpoint.
5. **Cả hai LLM method giảm mạnh trên test** (~30–34 pp EM với paper config) — không tune trên test.
6. **F1 giảm ít hơn EM** — nhiều lỗi partial overlap (prefix/list answers).

## Paper config (frozen before test)

```yaml
# Primary claim method
method: vanilla
rounds: 1

# Secondary — structured optimized (ablation winner on val)
method: debate
retrieval: off
memory: read_only
rounds: 1
closing: on

# Best non-LLM reader baseline (reporting only; not used for test one-shot)
method: finetuned_reader
checkpoint: checkpoints/legal_qa_reader/best_model
```

## Artifacts

- `docs/experiments/test_metrics/vanilla_test_metrics.json`
- `docs/experiments/test_metrics/debate_optimized_test_metrics.json`
- `docs/experiments/reader_metrics/finetuned_reader_val_metrics.json`
- `docs/experiments/reader_metrics/tuned_bm25_reader_val_metrics.json`
- Full history: `docs/experiments/p1_ablation_summary.csv`
- Validation error analysis (rerun 2026-06-29): `docs/experiments/error_analysis_val_rerun_20260629.md`

## Next Steps

- Phase 3 courtroom pilot LLM thật (D.3.8)
- Optional: finetuned_reader test split one-shot (sau khi chốt — không tune trên test)
- Human eval subset (E.2)
