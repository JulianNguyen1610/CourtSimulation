# Results Summary — Legal QA Debate

Updated: 2026-06-27

## Validation 53 (qwen3.5:9b, spark-063e)

| Method | EM | F1 | Fallback |
|---|---:|---:|---:|
| **vanilla debate** | **0.6792** | **0.9401** | 0 |
| structured, retrieval=off | 0.5849 | 0.8535 | 1.9% |
| structured, memory=read_only | 0.5849 | 0.8269 | 5.7% |
| structured r=1 (baseline) | 0.4906 | 0.8124 | 4.7% |
| CoT | 0.4717 | 0.8610 | 0 |
| direct | 0.2453 | 0.6634 | 0 |
| bm25+reader | 0.1887 | 0.4557 | 0 |

## Test 53 — B.5.8 one-shot (2026-06-27)

| Method | Config | EM | F1 | Fallback | Hits |
|---|---|---:|---:|---:|---:|
| **vanilla** | default | **0.3774** | **0.7712** | 0 | 20/53 |
| structured (optimized) | retrieval=off, memory=read_only, r=1 | 0.3208 | 0.6957 | 1.9% | 17/53 |

**Val → test gap:**

| Method | Val EM | Test EM | Δ EM | Val F1 | Test F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.6792 | 0.3774 | −30.2 pp | 0.9401 | 0.7712 | −16.9 pp |
| structured (optimized) | 0.5849 | 0.3208 | −26.4 pp | 0.8535* | 0.6957 | −15.8 pp |

*Structured val F1 from retrieval_off run (0.8535); memory_read_only val F1 was 0.8269.

## Interpretation

1. **Vanilla vẫn tốt nhất trên test** (EM 0.38 vs 0.32 structured) — khớp hướng validation.
2. **Cả hai method giảm mạnh trên test** (~26–30 pp EM) — test split khó hơn val, hoặc config ablation được chọn trên validation.
3. **F1 giảm ít hơn EM** — nhiều lỗi partial overlap (prefix/over-extract), không phải hoàn toàn sai.
4. **Không claim SOTA trên test** mà không báo cáo val/test gap và không tune thêm trên test.

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
```

## Artifacts

- `docs/experiments/test_metrics/vanilla_test_metrics.json`
- `docs/experiments/test_metrics/debate_optimized_test_metrics.json`
- Full history: `docs/experiments/p1_ablation_summary.csv`

## Next Steps

- Phase 3 courtroom pilot (D.3.8)
- Optional: error analysis test split nếu có `predictions.csv`
- Human eval subset (E.2)
