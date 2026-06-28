# Active Context

## Đang Làm Gì
- **Ưu tiên Phase 1 paper-ready** — xem `docs/experiments/phase1-completion-plan.md`
- Phase 1 core done; còn: re-val postprocess, error analysis test, vanilla retrieval=off
- Phase 3 courtroom tạm hoãn đến khi Phase 1 checklist xong

## Kết quả chính (ALQAC split seed=42)

### Validation 53

| Method | EM | F1 |
|---|---:|---:|
| Vanilla debate (qwen3.5:9b) | **0.6792** | **0.9401** |
| Structured, retrieval=off | 0.5849 | 0.8535 |
| **Finetuned reader** | **0.5849** | **0.7610** |
| Tuned BM25 + finetuned reader | 0.5283 | 0.7023 |
| Structured r=1 | 0.4906 | 0.8124 |
| Extractive QA (generic) | 0.3585 | 0.6413 |
| Direct | 0.2453 | 0.6634 |

### Test 53 (one-shot, 2026-06-27)

| Method | EM | F1 |
|---|---:|---:|
| Vanilla | **0.3774** | **0.7712** |
| Structured optimized | 0.3208 | 0.6957 |

### Fine-tuned reader (2026-06-27, spark-063e)
- Checkpoint: `checkpoints/legal_qa_reader/best_model` (XLM-R, 5 epochs)
- `finetuned_reader` val 53: EM=0.5849, F1=0.7610 (+22.6 pp EM vs generic reader)
- `tuned_bm25_reader` val 53: EM=0.5283 — BM25 vẫn giảm (−5.7 pp vs reader-only)

## Paper config (frozen trước test)

```yaml
# Primary
method: vanilla
rounds: 1

# Secondary
method: debate
retrieval: off
memory: read_only
rounds: 1
closing: on
```

## Tài liệu đã sync (2026-06-27)

| File | Nội dung |
|---|---|
| `docs/experiments/p1_ablation_summary.csv` | + finetuned_reader, tuned_bm25_reader rows |
| `docs/experiments/results-summary.md` | Reader results + interpretation |
| `docs/experiments/reader_metrics/*.json` | Raw reader val metrics |

## Bước Tiếp Theo (Phase 1)
1. Re-val vanilla + structured optimized sau fix prefix `Sau`
2. Error analysis **test** split
3. Vanilla `retrieval=off` trên val 53
4. Appendix case studies + limitations paragraph
5. *(Sau Phase 1)* Courtroom pilot D.3.8
