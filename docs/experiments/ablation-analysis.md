# Ablation Study Analysis — Phase 1 Structured Debate

Updated: 2026-06-27

## Goals

1. Quantify **retrieval ablation** impact (off vs BM25 vs BM25+rerank).
2. Quantify **memory ablation** impact (off/read_only/read_update).
3. Measure effect of **enabling closing** and **judge question**.
4. Set default for **paper config**: structured debate vs vanilla debate.

## Results

### Full Validation 53 (qwen3.5:9b, Ollama spark-063e)

| Variant               | EM    | F1     | Fall | Δ EM vs ref | RunID           |
|-----------------------|-------|--------|------|-------------|-----------------|
| **Reference**         | 0.4906| 0.8124 | 4.7% | —           | 20260619T212113Z|
| retrieval_off          | 0.5849 | 0.8535 | 1.9% | +9.4%       | 20260626T104936Z|
| retrieval_bm25_rerank | 0.5283 | 0.8096 | 1.9% | +3.8%       | 20260626T104936Z|
| memory_read_only      | 0.5849 | 0.8269 | 5.7% | +9.4%       | 20260626T104936Z|
| memory_read_update    | 0.5660 | 0.8478 | 3.8% | +7.5%       | 20260626T104936Z|
| **vanilla debate**     | 0.6792 | 0.9401 | 0.0% |             | server          |
| closing_off           | 0.4528 | 0.7806 | 3.8% | -3.8%       | 20260626T104936Z|
| judge_question_on     | 0.4528 | 0.8136 | 2.8% | -3.8%       | 20260626T104936Z|

*Note: Reference uses retrieval=BM25, memory=off, rounds=1.*

## Findings

### Retrieval Ablation

- **BM25 retrieval harms** structured debate: `retrieval_off` EM = **0.5849** > reference **0.4906**.
- Rerank (**BAAI/bge-m3**) improves over BM25-only (0.5283 vs 0.4906), but still worse than no retrieval.

### Memory Ablation

- **Memory helps**: `memory_read_only` matches `retrieval_off` (**0.5849** EM) without retrieval.
- `memory_update` slightly regresses (0.566) vs `memory_read_only` — memory transfer may add noise.

### Feature Ablation

- **Closing statements matter**: turning off closings EV drops (−3.8% EM), judge loses synthesis anchor.
- **Judge question neutral**: no EM gain, single trial.

## Error Analysis

`scripts/error_analysis.py` subset (vilqa-0 → vilqa-15, **N=16**) highlights:

| Case     | Gold               | Retrieval off               | BM25 (ref)          |
|----------|-------------------|----------------------------|----------------------|
| vilqa-0  | 07 năm             | 07 năm                     | 07 năm               |
| vilqa-7  | NHCSĐ áp dụng      | Chuẩn bị thi hành án       | Thi hành án         |
| vilqa-10 | Đông y, 01 tháng   | 01 tháng                    | Đông y, 1 tháng     |

- Retrieval noise: BM25 top-evidence **không match** với gold span.
- Retrieval off: **short span** dùng suffix ưu tiên.
- Memory helping: **vilqa-10** recall từ memory experience.

## Straight Delta

| Change                | EM Impact |
|-----------------------|-----------|
| BM25 → no retrieval   | +9.4%     |
| BM25 → rerank         | +3.8%     |
| Memory=off → read     | +9.4%     |

## Test Split — One-shot (2026-06-27, B.5.8)

| Method | Config | EM | F1 | Fallback |
|---|---|---:|---:|---:|
| vanilla | default | **0.3774** | **0.7712** | 0 |
| structured (optimized) | retrieval=off, memory=read_only, r=1 | 0.3208 | 0.6957 | 1.9% |

**Val → test:** vanilla EM −30.2 pp; structured EM −26.4 pp. Report both splits; do not tune on test.

Artifacts: `docs/experiments/test_metrics/`

## Recommendation

- **Default config for paper**:

```yaml
retrieval: off
memory: read_only
rounds: 1
closing: on
judge_question: off
# Fallback: vanilla debate on smaller subsets (EM=0.6792)
```

- Next:

```
1. ~~Re-run test split 1-shot~~ — DONE 2026-06-27
2. Phase 3 courtroom pilot (D.3.8)
3. Human eval subset (E.2)
```