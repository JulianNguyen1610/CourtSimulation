# Active Context

## Đang Làm Gì
- **Structured optimized re-val** (retrieval=off, memory=read_only, r=1) sau postprocess
- Xem `docs/experiments/phase1-completion-plan.md`

## Kết quả chính (ALQAC split seed=42)

### Validation 53

| Method | EM | F1 |
|---|---:|---:|
| **Vanilla r=1, retrieval=off** (paper + postprocess) | **0.7170** | **0.9142** |
| Vanilla r=3, bm25_only (prior SOTA) | 0.6792 | 0.9401 |
| Structured, retrieval=off (pre-postprocess) | 0.5849 | 0.8535 |
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

# Secondary — đang re-val
method: debate
retrieval: off
memory: read_only
rounds: 1
closing: on
```

## Bước Tiếp Theo (Phase 1)
1. **Chạy structured optimized re-val** trên server (lệnh trong phase1-completion-plan.md)
2. Error analysis structured vs vanilla (cùng postprocess)
3. Error analysis **test** split
4. Appendix case studies + limitations
