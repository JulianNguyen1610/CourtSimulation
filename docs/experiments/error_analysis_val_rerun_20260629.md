# Error Analysis — Validation Rerun (2026-06-29)

Postprocess v2 + rerun qwen3.5:9b, split validation 53 (seed=42).

## Artifacts

| Method | EM | F1 | Predictions |
|---|---:|---:|---|
| Vanilla (r=1, retrieval=off) | 0.7358 | 0.9295 | `docs/experiments/rerun_20260629/validation_vanilla/predictions.csv` |
| Structured debate (r=1, retrieval=off, memory=read_only) | 0.6038 | 0.8412 | `docs/experiments/rerun_20260629/validation_debate/predictions.csv` |

Per-method reports:
- `docs/experiments/rerun_20260629/validation_vanilla/error_analysis.md`
- `docs/experiments/rerun_20260629/validation_debate/error_analysis.md`

## Error taxonomy

| Error type | Vanilla (n=14) | Debate (n=21) |
|---|---:|---:|
| OVER_EXTRACTION | 9 (64.3%) | 11 (52.4%) |
| PARTIAL_SPAN | 1 (7.1%) | 7 (33.3%) |
| OTHER | 4 (28.6%) | 3 (14.3%) |

**Takeaway:** Postprocess v2 giảm mạnh lỗi prefix/span ngắn so với run cũ. Phần còn lại chủ yếu là **OVER_EXTRACTION** (list answers, định nghĩa dài) và **PARTIAL_SPAN** (debate chọn span ngắn hơn gold).

## Cross-method: vanilla vs structured debate

| Metric | Count |
|---|---:|
| Both correct | 28 |
| Both wrong | 10 |
| Vanilla wins (only vanilla EM=1) | 11 |
| Debate wins (only debate EM=1) | 4 |

### Debate fixed vanilla errors (4)

- `vilqa-450` — list answer (9 căn cứ quyền dân sự)
- `vilqa-136` — thiếu tail word (`liên tục`)
- `vilqa-499` — list answer (điều kiện pháp nhân)
- `vilqa-51` — định nghĩa dài (phạm tội chưa đạt)

### Debate broke vanilla correct (11)

- `vilqa-87`, `vilqa-331`, `vilqa-181`, `vilqa-189`, `vilqa-325`
- `vilqa-437`, `vilqa-98`, `vilqa-295`, `vilqa-321`, `vilqa-150`, `vilqa-282`

**Pattern:** Debate thường **over-extract hoặc partial span** trên câu ngắn (tên cơ quan, mức phạt, điều kiện ngắn) dù vanilla đã đúng.

## Remaining vanilla errors (14 cases)

| case_id | Type | F1 | Note |
|---|---|---:|---|
| vilqa-450 | OVER_EXTRACTION | 0.90 | List 9 items — limitation strict EM |
| vilqa-499 | OVER_EXTRACTION | 0.86 | List answer — limitation |
| vilqa-51 | OVER_EXTRACTION | 0.91 | Định nghĩa dài |
| vilqa-491 | OVER_EXTRACTION | 0.85 | Span dài hơn gold |
| vilqa-482 | OVER_EXTRACTION | 0.73 | List khai sinh/khai tử |
| vilqa-35 | OTHER | 0.91 | Thiếu prefix `từ` (ambiguous in context) |
| vilqa-40 | OTHER | 0.91 | Thiếu prefix `từ` |
| vilqa-136 | OTHER | 0.80 | Debate fix được; vanilla thiếu `liên tục` |
| vilqa-119 | OTHER | 0.71 | Thừa boilerplate `Di chúc có hiệu lực` |
| vilqa-116 | OVER_EXTRACTION | 0.76 | Span dài hơn gold |
| vilqa-343 | PARTIAL_SPAN | 0.44 | Chỉ lấy mức tiền, bỏ hình phạt |
| vilqa-83 | OVER_EXTRACTION | 0.36 | Over-extract nghiêm trọng |
| vilqa-395 | OVER_EXTRACTION | 0.32 | Over-extract nghiêm trọng |
| vilqa-476 | OVER_EXTRACTION | 0.83 | Span dài hơn gold |

## Suggested appendix case studies

1. **vilqa-236** — prefix `Sau` đã fix bởi postprocess (cả 2 method EM=1).
2. **vilqa-499** — list answer limitation (debate EM=1, vanilla EM=0).
3. **vilqa-331** — debate regression: vanilla đúng `15 năm`, debate partial `15`.

## Limitations (draft for paper)

- Strict EM penalizes formatting differences (prefix `từ`, list numbering) even when F1 is high.
- List/multi-span answers (vilqa-450, 482, 499) remain hard for extractive-style evaluation.
- Vanilla self-debate outperforms structured multi-agent debate on this 9B model (+13.2 pp EM).
- Validation gains do not automatically transfer to test (prior test EM ~0.38 vanilla vs ~0.32 structured).
