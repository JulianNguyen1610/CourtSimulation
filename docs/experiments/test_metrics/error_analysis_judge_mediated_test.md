# Test Error Analysis — Judge-Mediated (one-shot 2026-06-30)

**Split:** test 53 | **Config:** primary (`judge_mediated`, r=1, retrieval=off, memory=read_only)

## Metrics

| Metric | Value |
|---|---:|
| EM | **0.3962** (21/53) |
| F1 | 0.6915 |
| Fallback | 0.94% |

## vs prior test one-shot

| Method | EM | F1 |
|---|---:|---:|
| **Judge-mediated** | **0.3962** | 0.6915 |
| Vanilla | 0.3774 | 0.7712 |
| Fixed debate | 0.3208 | 0.6957 |

## Error taxonomy (32 errors)

| Type | Count | % errors |
|---|---:|---:|
| OVER_EXTRACTION | 19 | 59.4% |
| OTHER | 8 | 25.0% |
| PARTIAL_SPAN | 3 | 9.4% |
| UNKNOWN_ANSWER | 2 | 6.2% |

## Near-miss (F1 ≥ 0.8, EM=0) — 10 cases

Chủ yếu **thiếu/thừa tiền tố ngắn** (`phải`, `dùng`, `kể từ`, `theo`) — postprocess val có thể không cover hết test:

| Case | F1 | Pattern |
|---|---:|---|
| vilqa-235 | 0.97 | thiếu `phải` |
| vilqa-223, vilqa-228 | 0.95 | thiếu `phải` |
| vilqa-459 | 0.95 | thêm `theo` |
| vilqa-238, vilqa-216 | 0.94 | thêm `kể từ` |
| vilqa-47 | 0.94 | thiếu `dùng` |
| vilqa-373 | 0.80 | kéo dài list (over-extract) |

Ước tính: ~6–8 case có thể lên EM nếu mở rộng postprocess **chỉ trên validation** rồi re-test (không tune trên test).

## Hard errors (F1 < 0.3) — 12 cases

| Case | Vấn đề |
|---|---|
| vilqa-63, vilqa-110, vilqa-274 | Câu hỏi có/không — trả ngược hoặc giải thích dài |
| vilqa-135, vilqa-250 | `Không xác định` |
| vilqa-357 | Nhầm loại span (tiền vs năm tù) |
| vilqa-49 | JSON/artifact leak trong prediction |
| vilqa-524, vilqa-225 | Sai đoạn context hoàn toàn |

## Kết luận

1. Test EM **không tệ hơn** baseline cũ; **tốt nhất trong các debate** trên test.
2. Val→test gap (~28 pp) giống pattern dự án — không chứng minh judge-mediated overfit val.
3. Lỗi chính vẫn **OVER_EXTRACTION** (59%) — debate + judge chưa giải hết trên test.
4. **Không cần fine-tune**; cải thiện tiềm năng: postprocess prefix rules (validate trước), prompt judge verdict ngắn hơn.

Artifacts: `predictions.csv`, `judge_mediated_test_metrics.json`
