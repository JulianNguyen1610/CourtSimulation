# Error Analysis Report

**Run:** `d:\Research\docs\experiments\orchestrator_ablation\20260629T123354Z\orchestrator_fixed`
**Total predictions:** 53
**Methods:** debate

## 1. Overall Metrics by Method

| Method | N | EM rate | Avg F1 | Correct | Error rate |
|---|---|---|---|---|---|
| debate | 53 | 0.6038 | 0.8135 | 32 | 0.3962 |

## 2. Error Taxonomy by Method

### debate

| Error type | Count | % of errors |
|---|---|---|
| OVER_EXTRACTION | 13 | 61.9% |
| PARTIAL_SPAN | 7 | 33.3% |
| OTHER | 1 | 4.8% |

## 4. Error Type Definitions

- **CORRECT**: Exact match = 1.0
- **UNKNOWN_ANSWER**: Model trả 'Không xác định' hoặc rỗng
- **EMPTY_PARSE**: Fallback do JSON parse fail
- **ENGLISH_PARAPHRASE**: Answer chứa từ tiếng Anh (gold luôn tiếng Việt)
- **WRONG_SPAN_TYPE**: Hỏi tiền trả thời gian, hoặc ngược lại
- **OVER_EXTRACTION**: Answer > 12 từ cho câu hỏi cần span ngắn
- **PARTIAL_SPAN**: Answer ngắn nhưng F1 thấp (sai đoạn)
- **NUMERIC_MISMATCH**: Có số nhưng không khớp loại đơn vị
- **OTHER**: Lỗi chưa phân loại