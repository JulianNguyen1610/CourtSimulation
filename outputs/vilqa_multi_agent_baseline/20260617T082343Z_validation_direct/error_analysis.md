# Error Analysis Report

**Run:** `outputs\vilqa_multi_agent_baseline\20260617T082343Z_validation_direct`
**Total predictions:** 2
**Methods:** direct

## 1. Overall Metrics by Method

| Method | N | EM rate | Avg F1 | Correct | Error rate |
|---|---|---|---|---|---|
| direct | 2 | 0.0 | 0.6108 | 0 | 1.0 |

## 2. Error Taxonomy by Method

### direct

| Error type | Count | % of errors |
|---|---|---|
| OVER_EXTRACTION | 1 | 50.0% |
| OTHER | 1 | 50.0% |

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