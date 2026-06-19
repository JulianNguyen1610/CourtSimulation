# Error Analysis Report

**Run:** `outputs\vilqa_multi_agent_baseline\20260617T045228Z_validation_both`
**Total predictions:** 106
**Methods:** debate, direct

## 1. Overall Metrics by Method

| Method | N | EM rate | Avg F1 | Correct | Error rate |
|---|---|---|---|---|---|
| debate | 53 | 0.0377 | 0.1405 | 2 | 0.9623 |
| direct | 53 | 0.0377 | 0.1405 | 2 | 0.9623 |

## 2. Error Taxonomy by Method

### debate

| Error type | Count | % of errors |
|---|---|---|
| UNKNOWN_ANSWER | 29 | 56.9% |
| OTHER | 8 | 15.7% |
| WRONG_SPAN_TYPE | 7 | 13.7% |
| PARTIAL_SPAN | 7 | 13.7% |

### direct

| Error type | Count | % of errors |
|---|---|---|
| UNKNOWN_ANSWER | 29 | 56.9% |
| OTHER | 8 | 15.7% |
| WRONG_SPAN_TYPE | 7 | 13.7% |
| PARTIAL_SPAN | 7 | 13.7% |

## 3. Cross-Method Analysis (direct vs debate)

- **Total compared:** 53
- **Direct wins:** 0
- **Debate wins:** 0
- **Both correct:** 2
- **Both wrong:** 51

### Cases debate FIXED direct's errors:

### Cases debate BROKE direct's correct answers:

### Cases with same error (debate added no value):
- vilqa-112
- vilqa-285
- vilqa-352
- vilqa-327
- vilqa-138
- vilqa-470
- vilqa-447
- vilqa-125
- vilqa-482
- vilqa-273

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