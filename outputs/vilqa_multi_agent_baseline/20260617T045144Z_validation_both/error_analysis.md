# Error Analysis Report

**Run:** `outputs\vilqa_multi_agent_baseline\20260617T045144Z_validation_both`
**Total predictions:** 10
**Methods:** debate, direct

## 1. Overall Metrics by Method

| Method | N | EM rate | Avg F1 | Correct | Error rate |
|---|---|---|---|---|---|
| debate | 5 | 0.0 | 0.125 | 0 | 1.0 |
| direct | 5 | 0.0 | 0.125 | 0 | 1.0 |

## 2. Error Taxonomy by Method

### debate

| Error type | Count | % of errors |
|---|---|---|
| PARTIAL_SPAN | 2 | 40.0% |
| UNKNOWN_ANSWER | 1 | 20.0% |
| OTHER | 1 | 20.0% |
| WRONG_SPAN_TYPE | 1 | 20.0% |

### direct

| Error type | Count | % of errors |
|---|---|---|
| PARTIAL_SPAN | 2 | 40.0% |
| UNKNOWN_ANSWER | 1 | 20.0% |
| OTHER | 1 | 20.0% |
| WRONG_SPAN_TYPE | 1 | 20.0% |

## 3. Cross-Method Analysis (direct vs debate)

- **Total compared:** 5
- **Direct wins:** 0
- **Debate wins:** 0
- **Both correct:** 0
- **Both wrong:** 5

### Cases debate FIXED direct's errors:

### Cases debate BROKE direct's correct answers:

### Cases with same error (debate added no value):
- vilqa-112
- vilqa-285
- vilqa-352
- vilqa-327
- vilqa-138

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