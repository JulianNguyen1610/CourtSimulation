
---
Đã ghi báo cáo: outputs/vilqa_multi_agent_baseline/20260619T212113Z_validation_both/error_analysis.md
# Error Analysis Report

**Run:** `outputs/vilqa_multi_agent_baseline/20260619T212113Z_validation_both`
**Total predictions:** 106
**Methods:** debate, direct

## 1. Overall Metrics by Method

| Method | N | EM rate | Avg F1 | Correct | Error rate |
|---|---|---|---|---|---|
| debate | 53 | 0.4906 | 0.8124 | 26 | 0.5094 |
| direct | 53 | 0.2453 | 0.6634 | 13 | 0.7547 |

## 2. Error Taxonomy by Method

### debate

| Error type | Count | % of errors |
|---|---|---|
| OVER_EXTRACTION | 15 | 55.6% |
| OTHER | 6 | 22.2% |
| PARTIAL_SPAN | 6 | 22.2% |

### direct

| Error type | Count | % of errors |
|---|---|---|
| OVER_EXTRACTION | 27 | 67.5% |
| PARTIAL_SPAN | 8 | 20.0% |
| OTHER | 5 | 12.5% |

## 3. Cross-Method Analysis (direct vs debate)

- **Total compared:** 53
- **Direct wins:** 4
- **Debate wins:** 17
- **Both correct:** 9
- **Both wrong:** 23

### Cases debate FIXED direct's errors:
- vilqa-112
- vilqa-352
- vilqa-331
- vilqa-349
- vilqa-136
- vilqa-107
- vilqa-181
- vilqa-189
- vilqa-186
- vilqa-443

### Cases debate BROKE direct's correct answers:
- vilqa-236
- vilqa-125
- vilqa-36
- vilqa-499

### Cases with same error (debate added no value):
- vilqa-285
- vilqa-482
- vilqa-273
- vilqa-87
- vilqa-311
- vilqa-359
- vilqa-343
- vilqa-83
- vilqa-325
- vilqa-491

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
