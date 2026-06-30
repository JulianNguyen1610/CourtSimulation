# Error Analysis Report

**Run:** `docs\experiments\orchestrator_ablation\20260629T123354Z\orchestrator_judge_mediated`
**Total predictions:** 53
**Methods:** debate_judge_mediated

## 1. Overall Metrics by Method

| Method | N | EM rate | Avg F1 | Correct | Error rate |
|---|---|---|---|---|---|
| debate_judge_mediated | 53 | 0.6792 | 0.864 | 36 | 0.3208 |

## 2. Error Taxonomy by Method

### debate_judge_mediated

| Error type | Count | % of errors |
|---|---|---|
| PARTIAL_SPAN | 8 | 47.1% |
| OVER_EXTRACTION | 7 | 41.2% |
| OTHER | 2 | 11.8% |

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