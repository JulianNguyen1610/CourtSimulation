# Active Context

## Đang Làm Gì
- **Phase 1 validation qwen3.5:9b — milestone chính đạt được** trên server spark-063e.
- Run chính: `outputs/vilqa_multi_agent_baseline/20260619T212113Z_validation_both`.
- Ablation rounds 1/3/5 hoàn tất; **r=1 optimum** (EM 0.49 > r=5 0.45 > r=3 0.42).
- Error analysis hoàn tất; 4 case regression đã phân loại.
- Tiếp theo: baselines cot/vanilla/reader, ablation retrieval, test split, results-summary.

## Thay Đổi Gần Đây (2026-06-20)

### Kết quả chính — both validation 53
- direct: EM=0.2453, F1=0.6634
- debate r=1: EM=0.4906, F1=0.8124; fallback 4.72%
- So sánh công bằng (max_output_tokens=384, qwen3.5:9b)

### Ablation rounds (debate only)
| Rounds | EM | F1 |
|---:|---:|---:|
| 1 | 0.4906 | 0.8124 |
| 3 | 0.4151 | 0.7633 |
| 5 | 0.4528 | 0.8048 |

Kết luận: **1 round là optimum**; nhiều rounds gây belief drift (r=3 tệ nhất); r=5 phục hồi một phần nhưng không vượt r=1.

### Error analysis (`20260619T212113Z_validation_both`)
- Debate sửa 17 case, regression 4 case, cả hai sai 23 case.
- Lỗi chủ yếu: OVER_EXTRACTION (direct 67.5% → debate 55.6%).
- 4 regression: vilqa-236 (prefix "Sau"), vilqa-125 (span dài), vilqa-36 (over-extract hình phạt), vilqa-499 (list chỉ lấy 1 mục).

## Kết quả quan trọng cần nhớ
- Claim chính: debate r=1 > direct trên val 53 qwen3.5 (fair config).
- Không claim run max_tokens=128 hoặc dolphin3.
- Default config paper: rounds=1, bm25_only, memory=off, judge=on.

## Bước Tiếp Theo
1. Chạy cot + vanilla + reader baselines trên validation 53.
2. Ablation retrieval (off vs bm25 vs rerank) với debate r=1.
3. Viết `docs/experiments/results-summary.md`.
4. Cải thiện postprocess: prefix "Sau", list answers.
5. Test split 1 lần sau khi chốt config.
6. Phase 3 courtroom pilot qwen3.5/Gemini.
