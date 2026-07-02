# Active Context

## Trạng thái hiện tại (2026-07-01)

Phase 1 **paper-ready v1**: judge_mediated v1 validated trên cả val + test. Error analysis confirm v1 fix OVER_EXTRACTION (−47%).

## Config chính (frozen)

\\\yaml
method: debate
orchestrator: judge_mediated
rounds: 1
retrieval: off
memory: read_only
closing: on
\\\

- Code default: \configs/default.yaml\, \configs/ollama.yaml\, \BatchRunConfig\, CLI
- Predictions method: \debate_judge_mediated\
- Legacy repro: \--orchestrator fixed\

## Prompt v1 (2026-06-30 21:21) — VALIDATED 2026-07-01

4 file prompt patch, instruction-level constraints:

| File | Thay đổi |
|---|---|
| \configs/prompts/judge_verdict.txt\ | Length cap ≤15 từ; binary Q → \có\/\không\; number+unit chỉ trả số+đơn vị |
| \configs/prompts/judge_belief.txt\ | Length cap 15 từ; never expand shorter span |
| \configs/prompts/proponent_argument.txt\ | Shortest extractive span rule + example |
| \configs/prompts/opponent_rebuttal.txt\ | Shortest extractive span rule + example |

**Val v1:** EM 0.6792 → 0.7358 (+5.66 pp), F1 0.8640 → 0.8756 (+1.16 pp)
**Test v1:** EM 0.3962 → 0.4528 (+5.66 pp), F1 0.6915 → 0.7335 (+4.20 pp)
**Val→test gap:** −28.3 pp (giống v0) — delta generalize tốt.
**Best debate on test:** +7.5 pp vs vanilla, +13.2 pp vs fixed.

## Kiến trúc primary

\\\	ext
Judge → call_proponent / call_opponent / ask_question / request_closing / end_debate
         ↓
Proponent / Opponent → belief update
         ↓
Closing → Verdict (JSON) → postprocess
\\\

## Kết quả chính (ALQAC seed=42, qwen3.5:9b)

### Validation 53

| Method | EM | F1 | Vai trò |
|---|---:|---:|---|
| **Judge-mediated v1** | **0.7358** | 0.8756 | **Config chính (current)** |
| Vanilla r=1, retrieval=off | 0.7358 | 0.9295 | Baseline tie EM |
| **Judge-mediated v0** | 0.6792 | 0.8640 | Superseded |
| Fixed orchestrator | 0.6038 | 0.8412 | Ablation legacy |
| Finetuned reader | 0.5849 | 0.7610 | Non-LLM floor |

### Test 53 (one-shot)

| Method | EM | F1 | Fallback |
|---|---:|---:|---:|
| **Judge-mediated v1** | **0.4528** | **0.7335** | 1.89% |
| Judge-mediated v0 | 0.3962 | 0.6915 | 0.94% |
| Vanilla | 0.3774 | 0.7712 | 0 |
| Fixed debate | 0.3208 | 0.6957 | 1.9% |

**Val→test v1 gap:** 0.7358 → 0.4528 (−28.3 pp). Delta tổng quát hóa (EM gain giống nhau val/test).

## Error analysis — v0 vs v1 (test 53)

| Error Type | v0 | v1 | Delta |
|---|---:|---:|---|
| OVER_EXTRACTION | 19 (59.4%) | 10 (34.5%) | **−9 (−47%)** ✅ |
| PARTIAL_SPAN | 3 (9.4%) | 9 (31.0%) | +6 |
| OTHER | 8 (25.0%) | 8 (27.6%) | 0 |
| UNKNOWN_ANSWER | 2 (6.2%) | 2 (6.9%) | 0 |
| **Total errors** | 32 | 29 | **−3** ✅ |

**V1 impact verified:**
- OVER_EXTRACTION giảm 47% (length cap + shortest span rule hoạt động)
- PARTIAL_SPAN tăng (tradeoff: length cap aggressive → answer ngắn hơn gold)
- Net gain: +3 correct answers (+5.66 pp EM)

**Artifacts:**
- \	est_metrics/error_analysis_judge_mediated_test.md\ (v0)
- \error_analysis (2).md\ (v1)

## Bước tiếp theo

1. **Appendix case studies** (vilqa-236, vilqa-499, +1 head-to-head win)
2. **Paper tables + limitations** (val/test gap, strict EM, list answers, PARTIAL_SPAN tradeoff)
3. Phase 3 courtroom pilot LLM thật (sau khi chốt Phase 1 report)

## Vấn đề đã biết

- Val→test gap ~28 pp EM (mọi LLM method); test n=53 — V1 không cải thiện gap nhưng không tệ hơn
- **V1 tradeoff:** length cap giảm OVER_EXTRACTION nhưng tăng PARTIAL_SPAN (net positive)
- Strict EM nhạy prefix (\phải\, \kể từ\, \dùng\) — V1 prompt giảm near-miss
- Không fine-tune lại cần thiết cho orchestrator switch

## Metric tracking

| Method | Val EM | Test EM | Artifact |
|---|---:|---:|---|
| **judge_mediated v1 (current)** | **0.7358** | **0.4528** | \	est_metrics/judge_mediated_v1_test_metrics.json\ |
| judge_mediated v0 | 0.6792 | 0.3962 | \orchestrator_ablation/\, \	est_metrics/\ |
| vanilla r=1 retrieval=off | 0.7358 | 0.3774 | \erun_20260629/\, \anilla_test_metrics.json\ |
| fixed debate optimized | 0.6038 | 0.3208 | \erun_20260629/\, \debate_optimized_test_metrics.json\ |
| finetuned_reader | 0.5849 | — | \eader_metrics/\ |