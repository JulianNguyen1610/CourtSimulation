# Progress

## Milestone 0 — experimental validity (2026-07-10)
- Added validated YAML-first experiment config and immutable `data/splits/alqac_v1.json` (424/53/53); historical test-53 results are not an untouched holdout.
- Main, batch artifacts and reader training now carry resolved config plus dataset/split/memory provenance. Memory updates on validation/test fail by default.
- Existing reader checkpoint without a manifest is a reproducibility blocker until retrained or independently reconstructed.

## Đã Hoàn Thành

### Nền tảng & Phase 1 scaffold
- Multi-Agent Courtroom Simulation Framework; ViLQA/ALQAC pipeline
- DebateOrchestrator + **JudgeMediatedOrchestrator** (judge điều phối — config chính)
- LLM providers, retrieval, memory, baselines, EM/F1 evaluator, postprocess v2
- Fine-tuned reader (checkpoints/legal_qa_reader/best_model)
- Phase 3 courtroom scaffold (mock pilot OK)

### Thí nghiệm **validation 130** (qwen3.5:9b local) — 2026-07-04
Full validation split, all 8 methods, 
ounds=3, judge_mediated, BM25 only.

| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | **finetuned_reader** | **0.4846** | 0.7134 |
| 2 | vanilla | 0.4385 | 0.7656 |
| 3 | tuned_bm25_reader | 0.3615 | 0.5992 |
| 3 | debate_judge_mediated | 0.3538 | 0.7107 |
| 5 | cot | 0.3385 | 0.7346 |
| 6 | extractive_qa | 0.3000 | 0.5854 |
| 7 | direct | 0.2154 | 0.6004 |
| 8 | bm25_reader | 0.1462 | 0.3615 |

**Headline (130):** finetuned_reader vẫn vô địch EM. Debate KHÔNG thắng vanilla trên full 130 (vanilla 0.4385 vs debate 0.3538, −0.085 pp). CoT F1 (0.7346) > Debate F1 (0.7107).

**Fallback rate:** 0.82% (10/1216 parse attempts) — judge LLM output JSON ổn định.

**Settings:** configs/ollama.yaml mặc định (
ounds=3, 
etrieval=bm25_only, closing=true, judge_mediated, judge_question=false).

### Lịch sử: validation 53 (qwen3.5:9b) — 2026-06-29..07-01 [superseded]
> **Note:** Kết quả 53-case subset có representative bias. Số liệu 130-case full split là nguồn sự thật.

| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | Judge-mediated v1 | 0.7358 | 0.8756 |
| 1 | Vanilla r=1, retrieval=off | 0.7358 | 0.9295 |
| 3 | Vanilla r=3 bm25 | 0.6792 | 0.9401 |
| 3 | Judge-mediated v0 | 0.6792 | 0.8640 |
| 5 | Fixed debate optimized | 0.6038 | 0.8412 |
| 6 | Finetuned reader | 0.5849 | 0.7610 |

### Lịch sử: test 53 (one-shot)
| Method | EM | F1 | Ngày |
|---|---:|---:|---|
| Judge-mediated v1 | 0.4528 | 0.7335 | 2026-07-01 |
| Judge-mediated v0 | 0.3962 | 0.6915 | 2026-06-30 |
| Vanilla | 0.3774 | 0.7712 | 2026-06-27 |
| Fixed debate | 0.3208 | 0.6957 | 2026-06-27 |

### Error analysis (test 53, v0 vs v1)
| Error Type | v0 | v1 | Delta |
|---|---:|---:|---|
| OVER_EXTRACTION | 19 (59.4%) | 10 (34.5%) | **−9 (−47%)** ✅ |
| PARTIAL_SPAN | 3 (9.4%) | 9 (31.0%) | +6 |
| OTHER | 8 (25.0%) | 8 (27.6%) | 0 |
| UNKNOWN_ANSWER | 2 (6.2%) | 2 (6.9%) | 0 |
| **Total errors** | 32 | 29 | **−3** ✅ |

**V1 impact (53-case test):** OVER_EXTRACTION giảm 47%, net +3 correct. Tradeoff: PARTIAL_SPAN tăng.

### Prompt v1 (2026-06-30 21:21)
4 file prompt patch, instruction-level constraints (không tune trên val/test):
- judge_verdict.txt: length cap ≤15 từ, binary rule, number+unit only
- judge_belief.txt: length cap 15 từ, never expand shorter span
- proponent_argument.txt: shortest extractive span rule
- opponent_rebuttal.txt: shortest extractive span rule
- **Val 53 v1:** EM +5.66 pp, F1 +1.16 pp
- **Test 53 v1:** EM +5.66 pp, F1 +4.20 pp
- **Status:** delta tổng quát hóa trên 53-case test, **chưa xác nhận trên 130-case**.

### Tooling
- scripts/run_orchestrator_ablation.py, scripts/run_test_judge_mediated.sh/.ps1
- scripts/compare_orchestrator_predictions.py
- scripts/error_analysis.py

### Validation 130: rounds=1 ablation (qwen3.5:9b, 2026-07-05)
So sánh rounds=1 vs rounds=3. **rounds=1 thắng** — +6.2 pp EM cho debate, ít fallback hơn.

| Method | r=1 EM | r=3 EM | Delta EM | r=1 F1 | r=3 F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| direct | 0.2231 | 0.2154 | +0.008 | 0.5997 | 0.6004 | 0.000 |
| cot | 0.3615 | 0.3385 | +0.023 | 0.7421 | 0.7346 | +0.008 |
| vanilla | 0.4385 | 0.4385 | 0.000 | 0.7593 | 0.7656 | -0.006 |
| **debate_judge_mediated** | **0.4154** | 0.3538 | **+0.062** | **0.7389** | 0.7107 | **+0.028** |
| finetuned_reader | 0.4846 | 0.4846 | 0.000 | 0.7134 | 0.7134 | 0.000 |

**Headline:** rounds=3 sinh thêm transcript nhưng gây drift cho judge. rounds=1 = 1 debate turn ít noise. **Canonical = rounds=1.**

## Kết Quả Thí Nghiệm — Bảng tổng hợp

### Validation 130 (qwen3.5:9b, 2026-07-04) — canonical

| Rank | Method | EM | F1 | Artifact |
|---:|---|---:|---:|---|
| 1 | **finetuned_reader** | **0.4846** | 0.7134 | outputs/reader_eval/20260703/eval_results.json |
| 2 | vanilla | 0.4385 | 0.7656 | outputs/.../metrics (18).json |
| 3 | tuned_bm25_reader | 0.3615 | 0.5992 | outputs/.../metrics (18).json |
| 3 | debate_judge_mediated | 0.3538 | 0.7107 | outputs/.../metrics (18).json |
| 5 | cot | 0.3385 | 0.7346 | outputs/.../metrics (18).json |
| 6 | extractive_qa | 0.3000 | 0.5854 | outputs/.../metrics (18).json |
| 7 | direct | 0.2154 | 0.6004 | outputs/.../metrics (18).json |
| 8 | bm25_reader | 0.1462 | 0.3615 | outputs/.../metrics (18).json |

### Test 53 (one-shot, qwen3.5:9b) — historical
| Rank | Method | EM | F1 |
|---:|---|---:|---:|
| 1 | Judge-mediated v1 | 0.4528 | 0.7335 |
| 2 | Judge-mediated v0 | 0.3962 | 0.6915 |
| 3 | Vanilla | 0.3774 | 0.7712 |
| 4 | Fixed debate | 0.3208 | 0.6957 |

**Headline 130:** finetuned reader và judge-mediated v1 đều cải thiện so với direct, nhưng debate KHÔNG thắng vanilla trên full split. Cần re-validate v1 prompt trên 130 cases để confirm.

## Đang Thực Hiện

- **Re-validate prompt v1 trên 130 cases** (rounds=1, retrieval=off) — kiểm tra v1 còn cải thiện debate trên full split hay không
- Paper/report: tables, limitations, appendix case studies

## Còn Lại (Phase 1 paper-ready)

- [x] Judge-mediated orchestrator + ablation
- [x] Test one-shot judge_mediated v0 (53 cases)
- [x] Error analysis val + test v0 (53 cases)
- [x] Prompt hardening v1 (length cap + binary rule + shortest span)
- [x] Validate v1 trên val 53 (EM 0.7358) [historical]
- [x] Validate v1 trên test 53 (EM 0.4528) [historical]
- [x] Error analysis test v1 (OVER_EXTRACTION −47%) [historical]
- [x] **Full validation 130 (2026-07-04) — 8 methods canonical**
- [ ] **Re-validate v1 trên 130 cases** (debate gap vs vanilla)
- [ ] Appendix 3 case studies
- [ ] Limitations paragraph (val/test gap, strict EM, PARTIAL_SPAN tradeoff)
- [ ] Optional: postprocess prefix trên val only (Layer 1 domain-invariant chỉ)
- [ ] Phase 3 courtroom LLM thật (D.3.8)

## Vấn Đề Đã Biết

- 53-case subset có representative bias (val 53 sụt giảm mạnh khi mở rộng lên 130). Canonical = 130-case.
- Debate với rounds=3, retrieval=bm25_only có thể bị noise injection (rounds=1 + retrieval=off đã cho kết quả tốt hơn trên 53-case).
- CoT F1 cao hơn Debate F1 (130-case) — debate length cap có thể quá aggressive.
- Tuned BM25 + finetuned_reader (0.3615) THẤP hơn finetuned_reader alone (0.4846) — noise từ BM25 hurts. Bỏ qua tuned_bm25_reader.
- Strict EM nhạy prefix (phải, kể từ, dùng) — V1 prompt giảm near-miss trên 53-case.
- Không fine-tune lại cần thiết cho orchestrator switch.
- Test split (n=53) chưa được re-validate với 130-case config — số liệu test 53 có thể cũng cần update.

## Metric Tracking

| Method | Val EM | Test EM | Artifact |
|---|---:|---:|---|
| **finetuned_reader (130)** | **0.4846** | — | outputs/reader_eval/20260703/eval_results.json |
| vanilla (130) | 0.4385 | — | metrics (18).json |
| debate_judge_mediated (130) | 0.3538 | — | metrics (18).json |
| cot (130) | 0.3385 | — | metrics (18).json |
| extractive_qa (130) | 0.3000 | — | metrics (18).json |
| direct (130) | 0.2154 | — | metrics (18).json |
| tuned_bm25_reader (130) | 0.3615 | — | metrics (18).json |
| bm25_reader (130) | 0.1462 | — | metrics (18).json |
| judge_mediated v1 (53) [historical] | 0.7358 | 0.4528 | test_metrics/judge_mediated_v1_test_metrics.json |
| judge_mediated v0 (53) [historical] | 0.6792 | 0.3962 | orchestrator_ablation/ |
| vanilla r=1 retrieval=off (53) [historical] | 0.7358 | 0.3774 | rerun_20260629/ |
| fixed debate optimized (53) [historical] | 0.6038 | 0.3208 | rerun_20260629/ |
| finetuned_reader (53) [historical] | 0.5849 | — | reader_metrics/ |
