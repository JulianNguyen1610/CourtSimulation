# Phase 1 Completion Plan — Paper-Ready (M1+)

Updated: 2026-06-30  
Scope: Hoàn thành tốt Phase 1 ViLQA trước khi chuyển Phase 3.

---

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Baselines val 53 (LLM + reader) | Done |
| Ablation matrix (retrieval/memory/rounds/features) | Done |
| Orchestrator ablation (fixed vs judge_mediated) | Done (+7.5 pp EM val) |
| **Judge-mediated = project primary** | **Chốt** |
| Postprocess v2 + validation rerun | Done (vanilla 0.7358, fixed 0.6038) |
| Error analysis val (orchestrator head-to-head) | Done |
| Test one-shot vanilla + fixed debate | Done (2026-06-27) |
| **Test one-shot judge_mediated** | **Done** (EM=0.3962, 2026-06-30) |
| Error analysis test (judge_mediated) | Done |
| Fine-tuned reader val 53 | Done |
| Paper tables + limitations + appendix | **In progress** (tables done; appendix + limitations pending) |

---

## Definition of Done — Phase 1

1. [x] Bảng kết quả val 53 + test 53 (frozen primary config).
2. [x] Ablation có kết luận (retrieval, memory, r=1, closing, orchestrator).
3. [x] Error analysis val + test (taxonomy thống nhất).
4. [x] Không tune trên test.
5. [x] `results-summary.md` (2026-06-30).
6. [ ] Appendix case study (2–3 cases) + limitations paragraph.

---

## Frozen primary config

```yaml
method: debate
orchestrator: judge_mediated
rounds: 1
retrieval: off
memory: read_only
closing: on
```

| Split | EM | F1 |
|---|---:|---:|
| Validation 53 | 0.6792 | 0.8640 |
| Test 53 (one-shot) | 0.3962 | 0.6915 |

---

## Còn lại (P0 report)

### Appendix case studies

1. **vilqa-236** — prefix duration (postprocess)
2. **vilqa-499** — list answer (limitation)
3. Một case judge_mediated sửa fixed (val head-to-head, e.g. vilqa-359)

### Limitations paragraph

- Val→test gap ~28 pp (primary)
- Strict EM / prefix sensitivity (~10 near-miss test)
- OVER_EXTRACTION dominant
- Vanilla EM val > judge-mediated (không claim EM SOTA)

### Optional (chỉ sau khi validate trên val)

- Postprocess: `phải`, `dùng`, `kể từ` prefix rules
- `enable_llm_evaluator` subset N=20

**Không làm:** re-test, BM25 retrieval, rounds>1, fine-tune LLM cho orchestrator.

---

## Claims an toàn (có evidence)

1. Judge-mediated debate **điều phối bởi Judge** — config chính dự án.
2. Judge-mediated **+7.5 pp EM val** vs fixed orchestrator (0.6792 vs 0.6038).
3. Judge-mediated **best debate EM on test** (0.3962 vs fixed 0.3208).
4. Retrieval BM25 **giảm** EM; memory read-only **tăng** EM structured.
5. Val→test generalization yếu (~28 pp EM primary) — limitations.
6. Vanilla EM val cao nhất (0.7358) — baseline so sánh, không phải primary architecture.

---

## Checklist nhanh

```
[x] Judge-mediated primary config
[x] Orchestrator ablation val
[x] Test one-shot judge_mediated
[x] Error analysis val + test (primary)
[ ] Appendix 3 case studies
[ ] Limitations paragraph
[ ] Final paper tables sync
```

---

## Lệnh tham chiếu

```bash
# Test one-shot (đã chạy 2026-06-30)
bash scripts/run_test_judge_mediated.sh --execute

# So sánh orchestrator predictions
python scripts/compare_orchestrator_predictions.py

# Error analysis
python scripts/error_analysis.py docs/experiments/test_metrics/judge_mediated_test
```
