# Phase 1 Completion Plan — Paper-Ready (M1+)

Updated: 2026-06-27  
Scope: Hoàn thành tốt Phase 1 ViLQA trước khi chuyển Phase 3.

---

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Baselines val 53 (LLM + reader) | Done |
| Ablation matrix (retrieval/memory/rounds/features) | Done |
| Test one-shot (vanilla + structured optimized) | Done |
| Fine-tuned reader val 53 | Done |
| Error analysis val (structured vs direct) | Done |
| Error analysis **test** | **Todo** |
| Postprocess prefix/list fixes | **In progress** |
| Vanilla + retrieval=off ablation | **Todo** |
| Paper tables + limitations section | **Partial** |

---

## Definition of Done — Phase 1

1. Bảng kết quả **đầy đủ** val 53 (mọi method chính) + test 53 (frozen config).
2. Ablation **có kết luận** (retrieval off, memory, r=1, closing).
3. Error analysis **cả val và test** với taxonomy thống nhất.
4. Không tune trên test; mọi cải tiến postprocess/prompt verify trên **validation** trước.
5. `results-summary.md` + `p1_ablation_summary.csv` + appendix case study (2–3 cases).

---

## Tuần 1 — Chốt chất lượng metric (P0)

### 1.1 Postprocess & evaluation fixes

- [x] Strip prefix `Sau` khi span còn lại có trong context (`answer_postprocess.py`)
- [ ] Re-run **validation** vanilla + structured optimized sau fix; ghi delta EM/F1
- [ ] List-answer (vilqa-499): **không** sửa metric chính — báo cáo riêng `LIST_ANSWER` trong error analysis
- [ ] Optional: `relaxed_em` trong error report (prefix-insensitive) — analysis only

**Lệnh re-val (server):**

```bash
python -m src.main --config configs/ollama.yaml --run-batch \
  --split validation --method vanilla --retrieval-method off --rounds 1 \
  --llm local --local-model qwen3.5:9b --limit 0

python -m src.main --config configs/ollama.yaml --run-batch \
  --split validation --method debate --retrieval-method off \
  --memory-mode read_only --rounds 1 --llm local --local-model qwen3.5:9b --limit 0
```

### 1.2 Error analysis test split

Cần `predictions.csv` từ test runs (đã có metrics JSON).

```bash
python -m scripts.error_analysis outputs/.../<vanilla_test_run> --compare vanilla
python -m scripts.error_analysis outputs/.../<structured_test_run> --compare debate
```

Deliverable: `error_analysis_test.md` — top failure modes giải thích val→test gap (−30 pp).

### 1.3 Vanilla config chưa thử

Ablation tốt nhất cho structured: `retrieval=off`. Vanilla chưa chắc đã chạy với `retrieval=off`.

- Chạy val 53 → nếu EM > 0.6792, cập nhật paper config
- **Không** chạy lại test trừ khi val cải thiện rõ và đã frozen config mới

---

## Tuần 2 — Báo cáo & reproducibility (P0)

### 2.1 Bảng paper

| Table | Nguồn |
|---|---|
| Main results val | `results-summary.md` |
| Test one-shot | `test_metrics/*.json` |
| Ablation | `p1_ablation_summary.csv` |
| Reader baselines | `reader_metrics/*.json` |
| Error taxonomy | `error_analysis.md` (val + test) |

### 2.2 Case studies (appendix)

Chọn 3 cases từ error analysis:

1. **vilqa-236** — prefix duration (đã fix postprocess)
2. **vilqa-499** — list answer (limitation)
3. 1 case vanilla đúng / structured sai (debate value)

### 2.3 Reproducibility

```bash
python -m unittest discover -s tests -q
python scripts/verify_reader_deps.py
bash scripts/run_p1_ablations.sh          # dry-run commands
```

Ghi trong paper: split seed=42, model qwen3.5:9b, Ollama context 8192.

---

## Tuần 3 — Optional improvements (P1, chỉ nếu tuần 1–2 xong)

| Thí nghiệm | Kỳ vọng | Rủi ro |
|---|---|---|
| Hybrid reader → judge rerank | F1↑, EM~giữ | Effort trung bình |
| PhoBERT reader fine-tune | EM reader↑ | Train lại 1 lần |
| `enable_llm_evaluator` subset N=20 | Rubric scores | API cost |

**Không làm:** BM25 retrieval, rounds>1, tune trên test, nhiều lần test cherry-pick.

---

## Claims an toàn (đã có evidence)

1. Vanilla self-debate > structured debate trên val 53 (qwen3.5:9b).
2. Retrieval BM25 **giảm** EM (debate và reader).
3. Memory read-only **tăng** EM structured (+9.4 pp).
4. Fine-tuned reader = best non-LLM baseline (EM 0.5849 val).
5. Test generalization yếu hơn val (~30 pp EM) — báo cáo + error analysis.

---

## Checklist nhanh (copy vào PR / báo cáo)

```
[ ] Re-val sau postprocess fix
[ ] Error analysis test split
[ ] Vanilla retrieval=off (val only)
[ ] Cập nhật CSV nếu có run mới
[ ] Appendix 3 case studies
[ ] Limitations paragraph (val/test gap, strict EM, list answers)
[ ] Không mở Phase 3 courtroom cho đến khi 6 mục trên xong
```

---

## Thứ tự ưu tiên (3 việc đầu)

1. **Re-run validation** vanilla + structured optimized (sau fix `Sau`)
2. **Error analysis test** từ predictions đã có
3. **Viết limitations** + case studies từ error analysis
