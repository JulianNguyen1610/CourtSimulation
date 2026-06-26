# Active Context

## Đang Làm Gì
- **Bước 2–4 ablation P1**: pipeline đã verify (mock, 6 variants); **chờ chạy LLM thật trên server** (Ollama local không có, Gemini quota hết).
- B.3.9 memory leak: **done** — 4 unit tests pass.
- ABL-10 (judge_off vanilla): **đã có** EM=0.6792 — không cần chạy lại.

## Trạng thái ablation pending (cần qwen3.5:9b trên spark-063e)

| Variant | Nhóm | Status |
|---|---|---|
| retrieval_off | B.2 retrieval | pipeline OK, chưa có metric thật |
| retrieval_bm25_rerank | B.2 retrieval | pipeline OK (+ BGE-m3 load OK) |
| memory_read_only | B.3 memory | pipeline OK |
| memory_read_update | B.3 memory | pipeline OK (isolated memory file) |
| closing_off | ABL-11 | pipeline OK |
| judge_question_on | ABL-12 | pipeline OK |

## Bước Tiếp Theo
1. **Chạy trên server**: `bash scripts/run_p1_ablations.sh --execute` hoặc `.\scripts\run_p1_ablations.ps1 -Execute`
2. Gộp kết quả vào `p1_ablation_summary.csv`
3. Viết `docs/experiments/results-summary.md`
4. Test split 1 lần sau khi chốt config
