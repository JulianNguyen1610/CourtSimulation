# Progress

## Đã Hoàn Thành
- **Khởi tạo dự án**: Multi-Agent Courtroom Simulation Framework.
- **Phase 1 ViLQA scaffold**: data loader, split, `DebateOrchestrator`, baselines, retrieval, memory, evaluator EM/F1.
- **P0/P1 pipeline**: LLM providers, judge fallback, semantic rerank, memory ablation, debate loop, ablation matrix script.
- **Tài liệu dự án (2026-06-17)**:
  - `docs/project-status-and-overview.md` — tình trạng, kiến trúc, kết quả thí nghiệm.
  - `docs/advanced-techniques-analysis.md` — phân tích kỹ thuật từ 6 papers.
  - `docs/implementation-checklist.md` — checklist triển khai chi tiết (~144 mục).
- **Phase 3 courtroom scaffold (items 10–13)**:
  - Agents theo vai trò pháp lý: prosecutor, defense, defendant, judge LJP.
  - Courtroom protocol 3 giai đoạn: opening → debate → judgment (closing + deliberation + ruling).
  - `CourtCase` schema + loaders + pilot case VN.
  - `LJPEvaluator` cho charge/article/sentence metrics.
  - Backward compat Phase 1 qua `DebateAgent` + `compat.py`.
- **Tooling (2026-06-20)**:
  - `scripts/error_analysis.py` — taxonomy lỗi legal QA debate, cross-method direct vs debate.
  - `scripts/setup_server.sh` — verify GPU/Ollama/context/deps trên Linux server.
- **P1 validation qwen3.5:9b (server spark-063e, 2026-06-19/20)**:
  - Run chính `both` validation 53: debate r=1 > direct (fair config).
  - Ablation rounds 1/3/5 hoàn tất.
  - Error analysis trên run `20260619T212113Z_validation_both`.
  - Phân tích 4 case debate regression (prefix / over-extract / partial list).

## Kết Quả Thí Nghiệm

### Bảng tổng hợp (validation 53, ALQAC, qwen3.5:9b — runs chính)

| Thí nghiệm | Method | Rounds | EM | F1 | Fallback | Run / Ghi chú |
|---|---|---:|---:|---:|---:|---|
| **both (config fix)** | direct | 1 | 0.2453 | 0.6634 | — | `20260619T212113Z_validation_both` |
| **both (config fix)** | debate | 1 | **0.4906** | **0.8124** | 4.72% | **Tốt nhất** — debate thắng direct |
| debate ablation | debate | 3 | 0.4151 | 0.7633 | 1.89% | Thấp nhất trong ablation rounds |
| debate ablation | debate | 5 | 0.4528 | 0.8048 | 0.94% | Giữa r=1 và r=3; vẫn thua r=1 |

### Ablation rounds (debate only, validation 53, qwen3.5:9b)

| Rounds | EM | F1 | Δ EM vs r=1 | Fallback | Kết luận |
|---:|---:|---:|---:|---:|---|
| **1** | **0.4906** | **0.8124** | — | 4.72% | **Optimum** trên model 9B |
| 3 | 0.4151 | 0.7633 | −15.4% | 1.89% | Belief drift / context noise |
| 5 | 0.4528 | 0.8048 | −7.7% | 0.94% | Phục hồi một phần so với r=3; không vượt r=1 |

### Error analysis — run `20260619T212113Z_validation_both`

| Metric | direct | debate |
|---|---:|---:|
| EM | 0.2453 (13/53) | 0.4906 (26/53) |
| Lỗi OVER_EXTRACTION | 27 (67.5%) | 15 (55.6%) |
| Debate wins / Direct wins / Both OK / Both wrong | — | 17 / 4 / 9 / 23 |

**Debate regression (4 case direct đúng → debate sai):**

| case_id | Gold | Direct (EM) | Debate (EM) | Loại lỗi |
|---|---|---|---|---|
| vilqa-236 | `01 tháng` | `01 tháng` (1.0) | `Sau 01 tháng` (0.0, F1=0.8) | Prefix thừa "Sau" |
| vilqa-125 | `do hai bên thỏa thuận` | đúng (1.0) | `theo điều kiện do hai bên thỏa thuận và không bị tính lãi` | Over-extraction / span dài |
| vilqa-36 | `03 năm` | `03 năm` (1.0) | `phạt cải tạo không giam giữ đến 03 năm hoặc...` | Over-extraction (chọn cụm hình phạt dài) |
| vilqa-499 | Danh sách 4 điều kiện pháp nhân | full gold (1.0) | chỉ điều kiện (a) | Partial span / list under-extract |

### Runs tham chiếu (không dùng claim chính)

| Thí nghiệm | Kết quả | Lý do |
|---|---|---|
| qwen both, max_tokens=128 | debate >> direct | Direct handicap (JSON cắt) |
| dolphin3 both | direct F1 > debate | Model nhỏ; over-extract |
| dolphin3 direct | EM=0.2642 | Chỉ tham chiếu local |

### Runs kỹ thuật khác

| Thí nghiệm | Kết quả |
|---|---|
| Unit tests offline | 28 tests pass |
| Phase 3 courtroom smoke | MockLLM OK |

## Đang Thực Hiện
- Baselines còn thiếu: cot, vanilla, extractive_qa, bm25_reader trên validation 53.
- Ablation retrieval (off / bm25 / rerank) và memory.
- Validation courtroom pilot LLM thật.
- `docs/experiments/results-summary.md` cho paper.

## Còn Lại
- [ ] Bảng baselines đầy đủ trên validation 53.
- [ ] Ablation retrieval + memory.
- [ ] Test split — 1 lần sau khi chốt config (r=1, bm25_only).
- [ ] Fix postprocess: strip prefix "Sau", list-answer handling (vilqa-499).
- [ ] Batch courtroom runner + LJP metrics.
- [ ] Citation validity / hallucination checker.
- [ ] Human eval rubric.
- [ ] Phase 3 pilot LLM thật.

## Vấn Đề Đã Biết
- Strict EM nhạy prefix ("Sau 01 tháng" vs "01 tháng") — cân nhắc relaxed EM cho analysis.
- Debate over-extract vẫn 55.6% lỗi; direct 67.5%.
- Rounds>1 trên qwen3.5:9b không cải thiện EM; r=1 là default cho paper.
- Câu trả lời dạng danh sách (vilqa-499) debate chọn 1 mục thay vì full list.
- dolphin3 không phù hợp debate baseline.

## Metric Tracking — Bảng chính (validation 53, qwen3.5:9b fair config)

| Method | Rounds | EM | F1 | Run ID |
|---|---:|---:|---:|---|
| direct | 1 | 0.2453 | 0.6634 | 20260619T212113Z_validation_both |
| debate | 1 | **0.4906** | **0.8124** | 20260619T212113Z_validation_both |
| debate | 3 | 0.4151 | 0.7633 | server 2026-06-19 |
| debate | 5 | 0.4528 | 0.8048 | server 2026-06-19/20 |
