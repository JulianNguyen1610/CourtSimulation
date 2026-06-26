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
- **P1 validation qwen3.5:9b (server spark-063e, 2026-06-19/26)**:
  - Run chính `both` validation 53: structured debate r=1 > direct (fair config).
  - Ablation rounds 1/3/5 hoàn tất: **r=1 optimum**.
  - **Toàn bộ 6 baselines validation 53 hoàn tất** (direct, cot, vanilla, structured debate r=1/3/5, extractive_qa, bm25_reader).
  - Error analysis trên run `20260619T212113Z_validation_both`.
  - Phân tích 4 case debate regression (prefix / over-extract / partial list).
  - Python 3.10 compatibility fix (`datetime.UTC` → `timezone.utc`).
- **Ablation matrix P1 scaffold (2026-06-26)**:
  - `scripts/run_ablation_matrix.py` cập nhật: 10 variants, r=1 base, closing/judge_question flags, Python 3.10 compat fix.
  - `scripts/run_p1_ablations.sh` script bash chạy tuần tự trên server.
  - Unit test smoke OK cho tất cả ablation flags.
- **B.3.9 Memory leak fix (2026-06-26)**:
  - Bug: `_append_default_memories` lưu gold answer trong `context_excerpt` / `text` mà không qua `_sanitize_entry`.
  - Fix: chạy `_sanitize_entry()` trên mọi default entry; thêm `"prediction"` vào sanitize keys.
  - 4 new unit tests `MemoryLeakPreventionTest` — 35 total tests pass.

## Kết Quả Thí Nghiệm

### Bảng đầy đủ — validation 53 (ALQAC, qwen3.5:9b, server spark-063e)

| Rank | Method | EM | F1 | Category |
|---:|---:|---:|---:|---|
| **1** | **Vanilla debate (self-debate)** | **0.6792** | **0.9401** | Multi-agent LLM (single-prompt) |
| 2 | Structured debate r=1 | 0.4906 | 0.8124 | Multi-agent LLM (multi-turn) |
| 3 | Chain-of-Thought (CoT) | 0.4717 | 0.8610 | Single-agent LLM |
| 4 | Extractive QA reader | 0.3585 | 0.6413 | Reader (XLM-RoBERTa) |
| 5 | Direct prediction | 0.2453 | 0.6634 | Single-agent LLM |
| 6 | BM25 + reader | 0.1887 | 0.4557 | Retrieval + reader |

### Ablation rounds — structured debate only

| Rounds | EM | F1 | Δ EM vs r=1 | Fallback | Kết luận |
|---:|---:|---:|---:|---:|---|
| **1** | **0.4906** | **0.8124** | — | 4.72% | Optimum trong structured debate |
| 3 | 0.4151 | 0.7633 | −15.4% | 1.89% | Belief drift / context noise |
| 5 | 0.4528 | 0.8048 | −7.7% | 0.94% | Phục hồi một phần; vẫn thua r=1 |

### Error analysis — run `20260619T212113Z_validation_both` (structured debate vs direct)

| Metric | direct | structured debate |
|---|---:|---:|
| EM | 0.2453 (13/53) | 0.4906 (26/53) |
| Lỗi OVER_EXTRACTION | 27 (67.5%) | 15 (55.6%) |
| Debate wins / Direct wins / Both OK / Both wrong | — | 17 / 4 / 9 / 23 |

**Structured debate regression (4 case direct đúng → debate sai):**

| case_id | Gold | Direct (EM) | Debate (EM) | Loại lỗi |
|---|---|---|---|---|
| vilqa-236 | `01 tháng` | `01 tháng` (1.0) | `Sau 01 tháng` (0.0, F1=0.8) | Prefix thừa "Sau" |
| vilqa-125 | `do hai bên thỏa thuận` | đúng (1.0) | `theo điều kiện do hai bên thỏa thuận và không bị tính lãi` | Over-extraction / span dài |
| vilqa-36 | `03 năm` | `03 năm` (1.0) | `phạt cải tạo không giam giữ đến 03 năm hoặc...` | Over-extraction (chọn cụm hình phạt dài) |
| vilqa-499 | Danh sách 4 điều kiện pháp nhân | full gold (1.0) | chỉ điều kiện (a) | Partial span / list under-extract |

### Ablation matrix — cần chạy trên server

| ID | Variant | Retrieval | Memory | Rounds | Closing | Judge Q | Status |
|---|---|---|---|---|---|---|---|
| ABL-01 | retrieval_off | off | off | 1 | on | off | pipeline OK, chờ server |
| ABL-02 | retrieval_bm25 | bm25_only | off | 1 | on | off | EM=0.4906 ref |
| ABL-03 | retrieval_bm25_rerank | bm25_rerank | off | 1 | on | off | pipeline OK, chờ server |
| ABL-04 | memory_off | bm25_only | off | 1 | on | off | EM=0.4906 ref |
| ABL-05 | memory_read_only | bm25_only | read_only | 1 | on | off | pipeline OK, chờ server |
| ABL-06 | memory_read_update | bm25_only | read_update | 1 | on | off | pipeline OK, chờ server |
| ABL-10 | judge_off_vanilla | — | — | — | — | — | EM=0.6792 |
| ABL-11 | closing_off | bm25_only | off | 1 | off | off | pipeline OK, chờ server |
| ABL-12 | judge_question_on | bm25_only | off | 1 | on | on | pipeline OK, chờ server |

### Runs tham chiếu (không dùng claim chính)

| Thí nghiệm | Kết quả | Lý do |
|---|---|---|
| qwen both, max_tokens=128 | debate >> direct | Direct handicap (JSON cắt) |
| dolphin3 both | direct F1 > debate | Model nhỏ; over-extract |
| dolphin3 direct | EM=0.2642 | Chỉ tham chiếu local |

### Runs kỹ thuật khác

| Thí nghiệm | Kết quả |
|---|---|
| Unit tests offline | 35 tests pass (4 new B.3.9 leak tests) |
| Phase 3 courtroom smoke | MockLLM OK |

## Đang Thực Hiện
- Chạy 6 ablation variants trên server spark-063e (Ollama qwen3.5:9b) — local không có Ollama, Gemini quota hết.
- Tổng hợp kết quả vào `docs/experiments/p1_ablation_summary.csv` sau khi server run xong.

## Còn Lại
- [ ] Chạy ablation trên server: retrieval_off, bm25_rerank, memory_read_only, memory_read_update, closing_off, judge_question_on
- [ ] Fix postprocess: strip prefix "Sau", list-answer handling (vilqa-499)
- [ ] Test split — 1 lần sau khi chốt config
- [ ] Batch courtroom runner + LJP metrics
- [ ] Citation validity / hallucination checker
- [ ] Human eval rubric
- [ ] Phase 3 pilot LLM thật
- [ ] `docs/experiments/results-summary.md`

## Vấn Đề Đã Biết
- **Vanilla debate (0.68) vượt structured debate (0.49)**: Single-prompt self-debate hiệu quả hơn multi-turn structured debate trên model 9B.
- Strict EM nhạy prefix ("Sau 01 tháng" vs "01 tháng") — cân nhắc relaxed EM cho analysis.
- Structured debate over-extract vẫn 55.6% lỗi; direct 67.5%.
- Rounds>1 trên qwen3.5:9b không cải thiện EM; r=1 là default cho structured debate.
- Câu trả lời dạng danh sách (vilqa-499) debate chọn 1 mục thay vì full list.
- BM25 retrieval làm **giảm** QA performance (extractive_qa 0.36 → BM25+reader 0.19).
- dolphin3 không phù hợp debate baseline.
- **FIXED**: `_append_default_memories` leak gold answer — đã thêm `_sanitize_entry()` call.

## Metric Tracking — Bảng chính (validation 53, qwen3.5:9b fair config)

| Method | EM | F1 | Run ID |
|---:|---:|---|
| **vanilla debate** | **0.6792** | **0.9401** | server 2026-06-26 |
| structured debate r=1 | 0.4906 | 0.8124 | 20260619T212113Z_validation_both |
| cot | 0.4717 | 0.8610 | server 2026-06-26 |
| structured debate r=5 | 0.4528 | 0.8048 | server 2026-06-19/20 |
| structured debate r=3 | 0.4151 | 0.7633 | server 2026-06-19 |
| extractive_qa | 0.3585 | 0.6413 | server 2026-06-26 |
| direct | 0.2453 | 0.6634 | 20260619T212113Z_validation_both |
| bm25_reader | 0.1887 | 0.4557 | server 2026-06-26 |
