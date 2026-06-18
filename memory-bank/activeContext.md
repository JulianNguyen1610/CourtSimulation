# Active Context

## Đang Làm Gì
- Phase 3 courtroom LJP scaffold đã được implement (agents, protocol, session, data model, LJP metrics).
- Trọng tâm tiếp theo: validation pilot case VN bằng LLM thật, tích hợp dataset SimuCourt/VLegal-Bench, và batch courtroom experiments.
- Phase 1 ViLQA/ALQAC debate vẫn chạy qua `DebateOrchestrator` và `DebateAgent` (backward compatible).

## Thay Đổi Gần Đây
- **Phase 3 agents**: `ProsecutorAgent`, `DefenseAgent`, `DefendantAgent` + `BaseLegalAgent`; `JudgeAgent` mở rộng `open_session`, `deliberate`, `render_ljp_verdict`; `src/agents/compat.py` alias Phase 1.
- **Courtroom protocol**: `src/courtroom/protocol.py` (opening/debate_round/closing), `src/courtroom/session.py` (lifecycle 3 giai đoạn), `configs/courtroom.yaml`.
- **Data model**: `CourtCase`, `EvidenceItem`, `Testimony`, `JudgmentGroundTruth`, `LegalJudgment`, `CourtroomResult`, `LJPEvalResult` trong `src/models.py`.
- **Data loader**: `load_court_case_json`, `load_simucourt`, `load_vlegal` trong `src/data_loader.py`; pilot `data/processed/case_01_theft.json`.
- **LJP evaluation**: `src/evaluation/ljp_evaluator.py` — charge/article accuracy, sentence MAE/RMSE/bucket, citation validity hooks.
- **CLI**: `--run-courtroom`, `--courtroom-case`, `--courtroom-config` trong `src/main.py`.
- **Tests**: `tests/test_phase5_courtroom.py` — 23 tests pass tổng cộng.

## Bước Tiếp Theo
1. Chạy pilot courtroom mock: `python -m src.main --run-courtroom --llm mock`.
2. Chạy pilot VN theft case bằng Gemini/local khi API sẵn sàng.
3. Thử `load_simucourt()` / `load_vlegal()` với HF datasets; điều chỉnh field mapping nếu schema khác.
4. Thêm batch runner cho courtroom LJP (tương tự `BaselineBatchRunner`).
5. Tiếp tục P1 ablation validation song song khi cần so sánh QA vs LJP.
