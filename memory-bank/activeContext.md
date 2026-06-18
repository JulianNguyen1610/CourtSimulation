# Active Context

## Đang Làm Gì
- Phase 3 courtroom LJP scaffold đã được implement (agents, protocol, session, data model, LJP metrics).
- Trọng tâm tiếp theo: validation pilot case VN bằng LLM thật, tích hợp dataset SimuCourt/VLegal-Bench, và batch courtroom experiments.
- Phase 1 ViLQA/ALQAC debate vẫn chạy qua `DebateOrchestrator` và `DebateAgent` (backward compatible).

## Thay Đổi Gần Đây
- **P1 local debate QA fix**: Sau validation `dolphin3:latest`, direct > debate rõ rệt; lỗi chính là debate over-extraction/paraphrase, không phải JSON fallback. Đã siết prompt `judge_belief`, `judge_verdict`, `proponent_argument`, `opponent_rebuttal`, `proponent_strategy`, `opponent_strategy` để bắt answer/prediction là span tiếng Việt trích nguyên văn; thêm `src/utils/answer_postprocess.py` và tích hợp vào `_run_debate` để rút các span legal phổ biến mà không dùng gold answer.
- **Công bằng hoá so sánh**: `shorten_legal_answer` giờ áp dụng đồng đều cho `direct`, `cot`, `vanilla`, `debate` (trước đây chỉ debate) để loại bias. Thêm pattern tiền "… đồng trở lên". Sửa bug `ollama.yaml` direct `max_output_tokens` 128→256 (gây JSON bị cắt và dump thô ở direct).
- **Kết quả validation 53 (dolphin3, rounds=1)**: direct EM=0.2642/F1=0.6802 vs debate (sau fix) EM=0.1698/F1=0.5111. Phân loại direct-thắng/debate-thắng/cả-hai-sai = 11/6/33; phần lớn lỗi còn lại là suy luận của model nhỏ, không phải format.
- **BREAKTHROUGH validation 53 (qwen3.5:9b, rounds=1)**: debate EM=0.4717/F1=0.8106 **vượt xa** direct EM=0.0189/F1=0.4034 — lần đầu debate > direct rõ rệt. NGUYÊN NHÂN direct sụp: config Linux chạy `direct.max_output_tokens=128` → qwen3.5 (thinking) sinh JSON dài bị cắt → parse fail. Repo đã để direct/cot=384; cần `git pull` trên máy Linux.
- **Robustness fix (để direct baseline đáng tin)**: thêm `_recover_json_field` trong `src/baselines.py` và `JudgeAgent._recover_json_field` để khôi phục `answer`/`prediction` từ JSON bị cắt; bump `ollama.yaml` direct/cot `max_output_tokens` 256→384. Debate KHÔNG bị động vào (đang là method thắng).
- **Phase 3 agents**: `ProsecutorAgent`, `DefenseAgent`, `DefendantAgent` + `BaseLegalAgent`; `JudgeAgent` mở rộng `open_session`, `deliberate`, `render_ljp_verdict`; `src/agents/compat.py` alias Phase 1.
- **Courtroom protocol**: `src/courtroom/protocol.py` (opening/debate_round/closing), `src/courtroom/session.py` (lifecycle 3 giai đoạn), `configs/courtroom.yaml`.
- **Data model**: `CourtCase`, `EvidenceItem`, `Testimony`, `JudgmentGroundTruth`, `LegalJudgment`, `CourtroomResult`, `LJPEvalResult` trong `src/models.py`.
- **Data loader**: `load_court_case_json`, `load_simucourt`, `load_vlegal` trong `src/data_loader.py`; pilot `data/processed/case_01_theft.json`.
- **LJP evaluation**: `src/evaluation/ljp_evaluator.py` — charge/article accuracy, sentence MAE/RMSE/bucket, citation validity hooks.
- **CLI**: `--run-courtroom`, `--courtroom-case`, `--courtroom-config` trong `src/main.py`.
- **Tests**: `tests/test_phase5_courtroom.py` — 23 tests pass tổng cộng.

## Bước Tiếp Theo
1. Chạy lại P1 `debate --split validation --limit 10 --rounds 1` bằng Ollama để kiểm tra prompt/postprocess mới.
2. Nếu limit-10 cải thiện đủ, chạy full validation 53; nếu vẫn thua direct, làm error analysis trước ablation.
3. Chạy pilot courtroom mock: `python -m src.main --run-courtroom --llm mock`.
4. Chạy pilot VN theft case bằng Gemini/local khi API sẵn sàng.
5. Thử `load_simucourt()` / `load_vlegal()` với HF datasets; điều chỉnh field mapping nếu schema khác.
6. Thêm batch runner cho courtroom LJP (tương tự `BaselineBatchRunner`).
