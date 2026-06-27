# Active Context

## Đang Làm Gì
- **Phase 1 hoàn tất (M1)**: baselines val 53, ablation matrix, test split one-shot, báo cáo đồng bộ.
- **Phase 3 pilot (partial)**: mock courtroom + LJP eval + artifact save done (2026-06-27).
- **Tiếp theo**: courtroom LLM thật trên server (qwen3.5:9b), D.6 batch runner.

## Kết quả chính (qwen3.5:9b, ALQAC split seed=42)

### Validation 53

| Method | EM | F1 |
|---|---:|---:|
| Vanilla debate | **0.6792** | **0.9401** |
| Structured, retrieval=off | 0.5849 | 0.8535 |
| Structured r=1 (baseline) | 0.4906 | 0.8124 |
| Direct | 0.2453 | 0.6634 |

### Test 53 (one-shot, 2026-06-27)

| Method | EM | F1 |
|---|---:|---:|
| Vanilla | **0.3774** | **0.7712** |
| Structured optimized | 0.3208 | 0.6957 |

### Courtroom pilot mock (case_01_theft, 2026-06-27)
- 10 transcript turns, 13 phases completed
- Artifact: `outputs/courtroom_pilot/20260627T084344Z_vn-theft-001/`
- Gemini local: quota exceeded (429) — cần chạy trên server Ollama

## Paper config (frozen trước test)

```yaml
# Primary
method: vanilla
rounds: 1

# Secondary
method: debate
retrieval: off
memory: read_only
rounds: 1
closing: on
```

## Thay đổi gần đây (2026-06-27)
- `save_courtroom_result()` + `--save-courtroom` / `--courtroom-output-dir`
- LJP eval tự động khi có ground_truth; lưu `ljp_metrics.json`
- Finetuned reader wired vào `BaselineBatchRunner` (`finetuned_reader`, `tuned_bm25_reader`)
- `scripts/train_reader.py` + `src/reader/finetune_reader.py` scaffold
- 36 unit tests pass (+1 courtroom artifact test)

## Bước Tiếp Theo
1. **D.3.8** Courtroom pilot qwen3.5:9b trên server spark-063e
2. **D.6.1** `CourtroomBatchRunner` MVP
3. Fine-tune reader + eval `finetuned_reader` baseline
4. **E.2** Human eval subset
