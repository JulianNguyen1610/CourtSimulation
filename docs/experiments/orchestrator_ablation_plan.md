# Orchestrator Ablation: Fixed vs Judge-Mediated

## Mục tiêu

So sánh hai cách điều phối Phase 1 structured debate trên **cùng config paper secondary**:

| Tham số | Giá trị |
|---|---|
| Split | validation (53 cases) |
| Rounds | 1 |
| Retrieval | off |
| Memory | read_only |
| Closing | on |
| Judge question | off |
| Seed / split | 42 (mặc định `src.main`) |

## Biến thể

| Variant | `orchestrator` | Method trong metrics | Giả thuyết |
|---|---|---|---|
| **B (primary)** | `judge_mediated` | `debate_judge_mediated` | Judge điều phối — **config chính dự án** (val EM 0.6792) |
| A (legacy ablation) | `fixed` | `debate` | Turn order Python; EM ref 0.6038 |

Mọi thứ khác giữ nguyên (model, temperature, postprocess).

## Lệnh chạy

### 1. Dry-run (in command, không gọi API)

```bash
python scripts/run_orchestrator_ablation.py
```

### 2. Mock smoke (2 cases, CI/local)

```bash
python scripts/run_orchestrator_ablation.py --llm mock --limit 2 --execute
```

### 3. Server Ollama — full validation 53

```bash
export LOCAL_LLM_REASONING_EFFORT=none
bash scripts/run_orchestrator_ablation.sh --execute
```

Hoặc trực tiếp:

```bash
export LOCAL_LLM_REASONING_EFFORT=none
python scripts/run_orchestrator_ablation.py \
  --config configs/ollama.yaml \
  --llm local \
  --local-model qwen3.5:9b \
  --local-endpoint http://localhost:11434/v1/chat/completions \
  --local-timeout 1200 \
  --split validation \
  --limit 0 \
  --rounds 1 \
  --retrieval-method off \
  --memory-mode read_only \
  --execute \
  --continue-on-error
```

### 4. Windows (PowerShell)

```powershell
.\scripts\run_orchestrator_ablation.ps1 -Execute
```

### 5. Chỉ chạy một variant

```bash
python scripts/run_orchestrator_ablation.py \
  --variants judge_mediated \
  --llm local --limit 0 --execute
```

## Output

| Artifact | Đường dẫn |
|---|---|
| Commands | `outputs/orchestrator_ablation/<run_id>/commands.csv` |
| Metrics mỗi variant | `outputs/orchestrator_ablation/<run_id>/<variant>/<timestamp>_validation_debate/metrics.json` |
| Tổng hợp CSV | `docs/experiments/orchestrator_ablation_results.csv` (append khi `--execute`) |

## Đọc kết quả

```bash
python -c "
import json, pathlib
root = pathlib.Path('outputs/orchestrator_ablation')
for p in sorted(root.glob('*/*/metrics.json'))[-2:]:
    d = json.loads(p.read_text(encoding='utf-8'))
    for m, v in d.get('metrics_by_method', {}).items():
        print(p.parent.parent.name, m, v.get('exact_match'), v.get('f1'))
"
```

## Tiêu chí đánh giá

- **EM** là metric chính (extractive legal QA).
- Cải thiện có ý nghĩa nếu judge_mediated **≥ +2 pp EM** so với fixed mà không tăng fallback_rate đáng kể.
- Nếu EM không đổi: chạy error analysis head-to-head (`scripts/error_analysis.py`) trên hai run dir.

## Thời gian ước tính (qwen3.5:9b, validation 53)

- Fixed: ~53 × (2 agent turns + judge belief + verdict) ≈ vài giờ
- Judge-mediated: thêm ~1 LLM call/control step/turn → **~20–40% chậm hơn** fixed

Chạy qua đêm với `--continue-on-error` nếu một variant fail giữa chừng.

## Reference

- Fixed baseline rerun: `docs/experiments/rerun_20260629/validation_debate/` (EM 0.6038)
- Error analysis fixed: `docs/experiments/error_analysis_val_rerun_20260629.md`
