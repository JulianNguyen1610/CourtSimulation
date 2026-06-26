# P1 Ablation Study Plan

## Reference System
Full Phase 1 system: structured debate with `proponent` and `opponent`, judge belief tracking, closing statements, BM25 retrieval, read-only memory, 3 rounds, EM/F1 automated metrics, and optional LLM rubric evaluation.

## Controlled Variables
| Variant | Removed/Changed Component | Hypothesis | Primary Metric | Expected Effect |
|---|---|---|---|---|
| `retrieval_off` | No retrieved legal evidence | Retrieval should improve evidence grounding | EM/F1, legal_accuracy | Lower factual support |
| `bm25_plus_rerank` | Add semantic rerank after BM25 | Semantic rerank should improve top-k evidence quality | EM/F1, legal_accuracy | Higher evidence relevance, higher cost |
| `memory_off` | No memory context | Memory should help reuse prior strategies/cases | EM/F1, argument_quality | Lower debate quality if memory is useful |
| `memory_update_on` | Read and update memory after cases | Reflection memory may help later cases | EM/F1, fallback_rate | Potential gain, risk of noisy memory |
| `rounds_1` | One debate round | Fewer turns reduce reasoning depth | EM/F1, argument_quality | Lower cost, possibly lower quality |
| `rounds_5` | Five debate rounds | More debate may improve or cause drift | EM/F1, logical_consistency | Higher cost, possible hallucination risk |
| `judge_off_vanilla` | No structured judge | Judge should improve final answer selection | EM/F1 | Vanilla consensus may be weaker |
| `roles_prosecutor_defense` | Courtroom role naming | Legal roles may improve Phase 2 realism | rubric metrics | Not implemented for Phase 1 yet |

## Execution
Dry-run command generation:

```powershell
python scripts/run_ablation_matrix.py --llm mock --limit 5
```

Execute validation ablations after API/config is stable:

```powershell
python scripts/run_ablation_matrix.py --llm gemini --limit 20 --execute
```

Enable semantic rerank only when `sentence-transformers` model downloads are acceptable:

```powershell
python scripts/run_ablation_matrix.py --llm gemini --limit 20 --include-heavy-rerank --execute
```

## Result Table
The script writes:
- `outputs/p1_ablation_matrix/<timestamp>/commands.csv`
- `docs/experiments/p1_ablation_summary.csv` when `--execute` is used

Interpret only validation results during prompt/config tuning. Run test split once after the ablation decision is finalized.
