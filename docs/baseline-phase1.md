# ViLQA Multi-Agent Simple Baseline

## Objective

Set up and verify the first runnable baseline for a ViLQA/ALQAC-based multi-agent legal debate system with retrieval, memory, transcript artifacts, and answer-level evaluation.

## Dataset Assumption

- Dataset file: `data/ALQAC.csv`
- Task type: Vietnamese legal extractive question answering
- Required columns: `context`, `question`, `answer`
- Current split strategy: random split with seed `42`
  - Train: `80%`
  - Validation: `10%`
  - Test: `10%`

The loader only normalizes data and splits cases. Phase 2 adds deterministic mock debate orchestration without using API calls or model inference. Phase 3 adds lightweight BM25 retrieval, JSON-backed memory, and optional transcript saving. Phase 4 completes the simple baseline with exact-match/F1 evaluation and batch experiment outputs.

## Implemented Files

- `src/models.py`: shared Pydantic models for cases, evidence, memory, agent outputs, judge beliefs, verdicts, and evaluation results.
- `src/data_loader.py`: ViLQA/ALQAC CSV loader and reproducible split helper.
- `src/llm.py`: minimal LLM protocol and deterministic `MockLLM`.
- `src/agents/debate_agent.py`: `DebateAgent` for Proponent and Opponent.
- `src/agents/judge_agent.py`: `JudgeAgent` with belief tracking and robust JSON fallback parsing.
- `src/orchestrator.py`: `DebateOrchestrator` for `n` rounds.
- `src/retrieval/legal_retriever.py`: lightweight BM25 retriever over ViLQA contexts.
- `src/memory/memory_store.py`: JSON-backed memory store with regulations, experiences, and case memories.
- `src/artifacts.py`: JSON transcript/result saving.
- `src/evaluation/evaluator.py`: ViLQA exact-match and token-F1 evaluation.
- `src/baselines.py`: deterministic direct context-candidate baseline.
- `src/experiment_runner.py`: batch runner for `direct`, `debate`, or `both` methods.
- `src/main.py`: smoke-test CLI for validating data loading and mock debate execution.
- `configs/default.yaml`: default experiment configuration.
- `configs/prompts/`: prompt templates for Proponent, Opponent, Judge, Evaluator, and memory update.
- `tests/test_phase2_orchestrator.py`: sanity tests for debate artifacts and answer-leakage protection.
- `tests/test_phase3_retrieval_memory.py`: sanity tests for retrieval and memory round-trip behavior.
- `tests/test_phase4_evaluation_runner.py`: sanity tests for metrics and batch output artifacts.

## Verification

```bash
python -m src.main --dataset "data/ALQAC.csv"
python -m src.main --dataset "data/ALQAC.csv" --run-debate --case-index 0 --rounds 3
python -m src.main --dataset "data/ALQAC.csv" --run-debate --case-index 0 --rounds 2 --evidence-top-k 3 --save-result
python -m src.main --dataset "data/ALQAC.csv" --run-debate --case-index 0 --rounds 2 --update-memory
python -m src.main --dataset "data/ALQAC.csv" --run-batch --split validation --method both --limit 0 --rounds 1 --evidence-top-k 3
python -m unittest discover -s tests
python -m compileall src
```

Expected smoke-test output:

```text
Baseline scaffold loaded successfully.
Split sizes: train=424, validation=53, test=53
Mock debate completed.
Retrieved evidence: 3
Retrieved memory: R=0, E=0, C=0
Transcript turns: 4
Belief updates: 2
```

## Batch Outputs

Each batch run creates a timestamped directory under `outputs/vilqa_multi_agent_baseline/`:

```text
outputs/vilqa_multi_agent_baseline/<timestamp>_<split>_<method>/
  config.json
  metrics.json
  predictions.csv
```

The first full validation run is:

```text
outputs/vilqa_multi_agent_baseline/20260617T045228Z_validation_both/
```

Validation metrics from this run:

```json
{
  "direct": {"exact_match": 0.0377, "f1": 0.1405},
  "debate": {"exact_match": 0.0377, "f1": 0.1405}
}
```

The two methods are currently identical in score because `debate` still uses `MockLLM` and the same deterministic context-candidate fallback. This is a lower-bound systems baseline, not a claim that debate improves ViLQA.

## Current Limitations

- Retrieval is lexical BM25 only; semantic re-ranking is not implemented yet.
- The retrieval index is built from the training split in CLI runs to avoid fitting on validation/test examples.
- Memory update is simple append-only JSON and not yet evaluated for usefulness.
- `MockLLM` is deterministic and only validates orchestration. It is not an experimental model baseline.
- Direct answer selection is a weak regex heuristic, so EM/F1 are expected to be low.
- The gold `answer` is excluded from `CaseProfile.agent_view()` to prevent leakage into agents and judge prompts.

## Next Development Steps

1. Replace `MockLLM` with a real local/API LLM client.
2. Replace the direct regex heuristic with a proper extractive QA baseline.
3. Add semantic re-ranking (`bge-m3` or multilingual-e5) after BM25.
4. Evaluate whether memory update helps or hurts across validation folds.
5. Add error analysis for low-F1 predictions and ambiguous questions.
