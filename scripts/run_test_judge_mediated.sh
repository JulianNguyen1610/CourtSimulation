#!/usr/bin/env bash
# Test split one-shot — project primary config (judge_mediated debate).
#
# Usage:
#   bash scripts/run_test_judge_mediated.sh              # dry-run (prints command)
#   bash scripts/run_test_judge_mediated.sh --execute    # run full test 53
#   bash scripts/run_test_judge_mediated.sh --execute --limit 2  # smoke
#
# Prerequisites: Ollama + qwen3.5:9b, LOCAL_LLM_REASONING_EFFORT=none

set -euo pipefail

EXECUTE=false
LIMIT=0
OUTPUT_DIR="outputs/test_metrics/judge_mediated_test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --limit) LIMIT="${2:-0}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export LOCAL_LLM_REASONING_EFFORT="${LOCAL_LLM_REASONING_EFFORT:-none}"

CMD=(
    python -m src.main
    --config configs/ollama.yaml
    --run-batch
    --llm local
    --local-model qwen3.5:9b
    --local-endpoint http://localhost:11434/v1/chat/completions
    --local-timeout 1200
    --split test
    --method debate
    --limit "$LIMIT"
    --rounds 1
    --retrieval-method off
    --memory-mode read_only
    --orchestrator judge_mediated
    --output-dir "$OUTPUT_DIR"
    --save-debate-artifacts
)

echo "Test one-shot (judge_mediated, frozen primary config)"
echo "Command: ${CMD[*]}"

if $EXECUTE; then
    "${CMD[@]}"
    echo ""
    echo "Done. Metrics: ${OUTPUT_DIR}/*/metrics.json"
else
    echo "(dry-run — add --execute to run)"
fi
