#!/usr/bin/env bash
# Fixed vs judge-mediated orchestrator ablation (validation 53, paper secondary config).
#
# Usage:
#   bash scripts/run_orchestrator_ablation.sh              # dry-run
#   bash scripts/run_orchestrator_ablation.sh --execute    # run on server (Ollama)
#
# Prerequisites:
#   ollama serve && ollama pull qwen3.5:9b
#   export LOCAL_LLM_REASONING_EFFORT=none

set -euo pipefail

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
    EXECUTE=true
fi

export LOCAL_LLM_REASONING_EFFORT="${LOCAL_LLM_REASONING_EFFORT:-none}"

ARGS=(
    python scripts/run_orchestrator_ablation.py
    --config configs/ollama.yaml
    --llm local
    --local-model qwen3.5:9b
    --local-endpoint http://localhost:11434/v1/chat/completions
    --local-timeout 1200
    --split validation
    --limit 0
    --rounds 1
    --retrieval-method off
    --memory-mode read_only
    --continue-on-error
)

if $EXECUTE; then
    ARGS+=(--execute)
fi

echo "Command: ${ARGS[*]}"
"${ARGS[@]}"
