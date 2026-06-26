#!/usr/bin/env bash
# Run pending P1 ablation variants on server (qwen3.5:9b + Ollama).
#
# Usage:
#   bash scripts/run_p1_ablations.sh                    # dry-run (print commands)
#   bash scripts/run_p1_ablations.sh --execute           # actually run (validation 53)
#
# Or use the matrix runner directly:
#   python scripts/run_ablation_matrix.py --config configs/ollama.yaml --llm local \
#     --local-model qwen3.5:9b --split validation --limit 0 --include-heavy-rerank \
#     --pending-only --execute --continue-on-error

set -euo pipefail

CONFIG="configs/ollama.yaml"
LLM="local"
LOCAL_MODEL="qwen3.5:9b"
LOCAL_ENDPOINT="http://localhost:11434/v1/chat/completions"
SPLIT="validation"
LIMIT=0          # 0 = full split (53 cases)
TIMEOUT=1200
OUTPUT_ROOT="outputs/p1_ablation_matrix"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_DIR="$OUTPUT_ROOT/logs"
SUMMARY_CSV="$OUTPUT_ROOT/ablation_summary_${TIMESTAMP}.csv"

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
    EXECUTE=true
fi

export LOCAL_LLM_REASONING_EFFORT="${LOCAL_LLM_REASONING_EFFORT:-none}"

run_cmd() {
    local name="$1"; shift
    echo ""
    echo "===== ABLATION: $name ====="
    echo "Command: $*"
    if $EXECUTE; then
        mkdir -p "$LOG_DIR"
        time "$@" 2>&1 | tee "$LOG_DIR/${name}_${TIMESTAMP}.log"
    else
        echo "(dry-run — not executing)"
    fi
}

# Reference results (already run)
# ABL-02 retrieval=bm25_only: EM=0.4906 F1=0.8124 (20260619T212113Z_validation_both)
# ABL-04 memory=off:          EM=0.4906 F1=0.8124 (20260619T212113Z_validation_both)
# ABL-07 rounds=1:             EM=0.4906 F1=0.8124 (20260619T212113Z_validation_both)
# ABL-08 rounds=3:             EM=0.4151 F1=0.7633
# ABL-09 rounds=5:             EM=0.4528 F1=0.8048
# ABL-10 judge_off (vanilla):  EM=0.6792 F1=0.9401

# ---------------------------------------------------------------
# Group 1: Retrieval ablation (ABL-01, ABL-03)
# Base: structured debate r=1, memory=off
# ---------------------------------------------------------------

# ABL-01: retrieval=off
run_cmd "retrieval_off" \
    python -m src.main \
        --config "$CONFIG" \
        --run-batch \
        --llm "$LLM" \
        --local-model "$LOCAL_MODEL" \
        --local-endpoint "$LOCAL_ENDPOINT" \
        --local-timeout "$TIMEOUT" \
        --split "$SPLIT" \
        --method debate \
        --limit "$LIMIT" \
        --rounds 1 \
        --retrieval-method off \
        --memory-mode off \
        --output-dir "$OUTPUT_ROOT/retrieval_off"

# ABL-03: retrieval=bm25_rerank
run_cmd "retrieval_bm25_rerank" \
    python -m src.main \
        --config "$CONFIG" \
        --run-batch \
        --llm "$LLM" \
        --local-model "$LOCAL_MODEL" \
        --local-endpoint "$LOCAL_ENDPOINT" \
        --local-timeout "$TIMEOUT" \
        --split "$SPLIT" \
        --method debate \
        --limit "$LIMIT" \
        --rounds 1 \
        --retrieval-method bm25_rerank \
        --memory-mode off \
        --output-dir "$OUTPUT_ROOT/retrieval_bm25_rerank"

# ---------------------------------------------------------------
# Group 2: Memory ablation (ABL-05, ABL-06)
# Base: structured debate r=1, retrieval=bm25_only
# ---------------------------------------------------------------

# ABL-05: memory=read_only
run_cmd "memory_read_only" \
    python -m src.main \
        --config "$CONFIG" \
        --run-batch \
        --llm "$LLM" \
        --local-model "$LOCAL_MODEL" \
        --local-endpoint "$LOCAL_ENDPOINT" \
        --local-timeout "$TIMEOUT" \
        --split "$SPLIT" \
        --method debate \
        --limit "$LIMIT" \
        --rounds 1 \
        --retrieval-method bm25_only \
        --memory-mode read_only \
        --output-dir "$OUTPUT_ROOT/memory_read_only"

# ABL-06: memory=read_update (with --update-memory)
run_cmd "memory_read_update" \
    python -m src.main \
        --config "$CONFIG" \
        --run-batch \
        --llm "$LLM" \
        --local-model "$LOCAL_MODEL" \
        --local-endpoint "$LOCAL_ENDPOINT" \
        --local-timeout "$TIMEOUT" \
        --split "$SPLIT" \
        --method debate \
        --limit "$LIMIT" \
        --rounds 1 \
        --retrieval-method bm25_only \
        --memory-mode read_update \
        --update-memory \
        --memory-path memory-bank/ablation_memory_read_update.json \
        --output-dir "$OUTPUT_ROOT/memory_read_update"

# ---------------------------------------------------------------
# Group 3: Feature ablation (ABL-11, ABL-12)
# Base: structured debate r=1, retrieval=bm25_only, memory=off
# ---------------------------------------------------------------

# ABL-11: closing statements off
run_cmd "closing_off" \
    python -m src.main \
        --config "$CONFIG" \
        --run-batch \
        --llm "$LLM" \
        --local-model "$LOCAL_MODEL" \
        --local-endpoint "$LOCAL_ENDPOINT" \
        --local-timeout "$TIMEOUT" \
        --split "$SPLIT" \
        --method debate \
        --limit "$LIMIT" \
        --rounds 1 \
        --retrieval-method bm25_only \
        --memory-mode off \
        --disable-closing-statements \
        --output-dir "$OUTPUT_ROOT/closing_off"

# ABL-12: judge question on
run_cmd "judge_question_on" \
    python -m src.main \
        --config "$CONFIG" \
        --run-batch \
        --llm "$LLM" \
        --local-model "$LOCAL_MODEL" \
        --local-endpoint "$LOCAL_ENDPOINT" \
        --local-timeout "$TIMEOUT" \
        --split "$SPLIT" \
        --method debate \
        --limit "$LIMIT" \
        --rounds 1 \
        --retrieval-method bm25_only \
        --memory-mode off \
        --enable-judge-question \
        --output-dir "$OUTPUT_ROOT/judge_question_on"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "===== ABLATION MATRIX COMPLETE ====="
if $EXECUTE; then
    echo "Collecting metrics from all runs..."
    echo ""
    echo "| Variant | EM | F1 | Fallback | Run dir |"
    echo "|---|---|---|---|---|"
    for variant_dir in "$OUTPUT_ROOT"/{retrieval_off,retrieval_bm25_rerank,memory_read_only,memory_read_update,closing_off,judge_question_on}; do
        variant_name=$(basename "$variant_dir")
        latest_run=$(ls -td "$variant_dir"/*/ 2>/dev/null | head -1)
        if [[ -n "$latest_run" && -f "${latest_run}metrics.json" ]]; then
            metrics=$(python3 -c "
import json, sys
d = json.load(open('${latest_run}metrics.json'))
debate = d.get('metrics_by_method', {}).get('debate', {})
fb = d.get('fallbacks', {}).get('fallback_rate', 'N/A')
print(f\"{debate.get('exact_match','?'):.4f}  {debate.get('f1','?'):.4f}  {fb}\")
" 2>/dev/null || echo "? ? ?")
            echo "| $variant_name | $metrics | $latest_run |"
        else
            echo "| $variant_name | N/A | N/A | N/A | (no metrics found) |"
        fi
    done

    echo ""
    echo "Reference baselines (already run, not re-run):"
    echo "  ABL-02/04  retrieval=bm25_only, memory=off: EM=0.4906 F1=0.8124"
    echo "  ABL-07     rounds=1:                       EM=0.4906 F1=0.8124"
    echo "  ABL-10     judge_off (vanilla):             EM=0.6792 F1=0.9401"
    echo ""
    echo "Logs: $LOG_DIR/"
else
    echo "This was a dry-run. Re-run with --execute to actually run experiments."
fi
