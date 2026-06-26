# Run pending P1 ablation variants (retrieval, memory, features) on Windows/Linux with Ollama.
#
# Usage:
#   .\scripts\run_p1_ablations.ps1              # dry-run
#   .\scripts\run_p1_ablations.ps1 -Execute     # run all pending variants (validation 53)
#
# Prerequisites: Ollama + qwen3.5:9b, LOCAL_LLM_REASONING_EFFORT=none

param(
    [switch]$Execute,
    [int]$Limit = 0,
    [string]$Config = "configs/ollama.yaml",
    [string]$OutputRoot = "outputs/p1_ablation_matrix"
)

$ErrorActionPreference = "Stop"
$env:LOCAL_LLM_REASONING_EFFORT = if ($env:LOCAL_LLM_REASONING_EFFORT) { $env:LOCAL_LLM_REASONING_EFFORT } else { "none" }

$argsList = @(
    "scripts/run_ablation_matrix.py",
    "--config", $Config,
    "--llm", "local",
    "--local-model", "qwen3.5:9b",
    "--local-endpoint", "http://localhost:11434/v1/chat/completions",
    "--local-timeout", "1200",
    "--split", "validation",
    "--limit", "$Limit",
    "--include-heavy-rerank",
    "--pending-only",
    "--groups", "retrieval,memory,features",
    "--summary-csv", "docs/experiments/p1_ablation_pending_results.csv"
)

if ($Execute) {
    $argsList += "--execute"
}

Write-Host "Command: python $($argsList -join ' ')"
python @argsList
