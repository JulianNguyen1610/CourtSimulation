# Fixed vs judge-mediated orchestrator ablation (Windows).
#
# Usage:
#   .\scripts\run_orchestrator_ablation.ps1           # dry-run
#   .\scripts\run_orchestrator_ablation.ps1 -Execute  # run (needs Ollama or set -Llm gemini)

param(
    [switch]$Execute,
    [string]$Config = "configs/ollama.yaml",
    [ValidateSet("mock", "gemini", "openai", "local")]
    [string]$Llm = "local",
    [string]$Split = "validation",
    [int]$Limit = 0,
    [string]$OutputRoot = "outputs/orchestrator_ablation"
)

$ErrorActionPreference = "Stop"
$env:LOCAL_LLM_REASONING_EFFORT = if ($env:LOCAL_LLM_REASONING_EFFORT) { $env:LOCAL_LLM_REASONING_EFFORT } else { "none" }

$argsList = @(
    "scripts/run_orchestrator_ablation.py",
    "--config", $Config,
    "--llm", $Llm,
    "--split", $Split,
    "--limit", "$Limit",
    "--rounds", "1",
    "--retrieval-method", "off",
    "--memory-mode", "read_only",
    "--output-root", $OutputRoot,
    "--continue-on-error"
)

if ($Llm -eq "local") {
    $argsList += @(
        "--local-model", "qwen3.5:9b",
        "--local-endpoint", "http://localhost:11434/v1/chat/completions",
        "--local-timeout", "1200"
    )
}

if ($Execute) {
    $argsList += "--execute"
}

Write-Host "Command: python $($argsList -join ' ')"
python @argsList
