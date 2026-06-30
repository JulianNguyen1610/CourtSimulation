# Test split one-shot — project primary config (judge_mediated).
#
# Usage:
#   .\scripts\run_test_judge_mediated.ps1 -Execute
#   .\scripts\run_test_judge_mediated.ps1 -Execute -Limit 2   # smoke
#   .\scripts\run_test_judge_mediated.ps1 -Execute -Llm mock -Limit 2

param(
    [switch]$Execute,
    [int]$Limit = 0,
    [string]$Config = "configs/ollama.yaml",
    [ValidateSet("mock", "gemini", "openai", "local")]
    [string]$Llm = "local",
    [string]$OutputDir = "outputs/test_metrics/judge_mediated_test"
)

$ErrorActionPreference = "Stop"
$env:LOCAL_LLM_REASONING_EFFORT = if ($env:LOCAL_LLM_REASONING_EFFORT) { $env:LOCAL_LLM_REASONING_EFFORT } else { "none" }

$argsList = @(
    "-m", "src.main",
    "--config", $Config,
    "--run-batch",
    "--llm", $Llm,
    "--split", "test",
    "--method", "debate",
    "--limit", "$Limit",
    "--rounds", "1",
    "--retrieval-method", "off",
    "--memory-mode", "read_only",
    "--orchestrator", "judge_mediated",
    "--output-dir", $OutputDir,
    "--save-debate-artifacts"
)

if ($Llm -eq "local") {
    $argsList += @(
        "--local-model", "qwen3.5:9b",
        "--local-endpoint", "http://localhost:11434/v1/chat/completions",
        "--local-timeout", "1200"
    )
}

Write-Host "Test one-shot: judge_mediated | split=test | limit=$Limit"
Write-Host "Command: python $($argsList -join ' ')"

if ($Execute) {
    python @argsList
    Write-Host "`nDone. Check: $OutputDir"
} else {
    Write-Host "(dry-run — add -Execute)"
}
