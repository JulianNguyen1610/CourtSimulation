#!/usr/bin/env bash
# Cap Ollama runtime context to 8192 tokens (keep model name qwen3.5:9b unchanged).
#
# qwen3.5:9b defaults to num_ctx=262144 unless the Ollama *server* starts with
# OLLAMA_CONTEXT_LENGTH=8192. Shell exports alone do not affect a running daemon.
set -euo pipefail

echo "==> Stop loaded models (if any)..."
ollama ps -q | xargs -r ollama stop || true

echo ""
echo "Set server env (pick ONE):"
echo ""
echo "  A) systemd (persistent, recommended on Linux server):"
echo "     sudo mkdir -p /etc/systemd/system/ollama.service.d"
echo "     sudo cp configs/ollama/ollama.service.override.conf.example \\"
echo "         /etc/systemd/system/ollama.service.d/override.conf"
echo "     sudo systemctl daemon-reload && sudo systemctl restart ollama"
echo ""
echo "  B) manual serve in this terminal:"
echo "     export OLLAMA_CONTEXT_LENGTH=8192"
echo "     export OLLAMA_NUM_PARALLEL=1"
echo "     export OLLAMA_MAX_LOADED_MODELS=1"
echo "     ollama serve"
echo ""
read -r -p "Press Enter after Ollama server was restarted with OLLAMA_CONTEXT_LENGTH=8192..."

MODEL="qwen3.5:9b"
echo "==> Warm-up ${MODEL}..."
curl -sf http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}" \
  >/dev/null

echo "==> Runtime check (expect CONTEXT 8192, not 262144):"
ollama ps

echo ""
echo "Experiments (unchanged model name):"
echo "  export LOCAL_LLM_REASONING_EFFORT=none"
echo "  python -m src.main --config configs/ollama.yaml --run-batch --llm local \\"
echo "    --local-model ${MODEL} --method both --split validation --limit 0 --rounds 1"
