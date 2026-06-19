#!/usr/bin/env bash
# Setup script cho server Linux đã có GPU + Ollama + data.
# Chạy trên server: bash scripts/setup_server.sh
set -euo pipefail

echo "=== [1/6] Kiểm tra tài nguyên server ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "LỖI: Không thấy GPU"; exit 1; }
free -h | head -2
df -h ~ | tail -1

echo ""
echo "=== [2/6] Kiểm tra Ollama ==="
if ! systemctl is-active --quiet ollama; then
    echo "Ollama chưa chạy. Đang start..."
    sudo systemctl start ollama
    sleep 3
fi
curl -s http://localhost:11434/api/tags | grep -q "qwen3.5:9b" || {
    echo "LỖI: Model qwen3.5:9b chưa có. Chạy: ollama pull qwen3.5:9b"
    exit 1
}
echo "OK: Ollama chạy, qwen3.5:9b đã có."

echo ""
echo "=== [3/6] Kiểm tra context length (phải = 8192) ==="
# Trigger 1 request để model load vào RAM
curl -s http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > /dev/null
sleep 2
CTX=$(ollama ps 2>/dev/null | grep -oE '[0-9]+B|262144|8192' | head -1)
echo "Context hiện tại: ${CTX:-không xác định}"
if [[ "${CTX:-}" == "262144" ]]; then
    echo "CẢNH BÁO: Context vẫn 262144 (sẽ ăn hết VRAM). Cần fix:"
    echo "  sudo mkdir -p /etc/systemd/system/ollama.service.d"
    echo "  sudo cp configs/ollama/ollama.service.override.conf.example /etc/systemd/system/ollama.service.d/override.conf"
    echo "  sudo systemctl daemon-reload && sudo systemctl restart ollama"
    echo "  ollama stop qwen3.5:9b"
    exit 1
fi
echo "OK: Context đã cap hợp lý."

echo ""
echo "=== [4/6] Clone/Cập nhật repo + venv ==="
REPO_DIR="$HOME/CourtSimulation"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/JulianNguyen1610/CourtSimulation.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull origin main

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "OK: Dependencies đã cài."

echo ""
echo "=== [5/6] Kiểm tra data ==="
if [ ! -f "data/ALQAC.csv" ]; then
    echo "LỖI: data/ALQAC.csv không tồn tại. Copy từ máy local:"
    echo "  scp d:/Research/data/ALQAC.csv user@server:~/CourtSimulation/data/"
    exit 1
fi
ROWS=$(wc -l < data/ALQAC.csv)
echo "OK: data/ALQAC.csv có $ROWS dòng."

echo ""
echo "=== [6/6] Smoke test (MockLLM, không cần GPU) ==="
python -m src.main --run-batch --llm mock --method both --limit 2 --rounds 1 2>&1 | tail -5
echo "OK: Smoke test pass."

echo ""
echo "========================================"
echo "Setup hoàn tất! Chạy lệnh sau để bắt đầu thí nghiệm chính:"
echo ""
echo "cd ~/CourtSimulation && source venv/bin/activate"
echo "export LOCAL_LLM_REASONING_EFFORT=none"
echo "python -m src.main --config configs/ollama.yaml \\"
echo "    --run-batch --llm local --local-model qwen3.5:9b \\"
echo "    --method both --split validation --limit 0 --rounds 1 \\"
echo "    2>&1 | tee outputs/run_\$(date +%Y%m%d_%H%M).log"
echo "========================================"
