# Tech Context

## Ngôn Ngữ và Framework
- **Python**: Python 3.10+ syntax đang được dùng trong code (`str | None`, `list[...]`).
- **Deep Learning**: PyTorch được khai báo để chạy Hugging Face extractive QA reader.
- **NLP**: Hugging Face Transformers cho baseline `extractive_qa`/`bm25_reader`.
- **Semantic Retrieval**: `sentence-transformers` cho rerank `BAAI/bge-m3` và embedding memory `intfloat/multilingual-e5-large`.
- **Dataset Hub**: `datasets` cho UTS_VLC legal corpus từ Hugging Face.
- **LLM API**: `google-genai` cho Gemini, `openai` cho OpenAI, Ollama qua OpenAI-compatible endpoint (`LocalLLM`).
- **ML/IR**: BM25-lite tự cài trong `src/retrieval/legal_retriever.py`, không cần scikit-learn.
- **Config/Validation**: `pyyaml`, `pydantic`, `pandas`.

## Môi Trường
- **Phát triển**: Local Windows workspace.
- **GPU**: Chưa xác nhận; reader transformer có thể chạy CPU nhưng chậm.
- **RAM**: Chưa xác nhận; cần kiểm tra trước khi chạy reader lớn.

## Dependencies Quan Trọng
- `pandas`, `pydantic`, `pyyaml`, `google-genai`, `openai`
- **Reader fine-tuning (HF Trainer):** `torch>=2.0`, `transformers>=4.36`, `accelerate>=0.21`, `datasets>=2.14`, `sentencepiece>=0.1.99`
- **Retrieval/embeddings:** `sentence-transformers`, `datasets`

### Cài reader training stack trên server
```bash
pip install -U pip
pip install 'torch>=2.0.0' 'transformers>=4.36.0,<5.0.0' 'accelerate>=0.21.0' \
  'datasets>=2.14.0' 'sentencepiece>=0.1.99'
# hoặc: pip install -r requirements.txt
python -c "from src.reader.finetune_reader import check_reader_training_dependencies as c; print(c())"
```

## Ràng Buộc Kỹ Thuật
- API key không được hard-code; dùng `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `LOCAL_LLM_API_KEY`.
- Model thinking trên Ollama (ví dụ `qwen3.5:9b`) trả `content` rỗng nếu không tắt thinking; `LocalLLM` gửi `reasoning_effort: none` mặc định. Override bằng `LOCAL_LLM_REASONING_EFFORT` nếu cần.
- Model/temperature có thể cấu hình trong `configs/default.yaml` và override bằng env (`LLM_MODEL`, `JUDGE_LLM_MODEL`, `LLM_TEMPERATURE`, v.v.).
- `MockLLM` chỉ dùng cho unit test, CI/offline smoke; kết quả nghiên cứu phải chạy LLM thật.
- Reader transformer có thể tải model từ Hugging Face lần đầu; cần quản lý cache và tài nguyên.
- Semantic rerank và embedding memory lazy-load model; chỉ bật khi môi trường sẵn sàng.
- UTS_VLC loader dùng Hugging Face `datasets`; cần internet/cache dataset nếu chưa tải.
- Không tune hyperparameter/prompt trên test split.

## Setup
```powershell
pip install -r requirements.txt
$env:GEMINI_API_KEY = "<set outside code>"
python -m unittest discover -s tests
python -m src.main --run-batch --llm gemini --method both --limit 5 --rounds 1
```

### Ollama (local LLM)

**Context / VRAM:** `qwen3.5:9b` có thể mặc định `num_ctx=262144` (~20GB+ KV). Cap **8192** trên **Ollama server** (giữ nguyên tên model):

```bash
# systemd (khuyên dùng trên Linux server)
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo cp configs/ollama/ollama.service.override.conf.example \
    /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama stop qwen3.5:9b   # unload model loaded with old context

# Verify sau 1 request: ollama ps → CONTEXT 8192 (not 262144)
export LOCAL_LLM_REASONING_EFFORT=none
```

```powershell
python -m src.main --config configs/ollama.yaml --run-batch --llm local --local-model qwen3.5:9b --method both --split validation --limit 0 --rounds 1
```

Override nhanh:
```powershell
python -m src.main --run-batch --llm local --local-model qwen3.5:9b --local-endpoint http://localhost:11434/v1/chat/completions --method direct --limit 2
```

Để chạy offline/CI:

```powershell
python -m src.main --run-batch --llm mock --method both --limit 2 --rounds 1
```
