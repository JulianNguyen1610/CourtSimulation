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
- `pandas`
- `pydantic`
- `pyyaml`
- `google-genai`
- `openai`
- `transformers`
- `torch`
- `sentence-transformers`
- `datasets`

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
Đảm bảo Ollama đang chạy (`ollama serve` hoặc app Ollama). Kiểm tra model: `ollama list`.

```powershell
python -m src.main --config configs/ollama.yaml --run-batch --llm local --method direct --limit 5
python -m src.main --config configs/ollama.yaml --run-debate --llm local --local-model qwen3.5:9b --rounds 1
```

Override nhanh không cần sửa YAML:
```powershell
python -m src.main --run-batch --llm local --local-model dolphin3:latest --local-endpoint http://localhost:11434/v1/chat/completions --method direct --limit 2
```

Để chạy offline/CI:

```powershell
python -m src.main --run-batch --llm mock --method both --limit 2 --rounds 1
```
