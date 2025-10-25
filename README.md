# Sentiment LoRA FastAPI

Lightweight deployment of James Boggs' quant-grade sentiment model using
4-bit quantization + LoRA adapters on PyTorch 2.6 (CUDA 12.4).

## 🚀 Local Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 10000
