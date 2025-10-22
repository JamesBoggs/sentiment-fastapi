# quantize.py — One-time FinBERT compressor for Render Free tier
import torch
from transformers import AutoModelForSequenceClassification

MODEL_PATH = "models/sentiment.pt"
OUT_PATH = "models/sentiment_quant.pt"
BASE = "yiyanghkust/finbert-tone"

print("⏳ Loading original model...")
base_model = AutoModelForSequenceClassification.from_pretrained(BASE)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
base_model.load_state_dict(state_dict)

print("✅ Loaded. Quantizing...")
quantized_model = torch.quantization.quantize_dynamic(
    base_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print(f"💾 Saving quantized model to: {OUT_PATH}")
torch.save(quantized_model.state_dict(), OUT_PATH)

# Optional: show size
import os
size = os.path.getsize(OUT_PATH) / (1024**2)
print(f"✅ Done. File size: {size:.2f} MB")
