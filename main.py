import os

# === Quantize once, then stop to prevent OOM ===
if not os.path.exists("models/sentiment_quant.pt"):
    import quantize
    exit()  # prevent large model load on this run

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === App Setup ===
app = FastAPI(title="Sentiment API", version="1.0.0")

# === Load Tokenizer + Base ===
base_model = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(base_model)

# === Load Quantized State ===
state = torch.load("models/sentiment_quant.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# === Meta (optional) ===
META_PATH = "models/sentiment_meta.json"
try:
    with open(META_PATH) as f:
        meta = json.load(f)
        id2label = meta.get("id2label", {0: "negative", 1: "neutral", 2: "positive"})
except:
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

# === Inference Schema ===
class SentimentRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True, "model": "FinBERT", "version": "1.0.0"}

@app.post("/score")
def score(req: SentimentRequest):
    try:
        enc = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
        return {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
