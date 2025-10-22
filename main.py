from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, torch, json, gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === App Setup ===
app = FastAPI(title="Sentiment API", version="1.0.0")

# === File Paths ===
WEIGHT_PATH = "models/sentiment.pt"
META_PATH = "models/sentiment_meta.json"
GDRIVE_ID = "1iJ5CPcl4oNZ5Q8GD20QSDoVl2ROhNb7V"

# === Download Model if Missing ===
os.makedirs("models", exist_ok=True)
if not os.path.exists(WEIGHT_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_ID}",
        WEIGHT_PATH,
        quiet=False
    )

# === Load Tokenizer + Base ===
base_model = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(base_model)

# === Load Classification Head ===
model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
model.eval()

# === Load Meta (for label mapping) ===
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
