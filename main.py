# main.py — Sentiment Analysis FastAPI (FinBERT / transformer-based)
# ================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch, time, numpy as np, random

# === App setup ===
app = FastAPI(title="Sentiment Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model ===
MODEL_PATH = "models/sentiment.pt"
try:
    model = torch.load(MODEL_PATH, map_location="cpu")
    is_online = True
except Exception as e:
    model, is_online = None, False
    print(f"[WARN] Could not load sentiment.pt: {e}")

# === Meta endpoint ===
@app.get("/meta")
def get_meta():
    start = time.time()
    metrics = []

    if is_online:
        # Simulate aggregated sentiment metrics
        days = np.arange(0, 14)
        base = np.random.normal(0.6, 0.1, len(days))
        pos = np.clip(base + np.random.normal(0.05, 0.03, len(days)), 0, 1)
        neg = np.clip(1 - pos, 0, 1)
        neutral = 1 - (pos + neg) / 2
        confidence = np.random.uniform(0.7, 0.95, len(days))
        greed_index = pos - neg

        metrics.extend([
            {"id": "pos_ratio", "label": "Positive Ratio", "status": "online",
             "chartData": [{"x": int(i), "y": float(v)} for i, v in enumerate(pos)]},
            {"id": "neg_ratio", "label": "Negative Ratio", "status": "online",
             "chartData": [{"x": int(i), "y": float(v)} for i, v in enumerate(neg)]},
            {"id": "neutral_ratio", "label": "Neutral Ratio", "status": "online",
             "chartData": [{"x": int(i), "y": float(v)} for i, v in enumerate(neutral)]},
            {"id": "rolling_sentiment", "label": "Rolling Sentiment Index", "status": "online",
             "chartData": [{"x": int(i), "y": float(v)} for i, v in enumerate(base)]},
            {"id": "confidence", "label": "Model Confidence", "status": "online",
             "chartData": [{"x": int(i), "y": float(v)} for i, v in enumerate(confidence)]},
            {"id": "fear_greed", "label": "Fear / Greed Index", "status": "online",
             "chartData": [{"x": int(i), "y": float(v)} for i, v in enumerate(greed_index)]},
        ])

    latency = int((time.time() - start) * 1000)
    return {
        "name": "FinBERT Sentiment",
        "description": "Financial sentiment model analyzing market tone and text data.",
        "isOnline": is_online,
        "latency": latency,
        "metrics": metrics,
        "data": {"weights_loaded": bool(is_online)},
    }

# === Root endpoint ===
@app.get("/")
def root():
    return {"status": "sentiment-fastapi running", "model_loaded": is_online}
