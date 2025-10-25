from fastapi import FastAPI, Request
from app.model_loader import get_model
from app.predict import predict_sentiment
import time

app = FastAPI(title="Sentiment LoRA API", version="v2.0")

model, tokenizer = get_model()

@app.get("/")
def root():
    return {"status": "online", "version": "v2.0", "framework": "PyTorch 2.6.0+cu124"}

@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    text = body.get("text")
    if not text:
        return {"error": "missing 'text' field"}

    t0 = time.time()
    result = predict_sentiment(model, tokenizer, text)
    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "ok",
        "latency_ms": latency_ms,
        "result": result,
        "framework": "PyTorch 2.6.0+cu124",
        "adapter": "LoRA (4-bit quantized)"
    }
