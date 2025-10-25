from fastapi import HTTPException
import torch
from torch.nn.functional import softmax

def predict_sentiment(model, tokenizer, text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=-1).item()
        score = probs[0][sentiment].item()

        label_map = {0: "negative", 1: "positive"}
        return {"sentiment": label_map.get(sentiment, "unknown"), "score": round(score, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
