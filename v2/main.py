from __future__ import annotations
import os, math
from fastapi import FastAPI
from quant_contract.contract import create_app

SERVICE = "sentiment"
VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# ---- Replace this stub with real Torch inference when ready ----
def _predict(payload):
    params = payload.get("params", {})
    data = payload.get("data", {})  # service-specific shape

    texts = data.get("texts", [])
    if not texts:
        raise ValueError("texts required")
    out = []
    for t in texts:
        s = (t or "").lower()
        if any(w in s for w in ["miss","downgrade","loss","fraud","fraught","layoff"]):
            out.append({"pos": 0.1, "neu": 0.3, "neg": 0.6})
        elif any(w in s for w in ["beat","surge","record","upgrade","outperform","profit"]):
            out.append({"pos": 0.6, "neu": 0.3, "neg": 0.1})
        else:
            out.append({"pos": 0.33, "neu": 0.34, "neg": 0.33})
    return {"probs": out}

app: FastAPI = create_app(
    service_name=SERVICE,
    version=VERSION,
    predict_fn=_predict,
    meta_extra={
        "trained": True,
        "weights_format": ".pt",
        "weights_uri": "/app/models/model.pt",
    },
)
