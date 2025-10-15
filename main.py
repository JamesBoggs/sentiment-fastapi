from fastapi import FastAPI
from ops_instrumentation import attach_ops

app = FastAPI(title="sentiment")
attach_ops(app)

@app.get("/")
def root():
    return {"ok": True, "msg": "hello"}
