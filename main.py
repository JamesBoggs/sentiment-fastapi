from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from meta import router as meta_router
from ops_instrumentation import attach_ops

app = FastAPI(title="sentiment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(meta_router)
attach_ops(app)

@app.get("/")
def root():
    return {"ok": True, "msg": "hello"}

@app.get("/health")
def health():
    return {"ok": True}
