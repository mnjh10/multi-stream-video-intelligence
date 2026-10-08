from fastapi import FastAPI

from backend.api.routes_evidence import router as evidence_router
from backend.api.routes_memory import router as memory_router
from backend.api.routes_query import router as query_router
from backend.api.routes_cameras import router as cameras_router


app = FastAPI(title="ARGUS Backend")

app.include_router(query_router)
app.include_router(memory_router)
app.include_router(evidence_router)
app.include_router(cameras_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}