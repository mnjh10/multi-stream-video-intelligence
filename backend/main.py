from fastapi import FastAPI

from backend.api.routes_evidence import router as evidence_router
from backend.api.routes_memory import router as memory_router
from backend.api.routes_query import router as query_router
from backend.db.models import Base
from backend.db.session import engine

# Ensure database tables exist upon application startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ARGUS Backend",
    description="ARGUS Conversational Multi-Camera Video Intelligence API",
    version="1.0.0",
)

app.include_router(query_router)
app.include_router(memory_router)
app.include_router(evidence_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}