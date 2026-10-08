from fastapi import FastAPI

from backend.api.routes_memory import router as memory_router
from backend.api.routes_query import router as query_router


app = FastAPI(title="ARGUS Backend")

app.include_router(query_router)
app.include_router(memory_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}