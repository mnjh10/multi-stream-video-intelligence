from fastapi import FastAPI

app = FastAPI(title="ARGUS Backend")


@app.get("/health")
def health_check():
    return {"status": "ok"}