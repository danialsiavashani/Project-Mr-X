from fastapi import FastAPI
from app.api.routes import detections

app = FastAPI(title="Project Mr. X API")

app.include_router(detections.router)


@app.get("/")
def root():
    return {"status": "Project Mr. X backend is running"}