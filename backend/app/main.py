from fastapi import FastAPI
from app.api.routes import detections, species, analytics

app = FastAPI(title="Project Mr. X API")

app.include_router(detections.router)
app.include_router(species.router)
app.include_router(analytics.router)


@app.get("/")
def root():
    return {"status": "Project Mr. X backend is running"}