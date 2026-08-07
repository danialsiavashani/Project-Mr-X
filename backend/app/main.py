from fastapi import FastAPI
from app.api.routes import detections, species, analytics, auth, stream

app = FastAPI(title="Project Mr. X API")

app.include_router(detections.router)
app.include_router(species.router)
app.include_router(analytics.router)
app.include_router(auth.router)
app.include_router(stream.router)


@app.get("/")
def root():
    return {"status": "Project Mr. X backend is running"}