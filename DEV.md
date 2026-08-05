# Project Mr. X — Dev Startup

## First time only

**Backend environment (FastAPI):**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn sqlalchemy alembic python-dotenv requests
```

**ML environment (conda):**
```bash
conda activate bird-classifier-warmup
cd ml/wildlife-monitor
pip install -r requirements.txt --break-system-packages
```
(Conda env is shared across warm-up and wildlife-monitor ML work — not separate from the backend venv.)

**Database setup (SQLite, one-time):**
```bash
cd backend
venv\Scripts\activate
alembic upgrade head
python -m app.db.seed_species
```

**Environment variables:**
Create `backend/.env` (gitignored, never commit this):


---

## Every session

**Terminal 1 — Backend API:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```
Runs at `http://127.0.0.1:8000` — interactive docs at `/docs`.

**Terminal 2 — ML scripts (as needed, not always running):**
```bash
conda activate bird-classifier-warmup
cd D:\projects\PycharmProjects\bird-classifier-warmup
python ml/wildlife-monitor/scripts/<script_name>.py
```
Run from repo root, not from inside `scripts/` — several scripts use relative paths assuming this.

**Frontend:** not built yet (Topic 13, not started).

**Real camera / continuous detection loop:** not built yet (Topic 15, not started). Current ML scripts (`pipeline_test.py`, `video_test.py`, `webcam_live_test.py`, `run_detection_and_log.py`) are manual/one-off, not a running service.

---

## Known gotchas

- If `uvicorn --reload` seems stale (endpoint exists in code but not in `/docs`), check `http://127.0.0.1:8000/openapi.json` directly — sometimes a zombie process survives a reload. Kill it explicitly: `taskkill /IM python.exe /F`, then restart clean.
- Any new SQLAlchemy model must be imported in `app/db/models/__init__.py`, or foreign keys silently fail to resolve.
- Scripts run from the wrong working directory will create stray folders/files at repo root instead of the correct `ml/wildlife-monitor/...` path — always run from repo root unless otherwise noted.