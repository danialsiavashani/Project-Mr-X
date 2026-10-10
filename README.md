# Project Mr. X

**A full-stack backyard wildlife monitoring system**

> "At first I just wanted to know which birds visit my backyard. Then I wanted to know when. Then I wanted to know if the same one keeps coming back."

## What is this?

A personal computer vision project built out of curiosity: watch a backyard, identify the animals that visit, log every visit, and eventually predict behavior. The name comes from the long-term goal: recognizing one specific individual animal (Mr. X) and learning when he shows up.

This is not a tutorial project. It is a real system, built from foundations up, designed to run unattended on a Raspberry Pi 5 with an ArduCam, with the dashboard and API hosted in the cloud.

**Status in one line:** the ML pipeline, backend, live-stream relay, and analytics integration work end-to-end locally. Cloud deployment, the Next.js dashboard, and Pi hardware are not done yet. See [Current Status](#current-status) for the exact breakdown.

## How it works

```
ArduCam (USB)
  -> SharedCamera            single owner of the camera device (background thread, latest-frame buffer)
  -> MegaDetector            finds animals; person/vehicle boxes are skipped
  -> padded crop
  -> EfficientNet-B0         classifies species (15 classes)
  -> confidence threshold    below 0.65 -> logged as "unknown", not stored
  -> visit dedup             30-second window per species, so one visit = one event
  -> FastAPI                 stores the detection (species, confidence, crop path, timestamp, source)
  -> BehaviorPulse           behavioral analytics on the stored detections
  -> Next.js dashboard       live view + event history + analytics (planned)
```

### Deployment design

| Where | What runs |
|---|---|
| **Raspberry Pi 5** (local) | Camera capture, MegaDetector + EfficientNet inference, detection logging. Makes outbound requests only and is never exposed to the internet. |
| **Render** | FastAPI backend, database, live-stream relay |
| **Vercel** | Next.js dashboard |

The camera can only be opened by one process, so capture, detection, and stream pushing all live in a single Pi-side script that shares one `SharedCamera`. The cloud side never touches camera hardware.

### Live view

On-demand MJPEG, deliberately low-cost:

- The Pi pushes JPEG frames to the backend, which relays them to viewers.
- The viewing page sends a heartbeat every second. The Pi checks `should-stream` every 2 seconds and only pushes frames while someone is actually watching, so idle bandwidth is near zero.
- MJPEG was chosen because the Pi 5 has no hardware video encoder; avoiding a video codec avoids competing with inference for CPU.

### BehaviorPulse integration

[BehaviorPulse](https://behaviorpulse.com) is a separate behavioral analytics SaaS I built. Mr. X is one of its consumers: each stored detection maps to a BehaviorPulse observation (`observed_at`, `subject`, `source`, `confidence`), and the backend calls BehaviorPulse to generate pattern reports, which are saved locally as `AnalysisReport` rows (summary, prediction, pattern table, confidence).

- Validated end-to-end against 90 days of **synthetic** detections: one species with a deliberate weekly pattern, two with deliberately patternless noise, to check that the analytics report real patterns and don't invent fake ones.
- Integration surfaced real bugs on both sides, including SQLite silently dropping timezone info and a logging gap in BehaviorPulse's own audit trail, both fixed.
- Known v1 limitation: unfiltered queries mix species together, so reports are generated per species.
- No real camera data has been analyzed yet. Real patterns require weeks of real detections.

## The classifier

- **Detector:** MegaDetector `MDV6-yolov10-e`, downloaded from Zenodo (it is not in the Ultralytics registry).
- **Classifier:** EfficientNet-B0, two-stage fine-tuning (frozen backbone, then low-LR unfreeze). Best validation accuracy **91.89%** on a held-out validation split of 15 classes.
- **Classes (12 birds + 3 other):** American Crow, Anna's Hummingbird, Black Phoebe, Bushtit, California Quail, California Scrub-Jay, Eurasian Collared-Dove, House Finch, Mourning Dove, Northern Mockingbird, Song Sparrow, White-crowned Sparrow, plus Cat, Dog, Squirrel.
- **Data:** CUB-200-2011, iNaturalist (California-filtered), Kaggle Animals-10, Kaggle Cats vs Dogs.

### Design decisions worth knowing

- **Why MegaDetector, not COCO YOLOv8:** a COCO-trained YOLOv8 could not detect squirrels or deer and mislabeled a dog as a horse. That drove the pivot to MegaDetector, which outputs a generic "animal" box, and a single combined EfficientNet classifier downstream. The originally planned per-species model router was dropped.
- **Reproducible splits:** a shared `random.seed()` across classes silently shifted train/val splits whenever one class changed. Fixed with a per-class `random.Random(SEED)`, and `data/processed/` is wiped before each resplit.
- **Augmentation tried and reverted:** GaussianBlur and RandomResizedCrop were tested to fix blur-induced errors and made confident wrong answers worse, so they were removed.
- **Crop saving is caller-controlled:** the pipeline returns the crop; the live loop writes it to disk only on a genuine new visit, instead of saving a near-identical file every 2 seconds.
- **Dedup in the real write path:** visit deduplication is applied in the production loop, not only in test scripts.

### Known limitations

- **Distant or blurry subjects can be confidently misclassified** (for example, a distant cat labeled dog). The confidence threshold and a minimum-crop-size filter do not catch these. Fixing it needs distance-realistic training data from the real camera.
- Some bird classes (for example House Finch) have relatively thin training data.
- Live-view presence uses polling, so a backgrounded browser tab can pause the stream.

## Auth

- JWT access + refresh tokens (short-lived access token, longer-lived refresh token, `type` claim to block token misuse).
- Two fixed identities, `admin` and `viewer`, configured via a gitignored `.env`. No users table, deliberately: this is a single-tenant personal system.
- Route protection for the API and stream endpoints is the next step (see below).

## Stack

| Layer | Technology |
|---|---|
| Detection | MegaDetector (YOLOv10-based) via Ultralytics |
| Classification | EfficientNet-B0 (PyTorch, fine-tuned) |
| Backend | FastAPI, SQLAlchemy, Alembic |
| Database | SQLite locally; PostgreSQL planned for Render |
| Analytics | BehaviorPulse (own SaaS) |
| Frontend | Next.js (planned) |
| Hosting | Render (backend), Vercel (frontend) |
| Hardware | Raspberry Pi 5 + ArduCam USB camera (Pi not yet purchased) |
| Training | PyTorch on a local NVIDIA GTX 1060 |

## Repository layout

```
Project-Mr-X/
  ml/
    bird-classifier-warmup/   CNN foundations and training pipeline
    wildlife-monitor/         detection, classification, camera, and the live pipeline
  backend/                    FastAPI app (routes, models, migrations, auth, stream relay)
  frontend/                   Next.js dashboard (not started)
  docs/
```

## Why a baseline CNN?

Before fine-tuning EfficientNet, I built a baseline CNN from scratch to understand what transfer learning is replacing: custom architecture in PyTorch, a full training pipeline (data loading, augmentation, training loop, evaluation, checkpointing), confusion matrices and per-class error analysis, and sample predictions with visual output. The baseline is not the production model; it is the foundation that makes EfficientNet meaningful.

## The vision: three phases

**Phase 1: Detect, classify, analyze.** The Pi watches the backyard, every visit is logged with species, confidence, timestamp, and crop image, and a dashboard shows a live view and scrollable history, a Ring-doorbell experience for wildlife. Analytics: visit frequency and time-of-day/day-of-week patterns per species, with seasonal trends once enough data exists.

**Phase 2: Your backyard, your dataset.** After months of collection, retrain on real crops from this camera, lighting, and species mix. This is also the real fix for the distance/blur misclassification above.

**Phase 3: Meet Mr. X.** Individual re-identification for animals with distinguishing features (a scar, unusual plumage, asymmetric markings): "was that the same Blue Jay as last Wednesday?" This needs manually labeled crops of the same individual and is research-shaped, not a sure thing.

## Current Status

**Done and working locally**
- [x] Baseline CNN with full modular training pipeline, GPU training verified
- [x] Confusion matrix, per-class evaluation, sample predictions, error analysis
- [x] EfficientNet-B0 fine-tuning, 15 classes (91.89% validation accuracy)
- [x] MegaDetector integration, detect -> crop -> classify pipeline
- [x] FastAPI backend: detections, species, analytics endpoints, Alembic migrations
- [x] BehaviorPulse integration, validated on synthetic data
- [x] Continuous live pipeline: one shared camera, detection, visit dedup, gated streaming
- [x] Live-stream relay with heartbeat-gated bandwidth
- [x] JWT login and refresh endpoints

**Not done yet**
- [ ] Route protection on API and stream endpoints
- [ ] Stream-token auth for the live view
- [ ] Next.js dashboard (live view, event history, analytics)
- [ ] Migrate SQLite to PostgreSQL; backend dependency file
- [ ] Deploy backend to Render and frontend to Vercel
- [ ] Raspberry Pi 5 hardware setup and on-device performance testing
- [ ] Phase 1 real data collection
- [ ] Phase 2: backyard dataset training
- [ ] Phase 3: individual re-identification (Mr. X)

## Author

Danial Bahrami Siavashani, CS Graduate Student, CSU Chico. [GitHub](https://github.com/danialsiavashani)
