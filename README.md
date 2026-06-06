# Project Mr. X
### A Full-Stack Backyard Wildlife Monitoring System

> *"At first I just wanted to know which birds visit my backyard. Then I wanted to know when. Then I wanted to know if the same one keeps coming back."*

---

## What is this?

A personal computer vision project built out of genuine curiosity — to watch, classify, and eventually predict the behavior of wildlife visiting my backyard. The name comes from the long-term goal: to identify and track a specific individual animal (Mr. X) and predict their behavior patterns over time.

This is not a tutorial project. It is a real system designed in three phases, built from foundations up, running 24/7 on a Raspberry Pi 5 in my backyard.

---

## The Vision — Three Phases

### Phase 1 — Detect, Classify, Analyze
A Raspberry Pi 5 with an Arducam watches the backyard 24/7. YOLOv8 (pretrained on COCO) detects animals in each frame. Specialist EfficientNet models classify the species. Every detection is logged with species, confidence, timestamp, and crop image. A Next.js dashboard shows live detections and scrollable event history — think Ring doorbell, but for wildlife.

**Behavior analysis at this phase:**
- Which species visit and how often
- Time-of-day and seasonal patterns per species
- Population trends over weeks and months
- Weather and environmental correlations
- Anomaly detection — unusual visits or sudden disappearances

### Phase 2 — Your Backyard, Your Dataset
After months of data collection, the generic Kaggle-trained EfficientNet models are replaced with models trained on real crops from this specific backyard. Same camera, same lighting, same species that actually show up. Higher accuracy, less noise, more trustworthy behavior analysis.

**What changes:**
- Cleaner classifications mean more reliable behavioral patterns
- Species-specific profiles become more precise
- The system gets better at recognizing *your* birds, not just birds in general

### Phase 3 — Meet Mr. X
After enough data is collected, individual re-identification becomes possible for animals with distinguishing features — a scar, unusual plumage, asymmetric markings, or any visually unique trait.

Manually labeled crops of the same individual across multiple visits train a re-ID model. Now the system can ask not just *"was that a Blue Jay?"* but *"was that the same Blue Jay as last Wednesday?"*

**Behavior analysis at this phase:**
- Individual visit frequency and schedule prediction
- Territorial patterns per individual
- Long-term behavioral change detection
- "Mr. X is likely to visit around 3 PM on Wednesday"

---

## Why a Baseline CNN?

Before fine-tuning EfficientNet, I built a baseline CNN from scratch to deeply understand what transfer learning is actually replacing. This covers:

- Custom CNN architecture in PyTorch — every layer, every tensor shape, every parameter
- Full training pipeline — data loading, augmentation, training loop, evaluation, checkpointing
- Confusion matrix and per-class error analysis
- Sample predictions with visual output
- GPU training verified on NVIDIA GTX 1060 (CUDA 12.4)

The baseline CNN is not the production model. It is the foundation that makes EfficientNet meaningful.

---

## Architecture

```
Camera frame
→ YOLOv8 detection (pretrained, COCO)
→ filter: bird / cat / squirrel / other wildlife
→ model router
→ specialist EfficientNet classifier (one per species group)
→ detection event logged (crop, species, confidence, timestamp)
→ FastAPI backend
→ Next.js dashboard (live feed + event history + analytics)
```

**Design principle — separation of concerns:**

| Component | Role |
|---|---|
| YOLO | Finder — draws the box |
| EfficientNet | Identifier — names the species |
| Model Router | Dispatcher — sends crop to correct specialist |
| FastAPI | Orchestrator — inference, storage, API |
| Next.js | Product layer — live dashboard |
| SQLite → PostgreSQL | Evidence — every detection logged |

---

## Stack

| Layer | Technology |
|---|---|
| Detection | YOLOv8 (Ultralytics) |
| Classification | EfficientNet (PyTorch, fine-tuned) |
| Backend | FastAPI |
| Frontend | Next.js |
| Database | SQLite → PostgreSQL |
| Hardware | Raspberry Pi 5 + Arducam |
| Deployment | Docker |
| Training | PyTorch, CUDA 12.4 |

---

## Project Structure

```
bird-classifier-warmup/     ← CNN foundations and training pipeline (this repo)
wildlife-monitor/           ← full-stack capstone system (coming)
```

---

## Current Status

- [x] Baseline CNN — full modular training pipeline
- [x] GPU training verified (GTX 1060, CUDA 12.4)
- [x] Confusion matrix and per-class evaluation
- [x] Sample predictions with visual output
- [x] Error analysis — misclassified image inspection
- [ ] EfficientNet fine-tuning (birds, cats, squirrels)
- [ ] YOLO detection integration
- [ ] Model router
- [ ] FastAPI backend
- [ ] Next.js live dashboard
- [ ] Raspberry Pi 5 deployment
- [ ] Phase 1 — data collection begins
- [ ] Phase 2 — backyard dataset training
- [ ] Phase 3 — individual re-identification (Mr. X)

---

## Author

**Danial Bahrami Siavashani**
CS Graduate Student
[GitHub](https://github.com/danialsiavashani)
