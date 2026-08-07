import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.db.base import SessionLocal
from app.db.models.species import Species
from app.db.models.detection import Detection

random.seed(42)

db = SessionLocal()

species_by_name = {s.name: s.id for s in db.query(Species).all()}

now = datetime.now(timezone.utc)
DAYS_BACK = 90

PLACEHOLDER_CROP = "ml/wildlife-monitor/outputs/crops/synthetic_seed.jpg"
PACIFIC = ZoneInfo("America/Los_Angeles")

new_detections = []


def local_time(day_utc, hour, minute):
    local_dt = day_utc.astimezone(PACIFIC).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return local_dt.astimezone(timezone.utc)


def add_detection(species_name, dt, confidence):
    species_id = species_by_name.get(species_name)
    if species_id is None:
        print(f"WARNING: species '{species_name}' not found, skipping")
        return
    new_detections.append(Detection(
        species_id=species_id,
        confidence=round(confidence, 3),
        crop_path=PLACEHOLDER_CROP,
        detector_confidence=round(random.uniform(0.6, 0.95), 3),
        model_version="MDV6-yolov10-e",
        source="backyard_cam_1",
        timestamp=dt,
    ))


for day_offset in range(DAYS_BACK):
    day = now - timedelta(days=day_offset)
    day_pacific_weekday = day.astimezone(PACIFIC).weekday()

    # Squirrel: moderately common, tight midday window — a real, distinct pattern
    if random.random() < 0.55:
        hour = random.randint(10, 14)
        minute = random.randint(0, 59)
        add_detection("Squirrel", local_time(day, hour, minute), random.uniform(0.75, 0.98))

    # Anna's Hummingbird: bimodal — dawn AND dusk feeding, biologically realistic,
    # each visit independent so some days get one, both, or neither
    if random.random() < 0.45:
        hour = random.randint(6, 8)
        minute = random.randint(0, 59)
        add_detection("Anna's Hummingbird", local_time(day, hour, minute), random.uniform(0.8, 0.99))
    if random.random() < 0.35:
        hour = random.randint(17, 19)
        minute = random.randint(0, 59)
        add_detection("Anna's Hummingbird", local_time(day, hour, minute), random.uniform(0.8, 0.99))

    # Cat: the ONE deliberately engineered weekly pattern — Wed/Sat evenings only
    if day_pacific_weekday in (2, 5) and random.random() < 0.75:
        hour = random.randint(18, 20)
        minute = random.randint(0, 59)
        add_detection("Cat", local_time(day, hour, minute), random.uniform(0.7, 0.95))

    # --- Pure noise species: NO time-of-day or day-of-week structure at all ---
    # These test whether the analytics engine correctly avoids inventing
    # a fake pattern where none exists.
    if random.random() < 0.15:
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        add_detection("Dog", local_time(day, hour, minute), random.uniform(0.65, 0.95))
    if random.random() < 0.30:
        hour = random.randint(5, 20)
        minute = random.randint(0, 59)
        add_detection("American Crow", local_time(day, hour, minute), random.uniform(0.7, 0.97))

    # Morning-only species, moderate & varied frequency — real but looser patterns
    morning_species = {
        "Mourning Dove": 0.30,
        "Song Sparrow": 0.25,
        "White-crowned Sparrow": 0.20,
        "House Finch": 0.30,
        "Northern Mockingbird": 0.22,
    }
    for species_name, prob in morning_species.items():
        if random.random() < prob:
            hour = random.randint(6, 10)
            minute = random.randint(0, 59)
            add_detection(species_name, local_time(day, hour, minute), random.uniform(0.7, 0.97))

    # Rare species — genuinely infrequent, scattered throughout the day
    rare_species = [
        "Black Phoebe", "Bushtit", "California Quail",
        "California Scrub-Jay", "Eurasian Collared-Dove",
    ]
    for species_name in rare_species:
        if random.random() < 0.08:
            hour = random.randint(6, 19)
            minute = random.randint(0, 59)
            add_detection(species_name, local_time(day, hour, minute), random.uniform(0.65, 0.95))

db.add_all(new_detections)
db.commit()
db.close()

print(f"Seeded {len(new_detections)} synthetic detections across the last {DAYS_BACK} days.")