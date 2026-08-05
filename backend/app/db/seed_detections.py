import random
from datetime import datetime, timedelta

from app.db.base import SessionLocal
from app.db.models.species import Species
from app.db.models.detection import Detection

random.seed(42)  # reproducible

db = SessionLocal()

species_by_name = {s.name: s.id for s in db.query(Species).all()}

now = datetime.utcnow()
DAYS_BACK = 30

PLACEHOLDER_CROP = "ml/wildlife-monitor/outputs/crops/synthetic_seed.jpg"

new_detections = []


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

    # Squirrel: almost every day, midday cluster
    if random.random() < 0.85:
        hour = random.randint(10, 14)
        minute = random.randint(0, 59)
        dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        add_detection("Squirrel", dt, random.uniform(0.75, 0.98))

    # Anna's Hummingbird: frequent, morning AND afternoon visits
    for hour_range in [(7, 9), (14, 16)]:
        if random.random() < 0.7:
            hour = random.randint(*hour_range)
            minute = random.randint(0, 59)
            dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            add_detection("Anna's Hummingbird", dt, random.uniform(0.8, 0.99))

    # Cat: recurring specifically on Wed/Sat evenings — a real weekly pattern
    # for BehaviorPulse's recurring-day detection to actually find
    if day.weekday() in (2, 5) and random.random() < 0.8:  # Wed=2, Sat=5
        hour = random.randint(18, 20)
        minute = random.randint(0, 59)
        dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        add_detection("Cat", dt, random.uniform(0.7, 0.95))

    # General morning birds — scattered, moderate frequency
    morning_birds = [
        "House Finch", "Song Sparrow", "White-crowned Sparrow",
        "American Crow", "Mourning Dove", "Northern Mockingbird",
    ]
    for _ in range(random.randint(1, 3)):
        species_name = random.choice(morning_birds)
        hour = random.randint(6, 10)
        minute = random.randint(0, 59)
        dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        add_detection(species_name, dt, random.uniform(0.7, 0.97))

    # Thinner, occasional classes — real-world rarity reflected in frequency
    rare_species = [
        "Black Phoebe", "Bushtit", "California Quail",
        "California Scrub-Jay", "Eurasian Collared-Dove", "Dog",
    ]
    if random.random() < 0.25:
        species_name = random.choice(rare_species)
        hour = random.randint(6, 19)
        minute = random.randint(0, 59)
        dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        add_detection(species_name, dt, random.uniform(0.65, 0.95))

db.add_all(new_detections)
db.commit()
db.close()

print(f"Seeded {len(new_detections)} synthetic detections across the last {DAYS_BACK} days.")