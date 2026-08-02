from app.db.base import SessionLocal
from app.db.models.species import Species

SPECIES_DATA = [
    ("American Crow", "bird"),
    ("Anna's Hummingbird", "bird"),
    ("Black Phoebe", "bird"),
    ("Bushtit", "bird"),
    ("California Quail", "bird"),
    ("California Scrub-Jay", "bird"),
    ("Cat", "cat"),
    ("Dog", "dog"),
    ("Eurasian Collared-Dove", "bird"),
    ("House Finch", "bird"),
    ("Mourning Dove", "bird"),
    ("Northern Mockingbird", "bird"),
    ("Song Sparrow", "bird"),
    ("Squirrel", "squirrel"),
    ("White-crowned Sparrow", "bird"),
]

db = SessionLocal()

for name, group in SPECIES_DATA:
    existing = db.query(Species).filter(Species.name == name).first()
    if existing:
        print(f"Skipping '{name}' — already exists")
        continue
    species = Species(name=name, group=group)
    db.add(species)

db.commit()
db.close()
print("Species seeded.")