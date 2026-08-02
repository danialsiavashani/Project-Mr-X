from app.db.base import Base, engine
from app.db.models.species import Species
from app.db.models.detection import Detection

Base.metadata.create_all(bind=engine)
print("Tables created.")