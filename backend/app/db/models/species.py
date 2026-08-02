from sqlalchemy import Column, Integer, String
from app.db.base import Base


class Species(Base):
    __tablename__ = "species"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    group = Column(String, nullable=False)  # "bird" / "cat" / "squirrel"
    reference_photo_path = Column(String, nullable=True)
    description = Column(String, nullable=True)