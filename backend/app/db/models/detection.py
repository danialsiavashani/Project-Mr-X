from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.db.base import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    species_id = Column(Integer, ForeignKey("species.id"), nullable=False)
    confidence = Column(Float, nullable=False)  # classifier (species ID) confidence — the primary number
    crop_path = Column(String, nullable=False)
    frame_path = Column(String, nullable=True)
    detector_confidence = Column(Float, nullable=True)  # detection-stage confidence, secondary
    model_version = Column(String, nullable=True)
    source = Column(String, nullable=False, default="backyard_cam_1")