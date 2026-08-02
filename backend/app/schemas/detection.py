from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class DetectionCreate(BaseModel):
    species_id: int
    confidence: float
    crop_path: str
    frame_path: Optional[str] = None
    detector_confidence: Optional[float] = None
    model_version: Optional[str] = None
    source: str = "backyard_cam_1"


class DetectionResponse(DetectionCreate):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class DetectionWithSpecies(DetectionResponse):
    species_name: str
    species_group: str


class PaginatedDetections(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[DetectionWithSpecies]