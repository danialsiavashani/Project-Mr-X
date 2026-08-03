from pydantic import BaseModel
from typing import Optional


class SpeciesResponse(BaseModel):
    id: int
    name: str
    group: str
    reference_photo_path: Optional[str] = None
    description: Optional[str] = None

    class Config:
        from_attributes = True