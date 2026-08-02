from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models.detection import Detection
from app.db.models.species import Species
from app.schemas.detection import (
    DetectionCreate,
    DetectionResponse,
    DetectionWithSpecies,
    PaginatedDetections,
)

router = APIRouter()


@router.post("/detections", response_model=DetectionResponse)
def create_detection(detection: DetectionCreate, db: Session = Depends(get_db)):
    db_detection = Detection(**detection.model_dump())
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection


@router.get("/detections/latest", response_model=DetectionWithSpecies)
def get_latest_detection(db: Session = Depends(get_db)):
    result = (
        db.query(Detection, Species)
        .join(Species, Detection.species_id == Species.id)
        .order_by(Detection.timestamp.desc())
        .first()
    )

    if result is None:
        raise HTTPException(status_code=404, detail="No detections found")

    detection, species = result
    return DetectionWithSpecies(
        **detection.__dict__,
        species_name=species.name,
        species_group=species.group,
    )


@router.get("/detections", response_model=PaginatedDetections)
def list_detections(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    species_id: Optional[int] = None,
    source: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Detection, Species).join(Species, Detection.species_id == Species.id)

    if species_id is not None:
        query = query.filter(Detection.species_id == species_id)
    if source is not None:
        query = query.filter(Detection.source == source)
    if start_date is not None:
        query = query.filter(Detection.timestamp >= start_date)
    if end_date is not None:
        query = query.filter(Detection.timestamp <= end_date)

    total = query.count()

    results = (
        query.order_by(Detection.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    items = [
        DetectionWithSpecies(
            **detection.__dict__,
            species_name=species.name,
            species_group=species.group,
        )
        for detection, species in results
    ]

    return PaginatedDetections(total=total, skip=skip, limit=limit, items=items)