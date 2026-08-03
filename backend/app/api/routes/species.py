from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models.species import Species
from app.schemas.species import SpeciesResponse

router = APIRouter()


@router.get("/species", response_model=List[SpeciesResponse])
def list_species(db: Session = Depends(get_db)):
    return db.query(Species).all()