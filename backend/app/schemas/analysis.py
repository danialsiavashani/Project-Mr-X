from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any, List, Dict


class AnalysisRequest(BaseModel):
    species_id: Optional[int] = None
    source: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class AnalysisReportResponse(BaseModel):
    id: int
    created_at: datetime
    summary: str
    prediction: Optional[str] = None
    pattern_table: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None

    class Config:
        from_attributes = True