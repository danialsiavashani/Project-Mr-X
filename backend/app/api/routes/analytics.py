from datetime import datetime, timezone

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models.detection import Detection
from app.db.models.species import Species
from app.db.models.analysis_report import AnalysisReport
from app.schemas.analysis import AnalysisRequest, AnalysisReportResponse
from app.core.config import (
    BEHAVIORPULSE_CLIENT_ID,
    BEHAVIORPULSE_API_KEY,
    BEHAVIORPULSE_BASE_URL,
)

router = APIRouter()


def to_utc_isoformat(dt):
    """
    SQLite silently drops timezone info on round-trip, even when a
    timezone-aware datetime is stored. Always assume naive timestamps
    from the database are UTC (since that's how we write them) and
    attach it explicitly before sending anywhere external.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@router.post("/analytics/generate-report", response_model=AnalysisReportResponse)
def generate_report(filters: AnalysisRequest, db: Session = Depends(get_db)):
    query = db.query(Detection, Species).join(Species, Detection.species_id == Species.id)

    if filters.species_id is not None:
        query = query.filter(Detection.species_id == filters.species_id)
    if filters.source is not None:
        query = query.filter(Detection.source == filters.source)
    if filters.start_date is not None:
        query = query.filter(Detection.timestamp >= filters.start_date)
    if filters.end_date is not None:
        query = query.filter(Detection.timestamp <= filters.end_date)

    results = query.all()

    if not results:
        raise HTTPException(status_code=404, detail="No detections match the given filters")

    events = [
        {
            "observed_at": to_utc_isoformat(detection.timestamp),
            "subject": {"type": "animal", "label": species.name},
            "source": {"type": "camera", "id": detection.source},
            "confidence": detection.confidence,
        }
        for detection, species in results
    ]

    # print("Sample event being sent:", events[0] if events else "none")  # TEMP debug

    print(f"Matched {len(results)} rows")
    if results:
        timestamps = [d.timestamp for d, s in results]
        print(f"Earliest: {min(timestamps)}, Latest: {max(timestamps)}")
    print("Sample event being sent:", events[0] if events else "none")

    options = {"timezone": "America/Los_Angeles"}
    if filters.start_date is not None:
        options["date_from"] = to_utc_isoformat(filters.start_date)
    if filters.end_date is not None:
        options["date_to"] = to_utc_isoformat(filters.end_date)

    headers = {
        "X-Client-Id": BEHAVIORPULSE_CLIENT_ID,
        "X-Api-Key": BEHAVIORPULSE_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{BEHAVIORPULSE_BASE_URL}/v1/observations/analyze",
            json={"observations": events, "options": options},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"BehaviorPulse request failed: {e}")

    bp_result = response.json()

    report = AnalysisReport(
        filter_species_id=filters.species_id,
        filter_source=filters.source,
        filter_start_date=filters.start_date,
        filter_end_date=filters.end_date,
        summary=bp_result.get("summary", ""),
        prediction=bp_result.get("prediction"),
        pattern_table=bp_result.get("pattern_table"),
        confidence=bp_result.get("computed_confidence"),
        raw_response=bp_result,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return report