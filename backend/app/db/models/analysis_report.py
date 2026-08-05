from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.db.base import Base


class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # What filters were used to generate this report — lets you know what data it covers
    filter_species_id = Column(Integer, nullable=True)
    filter_source = Column(String, nullable=True)
    filter_start_date = Column(DateTime(timezone=True), nullable=True)
    filter_end_date = Column(DateTime(timezone=True), nullable=True)

    # BehaviorPulse's actual response
    summary = Column(String, nullable=False)
    prediction = Column(String, nullable=True)
    pattern_table = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)

    # Full raw response too, in case BehaviorPulse's schema grows and we
    # want fields later that we didn't think to pull out individually now
    raw_response = Column(JSON, nullable=True)