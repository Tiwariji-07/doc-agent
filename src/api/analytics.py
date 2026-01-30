"""
Analytics API endpoints.

Provides dashboard data and feedback submission endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.analytics.collector import track_feedback, get_analytics
from src.analytics.models import FeedbackEvent, AnalyticsOverview
from src.analytics.queries import mark_feedback_reviewed

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# === Request/Response Models ===

class FeedbackRequest(BaseModel):
    """User feedback submission."""
    query_id: str
    helpful: bool
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    success: bool
    message: str


# === Feedback Endpoint ===

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a query response.
    
    This is fire-and-forget - returns immediately.
    """
    event = FeedbackEvent(
        query_id=request.query_id,
        helpful=request.helpful,
        comment=request.comment,
    )
    await track_feedback(event)
    
    return FeedbackResponse(
        success=True,
        message="Feedback recorded" if request.helpful else "Thanks for your feedback, we'll review this",
    )


# === Dashboard Endpoints ===

@router.get("/overview")
async def get_overview(period: str = "today") -> AnalyticsOverview:
    """
    Get dashboard overview statistics.
    
    Args:
        period: Time period - "today", "week", or "month"
    """
    if period not in ("today", "week", "month"):
        raise HTTPException(status_code=400, detail="Invalid period. Use: today, week, month")
    
    analytics = await get_analytics()
    return await analytics.get_overview(period)


@router.get("/queries")
async def get_queries(limit: int = 50):
    """
    Get recent query history.
    
    Args:
        limit: Maximum number of queries to return (default 50)
    """
    analytics = await get_analytics()
    return await analytics.get_recent_queries(min(limit, 200))


@router.get("/feedback")
async def get_feedback_list(limit: int = 50):
    """
    Get feedback entries, optionally filtered to pending review.
    
    Args:
        limit: Maximum number of entries to return
    """
    analytics = await get_analytics()
    return await analytics.get_pending_feedback(min(limit, 200))


@router.post("/feedback/{feedback_id}/reviewed")
async def mark_reviewed(feedback_id: str):
    """Mark a feedback entry as reviewed."""
    success = await mark_feedback_reviewed(feedback_id)
    if not success:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return {"success": True}
