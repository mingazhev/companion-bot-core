"""Pydantic schemas for the internal HTTP service layer.

Request and response models for the two internal endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RefineRequest(BaseModel):
    """Optional body for ``POST /internal/refine/{user_id}``."""

    trigger: str = Field(
        default="internal_api",
        description="Label stored in the job payload identifying the enqueue source",
    )


class RefineResponse(BaseModel):
    """Response body returned by ``POST /internal/refine/{user_id}``."""

    queued: bool = Field(description="Always True when the job was successfully enqueued")
    user_id: str = Field(description="UUID of the user whose refinement was requested")
    queue_length: int = Field(
        description="Length of the refinement queue after enqueue (informational)",
    )


class DetectChangeRequest(BaseModel):
    """Request body for ``POST /internal/detect-change``."""

    text: str = Field(
        min_length=1,
        description="User message text to classify for configuration-change intent",
    )
