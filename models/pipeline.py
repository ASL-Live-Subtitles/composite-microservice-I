# models/pipeline.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# -----------------------------
# ASL Agent models
# -----------------------------
class AslAgentRequest(BaseModel):
    """Input to ASL Agent from the composite service."""
    glosses: List[str] = Field(
        ...,
        description="List of ASL gloss tokens.",
        json_schema_extra={"example": ["IX-1", "GOOD", "IDEA"]},
    )
    letters: List[str] = Field(
        default_factory=list,
        description="Optional list of fingerspelled letters.",
        json_schema_extra={"example": ["A", "I"]},
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional textual context to help the ASL agent.",
        json_schema_extra={"example": "Brainstorming a new sprint plan."},
    )


class AslAgentResponse(BaseModel):
    """Response from ASL Agent."""
    text: str = Field(
        ...,
        description="The composed natural language sentence.",
        json_schema_extra={"example": "I have a good idea."},
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score from the ASL agent (may be null).",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model used by ASL agent.",
    )


# -----------------------------
# Sentiment service models
# -----------------------------
class SentimentRequest(BaseModel):
    """Payload for the sentiment microservice."""
    text: str = Field(
        ...,
        description="Sentence to analyze.",
        json_schema_extra={"example": "The food was okay but the service was slow."},
    )


class SentimentResponse(BaseModel):
    """Subset of SentimentResult returned by the sentiment microservice."""
    id: UUID = Field(
        ...,
        description="ID of the sentiment record.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    text: str = Field(
        ...,
        description="Original text that was analyzed.",
        json_schema_extra={"example": "The food was okay but the service was slow."},
    )
    sentiment: str = Field(
        ...,
        description="Sentiment label produced by the sentiment microservice.",
        json_schema_extra={"example": "neutral"},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from the sentiment microservice.",
        json_schema_extra={"example": 0.63},
    )
    analyzed_at: datetime = Field(
        ...,
        description="Timestamp when the analysis was performed (UTC).",
        json_schema_extra={"example": "2025-11-15T22:01:36"},
    )


# -----------------------------
# Composite pipeline models
# -----------------------------
class PipelineInput(BaseModel):
    """
    Top-level input to the composite microservice.

    For now we assume glosses + letters are already available
    (image → glosses is handled by another atomic microservice).
    """
    glosses: List[str] = Field(
        ...,
        description="ASL gloss sequence generated from gesture recognition.",
        json_schema_extra={"example": ["IX-1", "GOOD", "IDEA"]},
    )
    letters: List[str] = Field(
        default_factory=list,
        description="Optional fingerspelling letters.",
        json_schema_extra={"example": ["A", "I"]},
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context to help ASL Agent generate better sentences.",
        json_schema_extra={"example": "Brainstorming a new sprint plan."},
    )


class PipelineResult(BaseModel):
    """
    Final composite result returned to the client.

    It includes:
    - original glosses + letters
    - the composed sentence from ASL Agent
    - the sentiment analysis result from the sentiment microservice
    """
    glosses: List[str]
    letters: List[str]
    context: Optional[str]
    sentence: str
    sentiment: SentimentResponse

    model_config = {
        "json_schema_extra": {
            "example": {
                "glosses": ["IX-1", "GOOD", "IDEA"],
                "letters": ["A", "I"],
                "context": "Brainstorming a new sprint plan.",
                "sentence": "I have a good idea.",
                "sentiment": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "text": "I have a good idea.",
                    "sentiment": "positive",
                    "confidence": 0.93,
                    "analyzed_at": "2025-11-15T22:05:00",
                },
            }
        }
    }
