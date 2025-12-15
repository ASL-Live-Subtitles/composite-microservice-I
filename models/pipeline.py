# models/pipeline.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
import hashlib

from pydantic import BaseModel, Field, model_validator


def compute_sentence_key(text: str) -> str:
    """
    Deterministic key for a sentence, used to enforce logical foreign-key style
    linkage between ASL Agent output and Sentiment results.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------
# Model Serving (video -> gloss) models
# -----------------------------
class VideoGlossRequest(BaseModel):
    """Payload sent to the model-serving microservice for gloss extraction."""

    video_url: Optional[str] = Field(
        default=None,
        description="URL of the uploaded video to transcribe into glosses.",
        json_schema_extra={"example": "https://storage.googleapis.com/bucket/sample.mp4"},
    )
    video_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded video payload when a URL is not available.",
        json_schema_extra={"example": "<base64-encoded-bytes>"},
    )

    @model_validator(mode="after")
    def _require_source(self):
        if not self.video_url and not self.video_b64:
            raise ValueError("Provide either video_url or video_b64 for gloss extraction.")
        return self


class VideoGlossResponse(BaseModel):
    """Gloss output returned by the model-serving microservice."""

    glosses: List[str] = Field(
        ...,
        description="Detected ASL gloss tokens from the video.",
        json_schema_extra={"example": ["IX-1", "GOOD", "IDEA"]},
    )
    letters: List[str] = Field(
        default_factory=list,
        description="Optional fingerspelling letters detected from the video.",
        json_schema_extra={"example": ["A", "I"]},
    )


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


class PipelineLinkage(BaseModel):
    """
    Logical foreign-key style linkage that ties together the composite pipeline
    instance and the downstream sentiment record.
    """
    pipeline_id: UUID = Field(
        default_factory=uuid4,
        description="Composite pipeline identifier.",
    )
    sentence_key: str = Field(
        ...,
        description="Deterministic key for the sentence returned by ASL Agent.",
    )
    sentiment_id: UUID = Field(
        ...,
        description="Foreign-key style reference to the sentiment record id.",
    )


# -----------------------------
# Composite pipeline models
# -----------------------------
class PipelineInput(BaseModel):
    """
    Top-level input to the composite microservice.

    Accepts either:
    - pre-computed glosses/letters, OR
    - a video reference (URL or base64) to run through the model-serving service.
    """
    glosses: List[str] = Field(
        default_factory=list,
        description="ASL gloss sequence generated from gesture recognition.",
        json_schema_extra={"example": ["IX-1", "GOOD", "IDEA"]},
    )
    letters: List[str] = Field(
        default_factory=list,
        description="Optional fingerspelling letters.",
        json_schema_extra={"example": ["A", "I"]},
    )
    video_url: Optional[str] = Field(
        default=None,
        description="URL to a video to be converted to glosses via model serving.",
        json_schema_extra={"example": "https://storage.googleapis.com/bucket/sample.mp4"},
    )
    video_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded video payload when URL is unavailable.",
        json_schema_extra={"example": "<base64-encoded-bytes>"},
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context to help ASL Agent generate better sentences.",
        json_schema_extra={"example": "Brainstorming a new sprint plan."},
    )

    @model_validator(mode="after")
    def _require_gloss_source(self):
        has_glosses = bool(self.glosses)
        has_video = bool(self.video_url or self.video_b64)
        if not has_glosses and not has_video:
            raise ValueError("Provide glosses or a video (video_url or video_b64) to run the pipeline.")
        return self


class PipelineResult(BaseModel):
    """
    Final composite result returned to the client.

    It includes:
    - original glosses + letters
    - the composed sentence from ASL Agent
    - the sentiment analysis result from the sentiment microservice
    - linkage metadata to enforce logical FK constraints between services
    """
    glosses: List[str]
    letters: List[str]
    context: Optional[str]
    sentence: str
    sentiment: SentimentResponse
    linkage: PipelineLinkage

    @staticmethod
    def _expected_key(sentence: str) -> str:
        return compute_sentence_key(sentence)

    @classmethod
    def _validate_linkage(cls, sentence: str, sentiment: SentimentResponse, linkage: PipelineLinkage) -> None:
        expected_key = cls._expected_key(sentence)
        sentiment_key = compute_sentence_key(sentiment.text)
        if sentiment_key != expected_key:
            raise ValueError("Sentiment record text does not match sentence produced by ASL Agent.")
        if linkage.sentence_key != expected_key:
            raise ValueError("Linkage sentence_key does not match the composed sentence.")
        if linkage.sentiment_id != sentiment.id:
            raise ValueError("Linkage sentiment_id must reference the sentiment record id.")

    @model_validator(mode="after")
    def _enforce_foreign_keys(self):
        self._validate_linkage(self.sentence, self.sentiment, self.linkage)
        return self

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
                "linkage": {
                    "pipeline_id": "bb80a9bc-8a32-4d24-9dbc-9086f545af8e",
                    "sentence_key": "2f40239f9c7ef8adc95fb4cf055f20f45ac1c4d3b8599b65a1f70ce773b2764d",
                    "sentiment_id": "550e8400-e29b-41d4-a716-446655440000",
                },
            }
        }
    }


class PipelineBatchInput(BaseModel):
    """Batch wrapper for running multiple pipeline inputs."""
    items: List[PipelineInput]
