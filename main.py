# main.py
from __future__ import annotations

import os

from datetime import datetime
from typing import Dict, Any, List
import asyncio
import concurrent.futures
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from models.pipeline import (
    PipelineInput,
    PipelineResult,
    AslAgentRequest,
    SentimentRequest,
    SentimentResponse,
    PipelineLinkage,
    compute_sentence_key,
    PipelineBatchInput,
)
from services.asl_agent_client import AslAgentClient
from services.sentiment_client import SentimentClient

# Load .env if present; allow local .env to override any pre-set env vars so the correct
# OpenAI key and service endpoints are used when running locally.
load_dotenv(override=True)

ASL_AGENT_BASE_URL = os.environ.get(
    "ASL_AGENT_BASE_URL",
    "https://asl-agent-746433182504.us-central1.run.app",
)
ASL_AGENT_SENTENCE_PATH = os.environ.get("ASL_AGENT_SENTENCE_PATH", "/compose/sentence")
SENTIMENT_BASE_URL = os.environ.get("SENTIMENT_BASE_URL", "http://34.138.252.36:8000")
SENTIMENT_SENTIMENTS_PATH = os.environ.get("SENTIMENT_SENTIMENTS_PATH", "/sentiments")


app = FastAPI(
    title="ASL Composite Microservice",
    description=(
        "Orchestrates ASL Agent and Sentiment microservices.\n\n"
        "Pipeline: glosses/letters -> ASL Agent -> sentence -> Sentiment service."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://localhost:5173",
        "https://storage.googleapis.com",
        "https://storage.googleapis.com/signtalk",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"message": "ok"}

@app.get("/", tags=["root"])
def root() -> Dict[str, Any]:
    """Simple health/root endpoint."""
    return {
        "message": "ASL Composite Microservice is running. See /docs for OpenAPI UI.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def run_pipeline_sync(payload: PipelineInput) -> PipelineResult:
    """
    Synchronous pipeline runner used by the threaded batch endpoint.
    Uses blocking httpx.Client inside a thread to satisfy the explicit
    requirement for thread-based parallel execution.
    """
    asl_url = f"{ASL_AGENT_BASE_URL.rstrip('/')}{ASL_AGENT_SENTENCE_PATH}"
    sent_url = f"{SENTIMENT_BASE_URL.rstrip('/')}{SENTIMENT_SENTIMENTS_PATH}"

    asl_payload = {
        "glosses": payload.glosses,
        "letters": payload.letters,
        "context": payload.context,
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }

    with httpx.Client(timeout=30.0) as client:
        asl_resp = client.post(asl_url, json=asl_payload)
        asl_resp.raise_for_status()
        asl_data = asl_resp.json()

        sent_req = {"text": asl_data["text"]}
        sent_resp = client.post(sent_url, json=sent_req)
        sent_resp.raise_for_status()
        sent_data = sent_resp.json()

    sentiment = SentimentResponse(**sent_data)
    sentence_key = compute_sentence_key(asl_data["text"])
    linkage = PipelineLinkage(
        pipeline_id=uuid4(),
        sentence_key=sentence_key,
        sentiment_id=sentiment.id,
    )

    return PipelineResult(
        glosses=payload.glosses,
        letters=payload.letters,
        context=payload.context,
        sentence=asl_data["text"],
        sentiment=sentiment,
        linkage=linkage,
    )


@app.post(
    "/asl-pipeline",
    response_model=PipelineResult,
    status_code=status.HTTP_201_CREATED,
    tags=["pipeline"],
)
async def run_asl_pipeline(payload: PipelineInput) -> PipelineResult:
    """
    Synchronous composite endpoint:

    1. Call ASL Agent /compose/sentence to generate a natural language sentence.
    2. Call Sentiment microservice /sentiments to analyze that sentence.
    3. Return a combined PipelineResult.
    """
    asl_client = AslAgentClient()
    sentiment_client = SentimentClient()

    try:
        # 1) Call ASL Agent
        asl_req = AslAgentRequest(
            glosses=payload.glosses,
            letters=payload.letters,
            context=payload.context,
        )
        asl_resp = await asl_client.compose_sentence(asl_req)

        # 2) Call Sentiment microservice
        sent_req = SentimentRequest(text=asl_resp.text)
        sent_resp = await sentiment_client.analyze(sent_req)

        sentence_key = compute_sentence_key(asl_resp.text)
        linkage = PipelineLinkage(
            pipeline_id=uuid4(),
            sentence_key=sentence_key,
            sentiment_id=sent_resp.id,
        )

        # 3) Combine into PipelineResult
        result = PipelineResult(
            glosses=payload.glosses,
            letters=payload.letters,
            context=payload.context,
            sentence=asl_resp.text,
            sentiment=sent_resp,
            linkage=linkage,
        )
        return result

    except httpx.HTTPStatusError as e:
        # Propagate upstream HTTP errors with their original status code
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Upstream service error calling {e.request.url}: {e.response.text}",
        ) from e

    except httpx.RequestError as e:
        # Network/transport issues reaching upstream services
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream connection error calling {e.request.url}: {e!s}",
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {e}",
        ) from e


@app.post(
    "/asl-pipeline/batch",
    response_model=List[PipelineResult],
    status_code=status.HTTP_201_CREATED,
    tags=["pipeline"],
)
async def run_asl_pipeline_batch(batch: PipelineBatchInput) -> List[PipelineResult]:
    """
    Batch pipeline endpoint that processes multiple payloads in parallel using
    a thread pool.
    """
    if not batch.items:
        return []

    loop = asyncio.get_running_loop()
    max_workers = min(3, max(1, len(batch.items)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, run_pipeline_sync, item)
            for item in batch.items
        ]
        results = await asyncio.gather(*tasks)
    return results


# Entrypoint for `python main.py`
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
