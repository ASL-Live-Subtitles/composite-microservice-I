# main.py
from __future__ import annotations

import os

from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from models.pipeline import (
    PipelineInput,
    PipelineResult,
    AslAgentRequest,
    SentimentRequest,
)
from services.asl_agent_client import AslAgentClient
from services.sentiment_client import SentimentClient

# Load .env if present
load_dotenv()


app = FastAPI(
    title="ASL Composite Microservice",
    description=(
        "Orchestrates ASL Agent and Sentiment microservices.\n\n"
        "Pipeline: glosses/letters -> ASL Agent -> sentence -> Sentiment service."
    ),
    version="0.1.0",
)


@app.get("/", tags=["root"])
def root() -> Dict[str, Any]:
    """Simple health/root endpoint."""
    return {
        "message": "ASL Composite Microservice is running. See /docs for OpenAPI UI.",
        "timestamp": datetime.utcnow().isoformat(),
    }


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

        # 3) Combine into PipelineResult
        result = PipelineResult(
            glosses=payload.glosses,
            letters=payload.letters,
            context=payload.context,
            sentence=asl_resp.text,
            sentiment=sent_resp,
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {e}",
        )


# Entrypoint for `python main.py`
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
