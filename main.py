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
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models.pipeline import (
    PipelineInput,
    PipelineResult,
    AslAgentRequest,
    SentimentRequest,
    SentimentResponse,
    PipelineLinkage,
    compute_sentence_key,
    PipelineBatchInput,
    VideoGlossRequest,
)
from services.asl_agent_client import AslAgentClient
from services.sentiment_client import SentimentClient
from services.model_serving_client import ModelServingClient

from auth.oauth import oauth_login, oauth_callback, get_roles
from auth.dependencies import require_user, require_roles
from auth.jwt import create_jwt
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware

from pub_sub.publisher import publish_event

# Load .env if present; allow local .env to override any pre-set env vars so the correct
# OpenAI key and service endpoints are used when running locally.
load_dotenv(override=True)

MODEL_SERVING_BASE_URL = os.environ.get(
    "MODEL_SERVING_BASE_URL",
    "http://localhost:9002",
)
MODEL_SERVING_VIDEO_GLOSS_PATH = os.environ.get("MODEL_SERVING_VIDEO_GLOSS_PATH", "/video-gloss")
ASL_AGENT_BASE_URL = os.environ.get(
    "ASL_AGENT_BASE_URL",
    "https://asl-agent-746433182504.us-central1.run.app",
)
ASL_AGENT_SENTENCE_PATH = os.environ.get("ASL_AGENT_SENTENCE_PATH", "/compose/sentence")
SENTIMENT_BASE_URL = os.environ.get("SENTIMENT_BASE_URL", "http://34.138.252.36:8000")
SENTIMENT_SENTIMENTS_PATH = os.environ.get("SENTIMENT_SENTIMENTS_PATH", "/sentiments")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://storage.googleapis.com/signtalk/index.html")

app = FastAPI(
    title="ASL Composite Microservice",
    description=(
        "Orchestrates ASL Agent and Sentiment microservices.\n\n"
        "Pipeline: glosses/letters -> ASL Agent -> sentence -> Sentiment service."
    ),
    version="0.1.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ["SESSION_SECRET"],
    https_only=True,
    same_site="lax",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://localhost:5173",
        "https://storage.googleapis.com",
        "https://storage.googleapis.com/signtalk",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"message": "ok"}


# Auth endpoints
@app.get("/login", tags=["auth"])
async def login(request: Request):
    return await oauth_login(request)

@app.get("/auth/callback", tags=["auth"])
async def auth_callback(request: Request):
    userinfo = await oauth_callback(request)
    roles = get_roles(userinfo)
    jwt_token = create_jwt(userinfo, roles)
    return RedirectResponse(f"{FRONTEND_URL}?token={jwt_token}")

@app.get("/logout", tags=["auth"])
def logout():
    return RedirectResponse(f"{FRONTEND_URL}?logout=1")

@app.get("/users", tags=["auth"])
def get_current_user(user: Dict[str, Any] = Depends(require_user)):
    return {
        "user": {
            "sub": user.get("sub"),
            "email": user.get("email"),
            "name": user.get("name"),
            "roles": user.get("roles", []),
        }
    }

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
    gloss_url = f"{MODEL_SERVING_BASE_URL.rstrip('/')}{MODEL_SERVING_VIDEO_GLOSS_PATH}"
    asl_url = f"{ASL_AGENT_BASE_URL.rstrip('/')}{ASL_AGENT_SENTENCE_PATH}"
    sent_url = f"{SENTIMENT_BASE_URL.rstrip('/')}{SENTIMENT_SENTIMENTS_PATH}"

    with httpx.Client(timeout=30.0) as client:
        glosses = payload.glosses
        letters = payload.letters

        if not glosses:
            video_req = {"video_url": payload.video_url, "video_b64": payload.video_b64}
            gloss_resp = client.post(gloss_url, json=video_req)
            gloss_resp.raise_for_status()
            gloss_data = gloss_resp.json()
            glosses = gloss_data.get("glosses", [])
            letters = gloss_data.get("letters", [])

            if not glosses:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Model-serving service returned no glosses for the provided video.",
                )

        asl_payload = {
            "glosses": glosses,
            "letters": letters,
            "context": payload.context,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        }

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
    dependencies=[Depends(require_user)],
)
async def run_asl_pipeline(
    payload: PipelineInput,
    user: Dict[str, Any] = Depends(require_user)) -> PipelineResult:
    """
    Synchronous composite endpoint:

    1. Call ASL Agent /compose/sentence to generate a natural language sentence.
    2. Call Sentiment microservice /sentiments to analyze that sentence.
    3. Return a combined PipelineResult.
    """
    asl_client = AslAgentClient()
    sentiment_client = SentimentClient()
    model_serving_client = ModelServingClient()

    try:
        glosses = payload.glosses
        letters = payload.letters

        # 1) Call model-serving for video -> gloss if glosses are not supplied
        if not glosses:
            video_req = VideoGlossRequest(
                video_url=payload.video_url,
                video_b64=payload.video_b64,
            )
            gloss_resp = await model_serving_client.video_to_gloss(video_req)
            glosses = gloss_resp.glosses
            letters = gloss_resp.letters

            if not glosses:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Model-serving service returned no glosses for the provided video.",
                )

        # 2) Call ASL Agent
        asl_req = AslAgentRequest(glosses=glosses, letters=letters, context=payload.context)
        asl_resp = await asl_client.compose_sentence(asl_req)

        # 3) Call Sentiment microservice
        sent_req = SentimentRequest(text=asl_resp.text)
        sent_resp = await sentiment_client.analyze(sent_req)

        sentence_key = compute_sentence_key(asl_resp.text)
        linkage = PipelineLinkage(
            pipeline_id=uuid4(),
            sentence_key=sentence_key,
            sentiment_id=sent_resp.id,
        )

        # 4) Combine into PipelineResult
        result = PipelineResult(
            glosses=glosses,
            letters=letters,
            context=payload.context,
            sentence=asl_resp.text,
            sentiment=sent_resp,
            linkage=linkage,
        )

        # 5) Publish event to Pub/Sub
        publish_event(result, source_api="/asl-pipeline", recipient_email=user.get("email"))

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
    dependencies=[Depends(require_roles("admin"))],
)
async def run_asl_pipeline_batch(
    batch: PipelineBatchInput,
    user: Dict[str, Any] = Depends(require_user)) -> List[PipelineResult]:
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

    publish_event(results, source_api="/asl-pipeline", recipient_email=user.get("email"))

    return results


# Entrypoint for `python main.py`
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)