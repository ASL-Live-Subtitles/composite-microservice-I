# services/asl_agent_client.py
from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx

from models.pipeline import AslAgentRequest, AslAgentResponse


class AslAgentClient:
    """
    Simple HTTP client for the ASL Agent microservice.

    It calls:
      POST {ASL_AGENT_BASE_URL}{ASL_AGENT_SENTENCE_PATH}
    with a JSON payload that includes glosses, letters, context, and
    OpenAI configuration (API key, model) from environment variables.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "ASL_AGENT_BASE_URL",
            "https://asl-agent-746433182504.us-central1.run.app",
        )
        self.path = path or os.environ.get(
            "ASL_AGENT_SENTENCE_PATH",
            "/compose/sentence",
        )
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.openai_model = openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.retries = retries
        self.retry_backoff = retry_backoff

    async def compose_sentence(self, req: AslAgentRequest) -> AslAgentResponse:
        """
        Call the ASL Agent /compose/sentence endpoint and return the sentence.
        """
        url = f"{self.base_url.rstrip('/')}{self.path}"

        payload = {
            "glosses": req.glosses,
            "letters": req.letters,
            "context": req.context,
            "openai_api_key": self.openai_api_key,
            "openai_model": self.openai_model,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(1, self.retries + 1):
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    # Retry on transient upstream failures (5xx) or network hiccups.
                    is_retryable_status = (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code in {500, 502, 503, 504}
                    )
                    if attempt < self.retries and (is_retryable_status or isinstance(exc, httpx.RequestError)):
                        await asyncio.sleep(self.retry_backoff * attempt)
                        continue
                    raise

        return AslAgentResponse(
            text=data["text"],
            confidence=data.get("confidence"),
            model=data.get("model"),
        )
