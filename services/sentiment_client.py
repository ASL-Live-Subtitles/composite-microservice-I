# services/sentiment_client.py
from __future__ import annotations

import os
from typing import Optional

import httpx

from models.pipeline import SentimentRequest, SentimentResponse


class SentimentClient:
    """
    HTTP client for the sentiment microservice you already deployed.

    It calls:
      POST {SENTIMENT_BASE_URL}{SENTIMENT_SENTIMENTS_PATH}
    with {"text": "..."} and expects a SentimentResult-compatible JSON.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "SENTIMENT_BASE_URL",
            "http://34.138.252.36:8000",
        )
        self.path = path or os.environ.get(
            "SENTIMENT_SENTIMENTS_PATH",
            "/sentiments",
        )

    async def analyze(self, req: SentimentRequest) -> SentimentResponse:
        url = f"{self.base_url.rstrip('/')}{self.path}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=req.model_dump())
            resp.raise_for_status()
            data = resp.json()

        # 這裡直接用 Pydantic 幫我們驗證與轉型
        return SentimentResponse(**data)
