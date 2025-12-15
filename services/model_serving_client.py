# services/model_serving_client.py
from __future__ import annotations

import os
from typing import Optional

import httpx

from models.pipeline import VideoGlossRequest, VideoGlossResponse


class ModelServingClient:
    """
    HTTP client for the model-serving microservice that converts video -> glosses.

    It calls:
      POST {MODEL_SERVING_BASE_URL}{MODEL_SERVING_VIDEO_GLOSS_PATH}
    with a payload containing either `video_url` or `video_b64`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "MODEL_SERVING_BASE_URL",
            "http://localhost:9002",
        )
        self.path = path or os.environ.get(
            "MODEL_SERVING_VIDEO_GLOSS_PATH",
            "/video-gloss",
        )

    async def video_to_gloss(self, req: VideoGlossRequest) -> VideoGlossResponse:
        url = f"{self.base_url.rstrip('/')}{self.path}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=req.model_dump())
            resp.raise_for_status()
            data = resp.json()
        return VideoGlossResponse(**data)
