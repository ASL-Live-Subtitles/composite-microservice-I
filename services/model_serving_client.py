# services/model_serving_client.py
from __future__ import annotations

import base64
import io
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
        
        headers = {}
        # Pass OpenAI key if available for better gloss inference
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["X-OpenAI-Key"] = api_key

        async with httpx.AsyncClient(timeout=60.0) as client:
            if req.video_b64:
                try:
                    video_bytes = base64.b64decode(req.video_b64)
                except Exception as e:
                    raise ValueError(f"Invalid base64 video data: {e}")

                # Prepare multipart/form-data upload
                # We send the bytes as a file named 'video.webm' (or generic)
                files = {"file": ("video.webm", video_bytes, "video/webm")}
                # Optional: pass target_fps or other form fields if needed
                form_data = {"target_fps": 5}

                resp = await client.post(url, files=files, data=form_data, headers=headers)

            elif req.video_url:
                # TODO: If video_url is supported by downstream or we download it here.
                # For now, this branch is not the primary path for the frontend recording.
                # We could potentially download the URL and forward bytes.
                raise NotImplementedError("video_url is not yet implemented in ModelServingClient adapter.")
            else:
                raise ValueError("VideoGlossRequest must provide video_b64.")

            resp.raise_for_status()
            data = resp.json()

        # Adapter: Map model-serving response (gloss, confidence, etc.) to internal VideoGlossResponse
        # Model Service returns: { "gloss": ["HELLO"], ... }
        # Pipeline expects: { "glosses": ["HELLO"], "letters": [] }
        glosses = data.get("gloss", [])
        letters = []  # Model service doesn't return separate letters currently

        return VideoGlossResponse(
            glosses=glosses,
            letters=letters
        )
