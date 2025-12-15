from google.cloud import pubsub_v1
import json
from typing import Union, List, Optional
from models.pipeline import PipelineResult
from datetime import datetime
import os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "hpml-472522")
PUBSUB_TOPIC = "asl-session-completed"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)

def publish_event(
    results: Union[PipelineResult, List[PipelineResult]],
    source_api: str,
    recipient_email: Optional[str] = None,
) -> None:
    if not isinstance(results, list):
        results = [results]

    event = {
        "event_type": "ASL_COMPLETED",
        "event_source": source_api,
        "recipient_email": recipient_email,   # <-- add this
        "batch_size": len(results),
        "timestamp": datetime.utcnow().isoformat(),
        "items": [
            {
                "sentence": r.sentence,
                "sentiment": r.sentiment.sentiment,
                "confidence": r.sentiment.confidence,
                "sentence_key": r.linkage.sentence_key,
            }
            for r in results
        ],
    }

    publisher.publish(topic_path, json.dumps(event).encode("utf-8"))