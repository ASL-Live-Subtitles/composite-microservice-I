# 🤝 ASL Composite Microservice

This microservice **orchestrates two atomic microservices**:

1. **ASL Agent** (Cloud Run)
   - Endpoint: `POST /compose/sentence`
   - Task: Convert ASL glosses + letters + context → natural language sentence

2. **Sentiment Microservice** 
   - Endpoint: `POST /sentiments`
   - Task: Perform sentiment analysis over the generated sentence and store it in MySQL

The composite service exposes a **single high-level API**:

> **POST /asl-pipeline**
> 
> Input: glosses + letters + context
> 
> Output: generated sentence + sentiment result


## 🧱 Architecture Overview

**Pipeline (synchronous):**

```
Client
  │
  │ POST /asl-pipeline
  ▼
ASL Composite Service
  │
  ├─▶ ASL Agent         (Cloud Run: /compose/sentence)
  │      └─ returns: { "text": "...", "confidence": null, "model": "gpt-4o-mini" }
  │
  └─▶ Sentiment Service (VM / FastAPI: /sentiments)
         └─ returns: SentimentResult JSON
  │
  ▼
Composite response: { glosses, letters, context, sentence, sentiment }
```

## 📂 Project Structure

```bash
asl-composite-service/
│── main.py                  # FastAPI app: /asl-pipeline endpoint
│── requirements.txt
│── .env.example
│── README.md
│
├── models/
│   └── pipeline.py          # Pydantic models for pipeline + external services
│
└── services/
    ├── asl_agent_client.py  # HTTP client for ASL Agent
    └── sentiment_client.py  # HTTP client for Sentiment microservice
```

## ⚙️ Setup & Run

### 1. Clone & enter project

```bash
git clone https://github.com/ASL-Live-Subtitles/composite-microservice-I.git
cd composite-microservice-I
```

### 2. Create & activate virtualenv

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env`and add following content  
**.env.example content:**

```env
# Model-serving endpoint (video -> gloss)
MODEL_SERVING_BASE_URL=http://localhost:9002
MODEL_SERVING_VIDEO_GLOSS_PATH=/video-gloss

# OpenAI config passed through to ASL Agent (if required)
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=[REPLACE WITH YOUR KEY]

# ASL Agent Cloud Run endpoint (gloss -> sentence)
ASL_AGENT_BASE_URL=https://asl-agent-746433182504.us-central1.run.app
ASL_AGENT_SENTENCE_PATH=/compose/sentence

# Sentiment microservice endpoint (sentence -> sentiment)
SENTIMENT_BASE_URL=http://34.138.252.36:8000
SENTIMENT_SENTIMENTS_PATH=/sentiments
```

### 5. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 6. Open API docs

- Local: http://localhost:8001/docs
- On VM: http://34.138.252.36:8001/docs


## 📌 Endpoints

### 💬 POST /asl-pipeline

Run the complete pipeline:

1. glosses + letters + context → ASL Agent → sentence
2. sentence → Sentiment Microservice → sentiment result

### ⚡️ POST /asl-pipeline/batch

Execute asl-pipeline concurrently for multiple payloads using thread-level parallelism via `ThreadPoolExecutor`.  

---

## 🧾 Data Models

### 1️⃣ PipelineInput (request to composite service)

```json
{
  "glosses": ["IX-1", "GOOD", "IDEA"],
  "letters": ["A", "I"],
  "context": "Brainstorming a new sprint plan."
}
```

**Fields:**

- `glosses`: string[] – ASL gloss tokens (required)
- `letters`: string[] – optional fingerspelling letters
- `context`: string | null – optional context to improve sentence quality

**Pydantic model (models/pipeline.py):**

```python
class PipelineInput(BaseModel):
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
```

### 2️⃣ ASL Agent side

#### ✅ Request from Composite → ASL Agent

```python
class AslAgentRequest(BaseModel):
    glosses: List[str]
    letters: List[str] = []
    context: Optional[str] = None
```

Composite service calls:

```http
POST {ASL_AGENT_BASE_URL}{ASL_AGENT_SENTENCE_PATH}
Content-Type: application/json
```

**Payload example:**

```json
{
  "glosses": ["IX-1", "GOOD", "IDEA"],
  "letters": ["A", "I"],
  "context": "Brainstorming a new sprint plan.",
  "openai_api_key": "",
  "openai_model": "gpt-4o-mini"
}
```

#### ✅ Real ASL Agent Response

ASL Agent currently returns:

```json
{
  "text": "I have a good idea.",
  "confidence": null,
  "model": "gpt-4o-mini"
}
```

**Corresponding Pydantic model:**

```python
class AslAgentResponse(BaseModel):
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
```

### 3️⃣ Sentiment microservice side

#### ✅ Request from Composite → Sentiment Service

Composite service calls your sentiment API:

```http
POST {SENTIMENT_BASE_URL}{SENTIMENT_SENTIMENTS_PATH}
Content-Type: application/json
```

**Payload model:**

```python
class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        description="Sentence to analyze.",
        json_schema_extra={"example": "The food was okay but the service was slow."},
    )
```

**Example JSON:**

```json
{
  "text": "I have a good idea."
}
```

#### ✅ Expected Response (SentimentResponse)

```python
class SentimentResponse(BaseModel):
    id: UUID
    text: str
    sentiment: str
    confidence: float
    analyzed_at: datetime
```

**Example:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "I have a good idea.",
  "sentiment": "positive",
  "confidence": 0.93,
  "analyzed_at": "2025-11-15T22:05:00"
}
```

### 4️⃣ PipelineResult (response from composite to client)

Complete pipeline output:

```python
class PipelineResult(BaseModel):
    glosses: List[str]
    letters: List[str]
    context: Optional[str]
    sentence: str
    sentiment: SentimentResponse
    linkage: PipelineLinkage
```

**Example:**

```json
{
  "glosses": ["IX-1", "GOOD", "IDEA"],
  "letters": ["A", "I"],
  "context": "Brainstorming a new sprint plan.",
  "sentence": "I have a good idea.",
  "sentiment": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "I have a good idea.",
    "sentiment": "positive",
    "confidence": 0.93,
    "analyzed_at": "2025-11-15T22:05:00"
  },
  "linkage": {
    "pipeline_id": "bb80a9bc-8a32-4d24-9dbc-9086f545af8e",
    "sentence_key": "2f40239f9c7ef8adc95fb4cf055f20f45ac1c4d3b8599b65a1f70ce773b2764d",
    "sentiment_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

The composite service applies a foreign-key style constraint by hashing the ASL Agent sentence into `sentence_key` and validating it against both the sentiment text and `linkage.sentiment_id` output. This validates the sentiment row match with the generated sentence.

## 🌐 Endpoint Details

### 🔹 POST /asl-pipeline

**Description:**
Run the full ASL → sentence → sentiment pipeline synchronously.

**Request body:** PipelineInput

```json
{
  "glosses": ["IX-1", "GOOD", "IDEA"],
  "letters": ["A", "I"],
  "context": "Brainstorming a new sprint plan."
}
```

**Response:** 201 Created + PipelineResult

```json
{
  "glosses": ["IX-1", "GOOD", "IDEA"],
  "letters": ["A", "I"],
  "context": "Brainstorming a new sprint plan.",
  "sentence": "I have a good idea.",
  "sentiment": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "I have a good idea.",
    "sentiment": "positive",
    "confidence": 0.93,
    "analyzed_at": "2025-11-15T22:05:00"
  }
}
```

## Testing the API
![](/images/test_result.png)

---

## 🧪 Example cURL

```bash
curl -X 'POST' \
  'http://localhost:8001/asl-pipeline' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "glosses": ["IX-1", "GOOD", "IDEA"],
  "letters": ["A", "I"],
  "context": "Brainstorming a new sprint plan."
}'
```
