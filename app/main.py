import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ROOT = Path(__file__).resolve().parent.parent
RELEASE = json.loads((ROOT / "release.json").read_text())
MODEL = os.getenv("MODEL", RELEASE["model"])
PROMPT_VERSION = os.getenv("PROMPT_VERSION", RELEASE["prompt_version"])
PROMPT_DIR = ROOT / "prompts"

REQUESTS = Counter(
    "llm_requests_total", "Requests to /ask", ["prompt_version", "model", "status"]
)
LATENCY = Histogram(
    "llm_request_latency_seconds",
    "End-to-end /ask latency",
    ["prompt_version", "model"],
    buckets=(0.25, 0.5, 1, 2, 5, 10, 30, 120),
)
TOKENS = Counter(
    "llm_tokens_total", "Tokens used", ["prompt_version", "model", "type"]
)

app = FastAPI(title="LLMOps service")

# Create both status series at 0 so alerts and the dashboard see errors from the first one
for _status in ("ok", "error"):
    REQUESTS.labels(prompt_version=PROMPT_VERSION, model=MODEL, status=_status)


class AskRequest(BaseModel):
    question: str
    context: str = ""


class AskResponse(BaseModel):
    answer: str
    model: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def load_prompt(version: str) -> str:
    path = PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Unknown prompt version: {version}")
    return path.read_text()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "prompt_version": PROMPT_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    labels = {"prompt_version": PROMPT_VERSION, "model": MODEL}
    start = time.perf_counter()
    try:
        prompt = load_prompt(PROMPT_VERSION).format(context=req.context, question=req.question)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            r.raise_for_status()
    except HTTPException:
        REQUESTS.labels(**labels, status="error").inc()
        raise
    except httpx.HTTPError as e:
        REQUESTS.labels(**labels, status="error").inc()
        raise HTTPException(status_code=502, detail=f"Model backend error: {e}")

    elapsed = time.perf_counter() - start
    data = r.json()
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    REQUESTS.labels(**labels, status="ok").inc()
    LATENCY.labels(**labels).observe(elapsed)
    TOKENS.labels(**labels, type="prompt").inc(prompt_tokens)
    TOKENS.labels(**labels, type="completion").inc(completion_tokens)

    return AskResponse(
        answer=data["response"].strip(),
        model=MODEL,
        prompt_version=PROMPT_VERSION,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int(elapsed * 1000),
    )
