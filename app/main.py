import os
import json
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ROOT = Path(__file__).resolve().parent.parent
RELEASE = json.loads((ROOT / "release.json").read_text())
MODEL = os.getenv("MODEL", RELEASE["model"])
PROMPT_VERSION = os.getenv("PROMPT_VERSION", RELEASE["prompt_version"])
PROMPT_DIR = ROOT / "prompts"

app = FastAPI(title="LLMOps service")


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


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    prompt = load_prompt(PROMPT_VERSION).format(context=req.context, question=req.question)
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            )
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Model backend error: {e}")
    data = r.json()
    return AskResponse(
        answer=data["response"].strip(),
        model=MODEL,
        prompt_version=PROMPT_VERSION,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
