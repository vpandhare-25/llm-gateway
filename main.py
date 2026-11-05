"""
LLM Gateway - Day 1: a simple "middleman" between your app and the AI.

Flow:  you  --question-->  this server  --question-->  Ollama (the AI)
       you  <--answer----  this server  <--answer----  Ollama

Day 2: before asking the AI, we check a semantic cache (cache.py). If a similar
question was answered recently, we return that answer instead (source="cache").
Every request is logged with its latency and token counts so /stats can compare.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

import cache  # our semantic cache (cache.py)

OLLAMA_URL = "http://localhost:11434/api/chat"  # where Ollama listens on your Mac
MODEL = "llama3.2"                               # the model you downloaded
LOG_FILE = Path(__file__).parent / "logs" / "requests.jsonl"

app = FastAPI(title="LLM Gateway")


class AskRequest(BaseModel):
    """What a user sends us: just a question."""
    question: str


def log_request(entry: dict) -> None:
    """Append one line of JSON per request to the log file (our 'notebook' of stats)."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/ask")
async def ask(req: AskRequest):
    start = time.perf_counter()  # start the stopwatch

    # Step 1: check the cache first. If a similar, fresh question was answered before,
    # return that answer right away and skip the AI completely.
    cached_answer = cache.get_cached_answer(req.question)
    if cached_answer is not None:
        latency_ms = round((time.perf_counter() - start) * 1000)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "question": req.question,
            "answer": cached_answer,
            "latency_ms": latency_ms,
            "input_tokens": 0,    # the AI was never called, so no tokens were used
            "output_tokens": 0,
            "source": "cache",
        }
        log_request(entry)
        return entry

    # Step 2: cache miss -> send the question to Ollama and wait for the full answer
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": req.question}],
            "stream": False,
        })
        resp.raise_for_status()
        data = resp.json()

    latency_ms = round((time.perf_counter() - start) * 1000)  # stop the stopwatch

    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "question": req.question,
        "answer": data["message"]["content"],
        "latency_ms": latency_ms,
        "input_tokens": data.get("prompt_eval_count", 0),   # tokens in the question
        "output_tokens": data.get("eval_count", 0),         # tokens in the answer
        "source": "llm",
    }

    # Step 3: save the new answer for next time, but only if it is safe to share
    # (not personal, not time-sensitive). Saving happens after we stop the stopwatch,
    # so it doesn't count toward the latency we report.
    if cache.should_cache(req.question):
        cache.save(req.question, entry["answer"])

    log_request(entry)
    return entry


def _average(numbers: List[float]) -> Optional[float]:
    """Average of a list, or None if the list is empty (avoids dividing by zero)."""
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers))


@app.get("/stats")
def stats():
    """Quick summary of everything logged so far."""
    if not LOG_FILE.exists():
        return {"requests": 0}
    rows = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line]
    if not rows:
        return {"requests": 0}

    # Split requests by where the answer came from.
    cache_rows = [r for r in rows if r.get("source") == "cache"]
    llm_rows = [r for r in rows if r.get("source") != "cache"]

    return {
        "requests": len(rows),
        "cache_hits": len(cache_rows),
        "llm_calls": len(llm_rows),
        # Hit rate = share of requests answered from the cache (0.0 to 1.0).
        "cache_hit_rate": round(len(cache_rows) / len(rows), 3),
        "avg_latency_ms": _average([r["latency_ms"] for r in rows]),
        "avg_latency_ms_cache": _average([r["latency_ms"] for r in cache_rows]),
        "avg_latency_ms_llm": _average([r["latency_ms"] for r in llm_rows]),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
    }
