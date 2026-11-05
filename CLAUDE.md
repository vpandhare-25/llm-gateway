# LLM Gateway

A semantic caching gateway that sits between an app and an LLM. Repeat or similar
questions are answered from a cache instead of calling the model, saving cost and time.
Resume project for a CS student. Keep code simple, readable, and well commented.

## Stack
- Python 3.9, FastAPI, uvicorn, httpx
- Ollama running locally (model: llama3.2) at http://localhost:11434
- ChromaDB (persistent, cosine) + sentence-transformers (all-MiniLM-L6-v2) for the cache
- Later: Streamlit dashboard

## Files
- main.py: FastAPI app. POST /ask forwards to Ollama and logs to logs/requests.jsonl. GET /stats summarizes logs.
- cache.py: semantic cache. find_closest() and save() are done.
  get_cached_answer() and should_cache() are TODO.

## Plan
- Day 2: implement get_cached_answer (similarity threshold + TTL) and should_cache
  (skip personal/time-sensitive questions), then wire cache into /ask: check cache first,
  return with source="cache" on hit; on miss call Ollama and save if should_cache.
  /stats should report cache hit rate and latency for cache vs llm.
- Day 3: benchmark script: ~500 questions with paraphrased repeats; measure hit rate,
  latency, estimated cost (tokens x published API price, e.g. via tiktoken), across
  thresholds 0.80 / 0.85 / 0.90 / 0.95. Also spot-check wrong cache hits.
- Day 4: Streamlit dashboard + README with results.

## Rules
- Python 3.9: use typing compatible syntax (no `X | None`).
- After each change, explain what it does in simple words so the student can explain it in interviews.
- Run with: source .venv/bin/activate && uvicorn main:app --reload
