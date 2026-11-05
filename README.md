# LLM Gateway: a semantic cache for LLM calls

A small gateway that sits between an app and a large language model (LLM). When a
question means the same thing as one answered before, the gateway returns the saved
answer instead of calling the model.

**Result on a 500-question benchmark:** at a similarity threshold of 0.85, the cache answered
**38% of requests** in **~14 ms** each instead of **~50 s**, cut the estimated API cost by
**39%**, and served **0 wrong answers**.

```
            question                          cache miss
  app  ───────────────▶  gateway  ───────────────────────────▶  LLM (Ollama, llama3.2)
       ◀───────────────   │   ▲   ◀───────────────────────────
            answer        │   │            answer (saved if safe)
                          ▼   │
                   semantic cache (ChromaDB)
                   cache hit: ~14 ms, no tokens
```

## How it works

1. **Embed the question.** The question becomes a vector of 384 numbers that captures its
   meaning ([sentence-transformers](https://www.sbert.net/), `all-MiniLM-L6-v2`, runs locally on the CPU).
2. **Find the closest past question** in ChromaDB, using cosine similarity.
3. **Reuse the answer only if it is safe** (`get_cached_answer` in [cache.py](cache.py)):
   - similarity ≥ **0.85** (the threshold was picked with the benchmark below), and
   - the answer is younger than **7 days** (TTL), so answers don't go stale forever.
4. **On a miss, call the LLM.** The answer is saved only if `should_cache` allows it.
   Personal or time-sensitive questions ("my calendar", "weather today") are never saved,
   because another user would get the wrong answer.
5. **Log every request** (source, latency, tokens) to `logs/requests.jsonl`. `GET /stats`
   and the dashboard read that log.

"Semantic" means the cache matches on **meaning**, not exact text. "What is photosynthesis?"
and "Can you explain photosynthesis?" share one cached answer.

## Benchmark

[benchmark/run_benchmark.py](benchmark/run_benchmark.py) uses **500 questions in 120 groups**
([questions.json](benchmark/questions.json)):

- **90 groups × 5 paraphrases**: the same question asked different ways
- **10 groups × 3 personal / time-sensitive questions**: these should never be cached
- **20 one-off questions**, including look-alike traps such as "capital of Germany" next to
  "capital of France", or "undo a git push" next to "undo a git commit"

**Method.** The script asks Ollama once per group and records the real answer and latency.
It then replays all 500 questions in a shuffled order through the real cache code, starting
from an empty cache, once per threshold. Whether a question is a hit depends only on the
question text, so replaying gives the same hit/miss results as 2,000 live calls, from 120
real LLM calls. Cost is estimated by counting tokens with `tiktoken` and multiplying by
gpt-4o-mini prices ($0.15 / $0.60 per 1M input / output tokens, checked 2026-11-05), as if
the same traffic went to a paid API.

### Results

| Threshold | Hit rate | Wrong hits | Repeats caught | Avg latency | Cost saved |
|---|---|---|---|---|---|
| no cache | 0% | – | – | 49.7 s | – |
| 0.80 | 49.8% | **2** | 247 / 360 | 23.5 s | 52.2% |
| **0.85** ✅ | **37.8%** | **0** | 189 / 360 | 30.3 s | **39.0%** |
| 0.90 | 23.6% | 0 | 118 / 360 | 38.0 s | 23.5% |
| 0.95 | 6.4% | 0 | 32 / 360 | 46.9 s | 5.7% |

*A repeat is a question whose meaning was already in the cache. 360 of the 500 questions
were repeats, so the best possible hit rate here is 72%. A cache hit takes ~14 ms; an
Ollama call took ~50 s on a MacBook Air.*

### What I learned

- **The threshold is a trade-off between savings and correctness.** Lowering it catches more
  reworded repeats, but at 0.80 the cache started serving wrong answers:
  - "How do I undo a git **push**?" got the answer to "How can I undo a git **commit** I just made?" (similarity 0.82)
  - "When did World War **1** end?" got the answer to "What year did **WWII** end?" (similarity 0.83)

  **0.85 was the lowest threshold with zero wrong hits**, so the gateway uses it.
- **Keyword filters are simple but leaky.** `should_cache` blocks words like "my", "today"
  and "latest". The benchmark found that it:
  - **leaked** 3 time-sensitive questions about "last night's game", because "last night" wasn't
    on the list. I added it, and the benchmark now shows 0 leaks. The numbers above are after this fix.
  - **over-blocked** "What is the difference between AC and DC current?", because "current" here means electricity, not "now"
- **The embedding model is faster on the CPU than on the Mac GPU.** On Apple's GPU (MPS), each
  new question length triggered a stall of up to ~3 s. On the CPU, lookups take a steady ~10 ms.

### Limitations

- The questions and paraphrases were written by hand. Real traffic has more variety, so
  real hit rates and error rates will differ.
- Latency for a miss uses one recorded Ollama call per group, not one call per question.
- The benchmark doesn't test the TTL, because all its answers are fresh.

## Dashboard

`streamlit run dashboard.py` shows:

- **live gateway stats:** hit rate, cache vs LLM latency, recent requests
- **the benchmark:** hit rate and cost saved per threshold, wrong hits to check by hand, and the `should_cache` check

It only reads the log and results files, so it works when the gateway is off.

## Run it

Requires Python 3.9+ and [Ollama](https://ollama.com) with `llama3.2` pulled (`ollama pull llama3.2`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload           # gateway on http://localhost:8000 (try /docs)
streamlit run dashboard.py          # dashboard on http://localhost:8501
python benchmark/run_benchmark.py   # benchmark (reuses saved answers in benchmark/answers.json)
```

Ask a question:

```bash
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is photosynthesis?"}'
curl -s localhost:8000/stats
```

The response's `"source"` field is `"llm"` the first time and `"cache"` for a similar question later.

## Run with Docker

The gateway and dashboard can also run in containers, so nothing but Docker needs to be
installed. Ollama stays on the host: Docker on a Mac can't use the GPU, which would make
the model much slower inside a container.

```bash
ollama serve                    # on the host, if Ollama isn't already running
docker compose up --build       # gateway on :8000, dashboard on :8501
```

- **One image, two services.** [docker-compose.yml](docker-compose.yml) starts the gateway
  (`uvicorn`) and the dashboard (`streamlit`) from the same image.
- **Reaching Ollama.** Inside a container, `localhost` is the container itself, so the
  gateway reads `OLLAMA_URL` from the environment. It is set to
  `http://host.docker.internal:11434`, Docker's name for the host machine.
- **Data that survives restarts.** The cache lives in a Docker volume (`cache-data`), and
  logs are written to the project's `logs/` folder, which the dashboard also reads.
- **A smaller image.** The [Dockerfile](Dockerfile) installs CPU-only PyTorch, which skips
  several GB of GPU libraries, and downloads the embedding model at build time instead of
  on every start.

Stop with `Ctrl+C`, or run `docker compose down` (add `-v` to also delete the cache volume).

## Project structure

| File | What it does |
|---|---|
| [main.py](main.py) | FastAPI app. `POST /ask` checks the cache first, then calls Ollama on a miss. `GET /stats` reports hit rate and latency for cache vs LLM. |
| [cache.py](cache.py) | Semantic cache: embeddings, ChromaDB lookups, the threshold + TTL check, and the `should_cache` rules. |
| [dashboard.py](dashboard.py) | Streamlit dashboard for live stats and benchmark results. |
| [benchmark/](benchmark/) | Benchmark questions, script, recorded Ollama answers and results. |
| [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) | Container image and the two-service setup (gateway + dashboard). |

## Ideas for next steps

- Replace the keyword filter with a small classifier for personal or time-sensitive questions.
- Use a different threshold per type of question (facts vs how-to).
- Add a "was this answer helpful?" signal to catch wrong hits in real traffic.

**Built with:** Python, FastAPI, Ollama (llama3.2), ChromaDB, sentence-transformers, tiktoken, Streamlit, Docker.
