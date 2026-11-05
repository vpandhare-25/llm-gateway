"""
LLM Gateway - Day 3: benchmark the semantic cache.

Question: how much time and money does the cache save, and how often does it give
a WRONG answer (a "wrong hit": a question gets the cached answer of a different question)?

How it works, in two phases:

  Phase 1 - real answers (slow, done once):
    Ask Ollama ONE question per group and save the answer, how long it took, and the
    answer text to answers.json. If the script is stopped, it picks up where it left off.

  Phase 2 - replay (fast, done for every threshold):
    Start with an EMPTY cache, then replay all questions in a shuffled order through the
    real cache code in cache.py (get_cached_answer, should_cache, save).
    - Hit:  the answer came from the cache. We time the real lookup.
    - Miss: we would have called the AI. Instead of calling it again, we use the answer
            and latency we recorded in phase 1.

Why replay instead of calling the AI 500 x 4 = 2,000 times? Whether a question is a hit
or a miss depends only on the question text, not on the AI. So replaying gives the same
hit/miss results in about a minute instead of many hours.

Run from the project folder:
    python benchmark/run_benchmark.py              # full run (about 120 Ollama calls, once)
    python benchmark/run_benchmark.py --groups 8   # quick test with 8 groups
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import httpx
import tiktoken

# Let this script import cache.py and main.py from the project folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cache  # noqa: E402  (the real cache code we are testing)
from main import MODEL, OLLAMA_URL  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions.json"
ANSWERS_FILE = HERE / "answers.json"      # phase 1 output: real Ollama answers
RESULTS_FILE = HERE / "results.json"      # summary numbers (for the Day 4 dashboard)
WRONG_HITS_FILE = HERE / "wrong_hits.json"  # every wrong hit, to spot-check by hand

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
SEED = 42  # fixed seed = the same shuffled order every run, so results are repeatable

# ---------- Cost estimate ----------
# Ollama is free, so we estimate what the SAME traffic would cost on a paid API.
# Tokens are counted with tiktoken (the tokenizer OpenAI models use), then multiplied
# by the published price. Prices change: check https://openai.com/api/pricing and
# update these numbers (and the date) before putting results in the README.
PRICE_MODEL = "gpt-4o-mini"
PRICE_PER_1M_INPUT_TOKENS = 0.15    # US dollars per 1 million input tokens
PRICE_PER_1M_OUTPUT_TOKENS = 0.60   # US dollars per 1 million output tokens
PRICE_CHECKED_ON = "not yet checked"
_tokenizer = tiktoken.get_encoding("o200k_base")  # tokenizer used by gpt-4o models


# ---------- Loading the questions ----------

def load_questions(max_groups: Optional[int]) -> List[Dict]:
    """
    Read questions.json and return one flat, shuffled list like:
      [{"question": "...", "group": "capital_france", "type": "repeat"}, ...]
    max_groups: if set, only use that many randomly picked groups (for a quick test).
    """
    groups = json.loads(QUESTIONS_FILE.read_text())["groups"]
    rng = random.Random(SEED)
    if max_groups:
        groups = rng.sample(groups, max_groups)

    questions = []
    for g in groups:
        for text in g["questions"]:
            questions.append({"question": text, "group": g["id"], "type": g["type"]})

    # The same text in two groups would make "was this hit right?" ambiguous.
    texts = [q["question"] for q in questions]
    if len(texts) != len(set(texts)):
        raise ValueError("questions.json has the same question in more than one group")

    rng.shuffle(questions)  # mix the paraphrases in, like real traffic
    return questions


# ---------- Phase 1: real answers from Ollama ----------

def ask_ollama(question: str) -> Dict:
    """Ask Ollama one question and time it (same request main.py sends)."""
    start = time.perf_counter()
    resp = httpx.post(OLLAMA_URL, timeout=300, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    })
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000
    return {"question": question, "answer": resp.json()["message"]["content"],
            "latency_ms": round(latency_ms)}


def fetch_answers(questions: List[Dict]) -> Dict[str, Dict]:
    """
    Make sure we have one real answer per group. Answers already in answers.json are
    reused, so the slow part only ever happens once.
    """
    answers = json.loads(ANSWERS_FILE.read_text()) if ANSWERS_FILE.exists() else {}

    # The first question of each group in the shuffled order is the one that would
    # really reach the AI (it is always a miss), so that is the one we ask.
    first_question = {}
    for q in questions:
        first_question.setdefault(q["group"], q["question"])

    todo = [g for g in first_question if g not in answers]
    for i, group in enumerate(todo, start=1):
        print(f"  [{i}/{len(todo)}] asking Ollama: {first_question[group]}")
        answers[group] = ask_ollama(first_question[group])
        ANSWERS_FILE.write_text(json.dumps(answers, indent=2))  # save after each one
    return answers


# ---------- Cost helpers ----------

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def llm_cost(question: str, answer: str) -> float:
    """Estimated dollars for one AI call: input tokens + output tokens, times price."""
    return (count_tokens(question) * PRICE_PER_1M_INPUT_TOKENS
            + count_tokens(answer) * PRICE_PER_1M_OUTPUT_TOKENS) / 1_000_000


# ---------- Phase 2: replay at one threshold ----------

def run_threshold(threshold: float, questions: List[Dict], answers: Dict[str, Dict],
                  client) -> Dict:
    """Replay every question through a fresh, empty cache at this threshold."""
    # Point cache.py at an empty in-memory collection and set the threshold to test.
    # get_cached_answer() reads these module settings each time it runs.
    name = f"bench_{int(threshold * 100)}"
    cache._collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    cache.SIMILARITY_THRESHOLD = threshold

    group_of = {q["question"]: q["group"] for q in questions}
    groups_in_cache = set()   # groups whose answer is already saved
    hits = correct_hits = repeats = missed_repeats = 0
    latencies, cache_latencies = [], []
    cost = 0.0
    wrong_hits = []

    for q in questions:
        text, group = q["question"], q["group"]
        recorded = answers[group]  # the real Ollama answer + latency from phase 1

        # A "repeat" = a question whose meaning is already in the cache.
        # A perfect cache would answer every repeat from the cache.
        is_repeat = group in groups_in_cache
        repeats += is_repeat

        start = time.perf_counter()
        cached_answer = cache.get_cached_answer(text)   # the real cache lookup
        lookup_ms = (time.perf_counter() - start) * 1000

        if cached_answer is not None:
            # HIT: answered from the cache. Was it the right group's answer?
            hits += 1
            latencies.append(lookup_ms)
            cache_latencies.append(lookup_ms)
            match = cache.find_closest(text)
            if group_of[match["question"]] == group:
                correct_hits += 1
            else:
                wrong_hits.append({
                    "threshold": threshold,
                    "question": text,
                    "matched_question": match["question"],
                    "similarity": round(match["similarity"], 3),
                })
        else:
            # MISS: the gateway would call the AI (lookup time + recorded AI time).
            missed_repeats += is_repeat
            latencies.append(lookup_ms + recorded["latency_ms"])
            cost += llm_cost(text, recorded["answer"])
            if cache.should_cache(text):
                cache.save(text, recorded["answer"])
                groups_in_cache.add(group)

    client.delete_collection(name)

    n = len(questions)
    return {
        "threshold": threshold,
        "hit_rate": round(hits / n, 3),
        "hits": hits,
        "correct_hits": correct_hits,
        "wrong_hits": len(wrong_hits),
        # Of the questions that COULD have been answered from the cache, how many were?
        "repeats": repeats,
        "missed_repeats": missed_repeats,
        "repeat_recall": round(correct_hits / repeats, 3) if repeats else None,
        "avg_latency_ms": round(sum(latencies) / n),
        "avg_cache_hit_latency_ms": (round(sum(cache_latencies) / len(cache_latencies), 1)
                                     if cache_latencies else None),
        "cost_usd": cost,
        "wrong_hit_examples": wrong_hits,
    }


# ---------- should_cache spot-check ----------

def check_should_cache(questions: List[Dict]) -> Dict[str, List[str]]:
    """
    Compare should_cache() to the labels in questions.json:
      leaked:       personal/time-sensitive questions it WOULD save (bad: wrong answers later)
      over_blocked: normal questions it would NOT save (only costs some cache hits)
    """
    leaked = [q["question"] for q in questions
              if q["type"] == "personal" and cache.should_cache(q["question"])]
    over_blocked = [q["question"] for q in questions
                    if q["type"] != "personal" and not cache.should_cache(q["question"])]
    return {"leaked": leaked, "over_blocked": over_blocked}


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the semantic cache.")
    parser.add_argument("--groups", type=int, default=None,
                        help="only use this many question groups (quick test)")
    args = parser.parse_args()

    questions = load_questions(args.groups)
    print(f"{len(questions)} questions in {len({q['group'] for q in questions})} groups")

    print("Phase 1: getting real answers from Ollama (only for groups not saved yet)")
    answers = fetch_answers(questions)

    # Baseline = no cache at all: every question goes to the AI.
    baseline_cost = sum(llm_cost(q["question"], answers[q["group"]]["answer"])
                        for q in questions)
    baseline_latency = sum(answers[q["group"]]["latency_ms"] for q in questions) / len(questions)

    print("Phase 2: replaying through the cache at each threshold")
    client = chromadb.EphemeralClient()  # in-memory database: the real cache is untouched
    results = []
    for t in THRESHOLDS:
        print(f"  threshold {t:.2f} ...")
        results.append(run_threshold(t, questions, answers, client))

    filter_check = check_should_cache(questions)

    # ---- Print a summary table ----
    n = len(questions)
    print()
    print(f"No cache: avg latency {baseline_latency:,.0f} ms, "
          f"est. cost ${baseline_cost / n * 1000:.4f} per 1,000 requests ({PRICE_MODEL})")
    print()
    print(f"{'threshold':>9} | {'hit rate':>8} | {'wrong hits':>10} | {'repeats caught':>14} | "
          f"{'avg latency':>11} | {'cost saved':>10}")
    print("-" * 80)
    for r in results:
        saved = 1 - r["cost_usd"] / baseline_cost if baseline_cost else 0
        caught = f"{r['correct_hits']}/{r['repeats']}"
        print(f"{r['threshold']:>9.2f} | {r['hit_rate']:>8.1%} | {r['wrong_hits']:>10} | "
              f"{caught:>14} | {r['avg_latency_ms']:>8,} ms | {saved:>10.1%}")

    print()
    print("Wrong hits (first 3 per threshold) - check these by hand:")
    for r in results:
        for w in r["wrong_hit_examples"][:3]:
            print(f"  [{w['threshold']:.2f}] \"{w['question']}\" got the answer to "
                  f"\"{w['matched_question']}\" (similarity {w['similarity']})")

    print()
    print(f"should_cache leaked {len(filter_check['leaked'])} personal/time-sensitive questions:",
          filter_check["leaked"])
    print(f"should_cache over-blocked {len(filter_check['over_blocked'])} normal questions:",
          filter_check["over_blocked"])

    # ---- Save everything for the README and the Day 4 dashboard ----
    all_wrong_hits = [w for r in results for w in r.pop("wrong_hit_examples")]
    for r in results:
        r["cost_saved_pct"] = round(100 * (1 - r["cost_usd"] / baseline_cost), 1)
        r["cost_usd"] = round(r["cost_usd"], 6)
    RESULTS_FILE.write_text(json.dumps({
        "questions": n,
        "groups": len({q["group"] for q in questions}),
        "model": MODEL,
        "price_model": PRICE_MODEL,
        "price_per_1m_input_tokens": PRICE_PER_1M_INPUT_TOKENS,
        "price_per_1m_output_tokens": PRICE_PER_1M_OUTPUT_TOKENS,
        "price_checked_on": PRICE_CHECKED_ON,
        "baseline_avg_latency_ms": round(baseline_latency),
        "baseline_cost_usd": round(baseline_cost, 6),
        "results": results,
        "should_cache_check": filter_check,
    }, indent=2))
    WRONG_HITS_FILE.write_text(json.dumps(all_wrong_hits, indent=2))
    print()
    print(f"Saved {RESULTS_FILE.name} and {WRONG_HITS_FILE.name} in the benchmark folder.")


if __name__ == "__main__":
    main()
