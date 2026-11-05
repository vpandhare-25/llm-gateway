"""
LLM Gateway - Day 2: the semantic cache (the assistant's "notebook").

SETUP PARTS (already written): loading the embedding model, the ChromaDB database,
saving answers, and finding the closest past question.

DECISION PARTS (bottom of file): when to reuse an answer, and when to save one.
These are what interviewers will ask about.
"""
import re
import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Settings (tuned with the Day 3 benchmark) ----------
SIMILARITY_THRESHOLD = 0.85   # how similar two questions must be to count as "the same" (0 to 1).
                              # Chosen from the Day 3 benchmark: 0.85 had 0 wrong hits on 500 questions;
                              # 0.80 caught more repeats but confused e.g. World War 1 with World War 2.
TTL_SECONDS = 7 * 24 * 3600   # how long a saved answer stays valid (here: 7 days)

# ---------- Setup (done for you) ----------
# Turns text into a list of 384 numbers that capture its meaning. Runs free on your laptop.
# device="cpu": this small model is faster on the CPU. On a Mac the default is the GPU
# ("mps"), which stalls for up to a few seconds whenever it sees a new question length.
_embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# A small database saved in the "chroma_db" folder. "cosine" = compare meaning by angle.
_db = chromadb.PersistentClient(path="chroma_db")
_collection = _db.get_or_create_collection("answers", metadata={"hnsw:space": "cosine"})


def _embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def find_closest(question: str):
    """
    Look up the single most similar past question.
    Returns None if the notebook is empty, otherwise a dict:
      {"question": ..., "answer": ..., "similarity": 0.0-1.0, "saved_at": unix_time}
    """
    if _collection.count() == 0:
        return None
    result = _collection.query(query_embeddings=[_embed(question)], n_results=1)
    meta = result["metadatas"][0][0]
    distance = result["distances"][0][0]
    return {
        "question": result["documents"][0][0],
        "answer": meta["answer"],
        "similarity": 1 - distance,   # cosine distance -> similarity
        "saved_at": meta["saved_at"],
    }


def save(question: str, answer: str) -> None:
    """Write a question + answer into the notebook."""
    _collection.add(
        ids=[str(time.time_ns())],
        documents=[question],
        embeddings=[_embed(question)],
        metadatas=[{"answer": answer, "saved_at": time.time()}],
    )


# ---------- YOUR CODE ----------

def get_cached_answer(question: str) -> Optional[str]:
    """
    Decide if we can reuse an old answer.

    Returns the saved answer (a string) if it is safe to reuse, otherwise None.
    A saved answer is reused only if BOTH are true:
      - the saved question means almost the same thing (similarity >= threshold)
      - the saved answer is not too old (younger than TTL_SECONDS)
    """
    closest = find_closest(question)

    # 1. Nothing saved yet -> nothing to reuse.
    if closest is None:
        return None

    # 2. The closest question is not similar enough -> treat it as a new question.
    if closest["similarity"] < SIMILARITY_THRESHOLD:
        return None

    # 3. The answer is too old -> it might be out of date, so ask the AI again.
    age_seconds = time.time() - closest["saved_at"]
    if age_seconds > TTL_SECONDS:
        return None

    # 4. Similar enough and fresh enough -> cache hit!
    return closest["answer"]


# Words that suggest a question is about ONE person (so the answer shouldn't be
# shared with others) or about a specific moment in time (so the answer goes stale fast).
PERSONAL_WORDS = {"my", "me", "mine", "myself", "i'm", "i've", "our", "ours"}
TIME_WORDS = {
    "today", "tonight", "tomorrow", "yesterday", "now", "currently", "current",
    "latest", "recent", "recently", "this week", "this month", "this year",
    "last night",  # added after the benchmark: "Who won the game last night?" was being cached
}


def should_cache(question: str) -> bool:
    """
    Decide if this question is safe to save and reuse for other people.

    Returns False for personal or time-sensitive questions, True otherwise.
    This is a simple keyword check. We match WHOLE words, so "my" matches
    "reset my password" but not "mystery" or "economy".
    """
    text = question.lower()
    for phrase in PERSONAL_WORDS | TIME_WORDS:
        # \b means "word boundary", so we only match the whole word/phrase.
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return False
    return True
