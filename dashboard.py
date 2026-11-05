"""
LLM Gateway - Day 4: dashboard.

Shows two things on one page:
  1. Live gateway: what really happened in logs/requests.jsonl (every /ask call).
  2. Benchmark:    the Day 3 results in benchmark/results.json (500 test questions).

It only READS files, so it works even when the gateway server is not running.

Run with:  source .venv/bin/activate && streamlit run dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
LOG_FILE = ROOT / "logs" / "requests.jsonl"
RESULTS_FILE = ROOT / "benchmark" / "results.json"
WRONG_HITS_FILE = ROOT / "benchmark" / "wrong_hits.json"

st.set_page_config(page_title="LLM Gateway", layout="wide")
st.title("LLM Gateway")
st.caption("A semantic cache between an app and an LLM: similar questions are answered "
           "from the cache instead of calling the model.")
st.button("Refresh")  # clicking any button re-runs the script, which re-reads the files


# ---------- Part 1: live gateway ----------

st.header("Live gateway")

if not LOG_FILE.exists():
    st.info("No requests yet. Start the gateway and send a question to /ask.")
else:
    # One JSON object per line -> one row per request in a table (a DataFrame).
    rows = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line]
    log = pd.DataFrame(rows)
    # "2026-11-05T04:51:30.98+00:00" -> "2026-11-05 04:51 UTC" (easier to read)
    log["time"] = pd.to_datetime(log["time"]).dt.strftime("%Y-%m-%d %H:%M UTC")
    cache_rows = log[log["source"] == "cache"]
    llm_rows = log[log["source"] != "cache"]

    # Top row of big numbers
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Requests", len(log))
    col2.metric("Cache hit rate", f"{len(cache_rows) / len(log):.0%}")
    col3.metric("Avg latency: cache",
                f"{cache_rows['latency_ms'].mean():,.0f} ms" if len(cache_rows) else "n/a")
    col4.metric("Avg latency: LLM",
                f"{llm_rows['latency_ms'].mean() / 1000:,.1f} s" if len(llm_rows) else "n/a")

    left, right = st.columns(2)
    with left:
        st.subheader("Average latency (ms)")
        st.bar_chart(log.groupby("source")["latency_ms"].mean())
    with right:
        st.subheader("Where answers came from")
        st.bar_chart(log["source"].value_counts())

    st.subheader("Recent requests")
    recent = log[["time", "question", "source", "latency_ms", "output_tokens"]].tail(20)
    st.dataframe(recent.iloc[::-1], hide_index=True, width="stretch")  # newest first


# ---------- Part 2: benchmark results ----------

st.header("Benchmark: which similarity threshold is best?")

if not RESULTS_FILE.exists():
    st.info("No benchmark results yet. Run: python benchmark/run_benchmark.py")
else:
    bench = json.loads(RESULTS_FILE.read_text())
    if bench["questions"] < 500:
        st.warning(f"These are quick-test results ({bench['questions']} questions). "
                   "Run the full benchmark for the real numbers.")

    st.write(f"{bench['questions']} questions in {bench['groups']} groups, replayed through "
             f"the cache at each threshold. Answers from **{bench['model']}** via Ollama. "
             f"Cost estimated at **{bench['price_model']}** prices "
             # "\$" because Streamlit reads $...$ as a math formula
             f"(\\${bench['price_per_1m_input_tokens']} / "
             f"\\${bench['price_per_1m_output_tokens']} per 1M input/output tokens, "
             f"checked: {bench['price_checked_on']}).")

    results = pd.DataFrame(bench["results"])
    results["hit_rate_pct"] = results["hit_rate"] * 100
    # Text labels ("0.80") so charts show only the 4 tested thresholds, not 0.81, 0.82...
    results["threshold"] = results["threshold"].map("{:.2f}".format)

    # The trade-off in two charts: a LOWER threshold = more hits but more wrong hits.
    left, right = st.columns(2)
    with left:
        st.subheader("Hit rate and cost saved (%)")
        st.line_chart(results.set_index("threshold")[["hit_rate_pct", "cost_saved_pct"]])
    with right:
        st.subheader("Wrong hits (wrong answer served)")
        st.bar_chart(results.set_index("threshold")["wrong_hits"])

    st.subheader("All numbers")
    table = results[["threshold", "hit_rate_pct", "wrong_hits", "correct_hits", "repeats",
                     "avg_latency_ms", "avg_cache_hit_latency_ms", "cost_saved_pct"]]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Without a cache: average latency "
               f"{bench['baseline_avg_latency_ms'] / 1000:,.1f} s per request. "
               "'repeats' = questions whose meaning was already in the cache; "
               "'correct_hits' = how many of those the cache caught.")

    # Spot-check: every time the cache answered with a DIFFERENT question's answer.
    st.subheader("Wrong hits to check by hand")
    wrong = json.loads(WRONG_HITS_FILE.read_text()) if WRONG_HITS_FILE.exists() else []
    if wrong:
        st.dataframe(pd.DataFrame(wrong), hide_index=True, width="stretch")
    else:
        st.success("No wrong hits.")

    # How well the keyword filter in should_cache() did against the labels.
    st.subheader("should_cache check")
    check = bench["should_cache_check"]
    left, right = st.columns(2)
    with left:
        st.write(f"**Leaked** (personal questions it would cache): {len(check['leaked'])}")
        for q in check["leaked"]:
            st.write(f"- {q}")
    with right:
        st.write(f"**Over-blocked** (normal questions it refused): {len(check['over_blocked'])}")
        for q in check["over_blocked"]:
            st.write(f"- {q}")
