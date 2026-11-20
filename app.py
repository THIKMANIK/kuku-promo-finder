"""
app.py - small web page for the promo finder.   Run:  streamlit run app.py

Upload one or more episode screenplays (.txt) -> pick a model -> get ranked promos.
"""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from llm import LLM, PROVIDERS
from parser import parse_episode, to_timestamp
from run import run

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Microdrama Promo Finder")
st.caption("Reads screenplays, finds 20-60s promo clips that open on a hook and end on a cliffhanger, ranks them.")

with st.sidebar:
    provider = st.selectbox("Model provider", list(PROVIDERS) + ["mock"], index=0,
                            help="mock = rule-based baseline, no AI, no key needed")
    model = st.text_input("Model name (optional)", "")
    cross = st.checkbox("Also build cross-episode promos", value=True)
    title = st.text_input("Show title", os.getenv("SHOW_TITLE", ""),
                          help="Used when comparing promos across the show (cross-episode and posting order)")

files = st.file_uploader("Episode screenplays (.txt)", type="txt", accept_multiple_files=True)

if files and st.button("Find promos", type="primary"):
    tmp = Path(tempfile.mkdtemp())
    episodes = []
    for f in files:
        p = tmp / f.name
        p.write_bytes(f.getvalue())
        episodes.append(parse_episode(p))
    episodes.sort(key=lambda e: e["episode"])
    try:
        llm = LLM(provider, model or None)
    except Exception as e:
        st.error(str(e))
        st.stop()
    log_box = st.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        log_box.code("\n".join(logs[-12:]))

    with st.spinner(f"Running {llm.name} on {len(episodes)} episode(s)..."):
        result = run(episodes, llm, tmp / "out", cross=cross, log=log, title=title.strip() or None)
    if result.get("stopped"):
        st.error("The model's daily limit was reached before any episode was done. Try again after it resets.")
        st.stop()
    st.session_state["result"] = result
    st.session_state["episodes"] = {e["episode"]: e for e in episodes}

if "result" in st.session_state:
    res, eps = st.session_state["result"], st.session_state["episodes"]
    ranked, stats = res["ranked"], res["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ranked promos", len(ranked))
    c2.metric("Candidates checked", stats["candidates"])
    c3.metric("Rejected by checks", stats["status_counts"].get("rejected", 0))
    c4.metric("AI calls", stats["llm_calls"])

    table = pd.DataFrame([{
        "rank": p["rank"], "score": p["final_score"],
        "clip": " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"]),
        "sec": p["duration"], "hook": p.get("hook"), "cliffhanger": p.get("cliffhanger"),
        "why": p.get("why_it_works"),
    } for p in ranked])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Look inside a promo")
    if ranked:
        pick = st.selectbox("Promo", ranked, format_func=lambda p: f"#{p['rank']} {p.get('title', '')}")
        st.write(f"**Why this rank:** {pick['rank_reason']}")
        if pick.get("issues"):
            st.warning("Auto-fixes: " + "; ".join(pick["issues"]))
        for s in pick["segments"]:
            st.markdown(f"**Episode {s['episode']}  {s['start']} - {s['end']}**")
            lines = [l for l in eps[s["episode"]]["lines"]
                     if s["start_sec"] - 0.01 <= l["start"] < s["end_sec"] and l["type"] != "CAMERA"]
            st.code("\n".join(f"{to_timestamp(l['start'])}  {l['speaker'] or l['type']}: {l['text']}" for l in lines))

    st.download_button("Download ranked promos (JSON)", json.dumps(ranked, indent=2), "promos_ranked.json")
    st.download_button("Download ranked promos (CSV)", table.to_csv(index=False), "promos_ranked.csv")
