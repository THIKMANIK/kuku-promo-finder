"""
finder.py - asks the AI for promo candidates. Also contains the rule-based baseline (LLM_PROVIDER=mock).

Three AI steps:
  1. series context  (1 call per show)     : premise, characters + aliases, big reveals
  2. find promos     (1 call per episode)  : 2-4 candidates as line-id ranges
  3. cross-episode   (1 call per show)     : join strong moments from different episodes
Every raw AI response is saved unedited to outputs/raw/.
"""
import hashlib
import json
import os
import random
from pathlib import Path

from llm import parse_json
from parser import render_for_llm
from validator import MAX_LEN, MIN_LEN, cliffhanger_signal, hook_signal

PROMPTS = Path(__file__).parent / "prompts"


def _include_camera() -> bool:
    """LLM_INCLUDE_CAMERA=true in .env sends CAMERA lines to the AI too (default: false)."""
    return os.getenv("LLM_INCLUDE_CAMERA", "false").strip().lower() in {"1", "true", "yes"}


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def _save_raw(raw_dir, name, raw_text):
    if raw_dir:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        Path(raw_dir, name).write_text(raw_text, encoding="utf-8")


def _meta_path(raw_dir, name):
    return Path(raw_dir, Path(name).stem + ".meta.json")  # ep_001.json -> ep_001.meta.json


def _ask_or_reuse(llm, raw_dir, name, system, user, check_input=False, temperature=None):
    """Resumable runs: if raw/<name> already exists in this output folder, reuse that saved response
    instead of calling the AI again. Otherwise ask, and save the raw response unedited.
    The model that produced it is saved next to it (raw/<name>.meta.json) and read back on resume.
    check_input=True: only reuse if the input is exactly the same as last time (a fingerprint of system+user
    is saved in the meta file). Used where the input can change between runs (final posting order).
    Returns (parsed JSON, raw text, model name)."""
    cached = Path(raw_dir, name) if raw_dir else None
    fingerprint = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()[:16]
    if cached and cached.exists():
        raw = cached.read_text(encoding="utf-8")
        meta_file = _meta_path(raw_dir, name)
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        if check_input and meta.get("input") != fingerprint:
            print(f"  (saved {name} was made from different input, asking again)")
        else:
            try:
                data = parse_json(raw)
                model = meta.get("model", "unknown")
                print(f"  (reusing saved {name}, made by {model})")
                return data, raw, model
            except ValueError:
                print(f"  (saved {name} is not valid JSON, asking again)")
    data, raw = llm.ask_json(system, user) if temperature is None else llm.ask_json(system, user, temperature=temperature)
    _save_raw(raw_dir, name, raw)
    if raw_dir:
        meta = {"model": llm.name, **({"input": fingerprint} if check_input else {})}
        _meta_path(raw_dir, name).write_text(json.dumps(meta), encoding="utf-8")
    return data, raw, llm.name


# ---------------------------------------------------------------- 1. series context
def build_series_context(llm, episodes: list, raw_dir=None) -> dict:
    loglines = "\n".join(f"Episode {e['episode']}: {e['logline']}" for e in episodes)
    if llm.client is None:  # mock: no AI, just the loglines
        return {"premise": episodes[0]["logline"] if episodes else "", "main_characters": [], "big_reveals": []}
    data, _, _ = _ask_or_reuse(llm, raw_dir, "series_context.json", _prompt("series_context"), loglines)
    return data


def context_text(ctx: dict) -> str:
    chars = "\n".join(
        f"- {c.get('name')}: {c.get('role')} (also called: {', '.join(c.get('aliases', [])) or '-'})"
        for c in ctx.get("main_characters", [])
    )
    reveals = "\n".join(f"- Ep {r.get('episode')}: {r.get('what')}" for r in ctx.get("big_reveals", []))
    return f"SERIES PREMISE: {ctx.get('premise', '')}\nCHARACTERS:\n{chars}\nBIG REVEALS (spoilers):\n{reveals}"


# ---------------------------------------------------------------- 2. per-episode promos
def find_promos_in_episode(llm, ep: dict, ctx: dict, raw_dir=None) -> list:
    if llm.client is None:
        return [{**p, "model": llm.name} for p in baseline_promos(ep)]
    user = (
        f"{context_text(ctx)}\n\n"
        f"EPISODE {ep['episode']} (length {ep['duration']:.1f}s)\n"
        f"Logline: {ep['logline']}\n\nSCREENPLAY:\n{render_for_llm(ep, _include_camera())}"
    )
    data, _, model = _ask_or_reuse(llm, raw_dir, f"ep_{ep['episode']:03d}.json", _prompt("find_promos"), user)
    promos = data.get("promos", []) if isinstance(data, dict) else []
    for p in promos:
        p["model"] = model
        for s in p.get("segments", []):
            s["episode"] = ep["episode"]
    return promos


# ---------------------------------------------------------------- 3. cross-episode promos
def _title_line(title) -> str:
    """First line for the calls that compare promos across the show (cross-episode, final posting order)."""
    return f"SHOW TITLE: {title}\n\n" if title else ""


def find_cross_episode_promos(llm, validated: list, ctx: dict, raw_dir=None, top_n: int = 25, title=None) -> list:
    """Give the AI the strongest single-episode moments and let it join moments from different episodes."""
    if llm.client is None:
        return []
    pool = sorted(
        (p for p in validated if p["status"] != "rejected" and len(p["segments"]) == 1),
        key=lambda p: p.get("hook_score", 0) + p.get("cliffhanger_score", 0), reverse=True,
    )[:top_n]
    if len(pool) < 2:
        return []
    moments = "\n".join(
        f"- Episode {p['segments'][0]['episode']}, lines {p['segments'][0]['start_id']}-{p['segments'][0]['end_id']} "
        f"({p['duration']}s): hook = {p.get('hook')}; ends on = {p.get('cliffhanger')}"
        for p in sorted(pool, key=lambda p: (p["segments"][0]["episode"], p["segments"][0]["start_sec"]))
    )
    data, _, model = _ask_or_reuse(llm, raw_dir, "cross_episode.json", _prompt("cross_episode"),
                                   f"{_title_line(title)}{context_text(ctx)}\n\nMOMENTS:\n{moments}")
    promos = data.get("promos", []) if isinstance(data, dict) else []
    for p in promos:
        p["cross_episode"] = True
        p["model"] = model
    return promos


# ---------------------------------------------------------------- 4. final posting order
POSTING_SEEDS = (1, 2, 3)  # one posting-order call per seed; the final order is the average position


def final_posting_order(llm, ranked: list, ctx: dict, raw_dir=None, top_n: int = 20, title=None):
    """
    Orders the top promos for posting. The AI compares them in one call; to make the order stable we
    make that call once per seed in POSTING_SEEDS, each time with the promos shuffled (fixed seed, so it is
    reproducible) and re-labelled P01.. in the shuffled order, so the code rank can't leak through position.
    Final order = average position over the runs (missing from a run = position top_n + 1); ties go to the
    better code score. posting_reason comes from the run where the promo placed best.
    Returns (full ranked list, stability dict). Promos below top_n keep their code order.
    """
    top, rest = ranked[:top_n], ranked[top_n:]
    if llm.client is None or len(top) < 2:  # mock / nothing to compare: keep the code order
        return ranked, None
    seg = lambda p: " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
    positions = {i: [] for i in range(len(top))}  # index in top -> position in each run
    reasons = {i: [] for i in range(len(top))}     # index in top -> (position, run, reason)
    model = llm.name
    for run_no, seed in enumerate(POSTING_SEEDS, 1):
        order_idx = list(range(len(top)))
        random.Random(seed).shuffle(order_idx)
        ids = {f"P{k:02d}": i for k, i in enumerate(order_idx, 1)}
        listing = "\n".join(
            f"- id: {pid} | episode: {seg(top[i])} | length: {top[i]['duration']}s | hook: {top[i].get('hook')} | "
            f"cliffhanger: {top[i].get('cliffhanger')} | why it works: {top[i].get('why_it_works')}"
            for pid, i in ids.items()
        )
        data, _, model = _ask_or_reuse(llm, raw_dir, f"final_rank_{run_no}.json", _prompt("final_rank"),
                                       f"{_title_line(title)}{context_text(ctx)}\n\nCANDIDATES:\n{listing}",
                                       check_input=True, temperature=0)
        seen = []
        for item in (data.get("order", []) if isinstance(data, dict) else []):
            pid = str(item.get("id", "")).strip() if isinstance(item, dict) else ""
            if pid in ids and ids[pid] not in seen:  # unknown or repeated ids are ignored
                seen.append(ids[pid])
                reasons[ids[pid]].append((len(seen), run_no, str(item.get("reason", "")).strip()))
        for i in range(len(top)):
            positions[i].append(seen.index(i) + 1 if i in seen else top_n + 1)
        if len(seen) < len(top):
            print(f"  ! posting run {run_no} left out {len(top) - len(seen)} promo(s); they count as position {top_n + 1}")

    avg = {i: sum(v) / len(v) for i, v in positions.items()}
    order = sorted(range(len(top)), key=lambda i: (avg[i], -top[i]["final_score"], i))
    final = []
    for i in order:
        p = top[i]
        p["posting_positions"] = positions[i]
        p["posting_avg"] = round(avg[i], 2)
        if reasons[i]:
            p["posting_reason"] = min(reasons[i])[2]  # from the run where it placed best (earliest run on ties)
            p["posting_model"] = model
        final.append(p)
    top5 = order[:5]
    stability = {
        "runs": len(POSTING_SEEDS),
        "seeds": list(POSTING_SEEDS),
        "top5_agreement": sum(all(pos <= 5 for pos in positions[i]) for i in top5),
        "top10": [{"rank": r, "code_rank": top[i].get("code_rank"), "clip": seg(top[i]), "title": top[i].get("title"),
                   "positions": positions[i], "avg": round(avg[i], 2)} for r, i in enumerate(order[:10], 1)],
    }
    return final + rest, stability


# ---------------------------------------------------------------- rule-based baseline (no AI)
def baseline_promos(ep: dict, k: int = 3) -> list:
    """
    Simple rules, no AI: start on the line with the strongest hook signal, run ~45s,
    end on the line whose last 5 seconds look most unresolved. Used as (a) offline test mode and
    (b) a baseline to show what the AI adds on top of keyword rules.
    """
    lines = ep["lines"]
    starts = sorted(
        (i for i, l in enumerate(lines) if l["type"] != "CAMERA"),
        key=lambda i: hook_signal(lines[i]), reverse=True,
    )
    promos, used = [], []
    for sid in starts:
        s0 = lines[sid]["start"]
        if any(a - 15 < s0 < b for a, b in used):
            continue
        best = None
        for eid in range(sid, len(lines)):
            dur = lines[eid]["end"] - s0
            if dur > MAX_LEN - 2:
                break
            if dur < MIN_LEN + 5:
                continue
            tail = [l for l in lines[sid:eid + 1] if l["end"] >= lines[eid]["end"] - 5]
            sc = cliffhanger_signal(tail)
            if best is None or sc >= best[0]:
                best = (sc, eid)
        if best is None:
            continue
        hs = hook_signal(lines[sid])
        promos.append({
            "title": f"baseline ep{ep['episode']} #{len(promos) + 1}",
            "segments": [{"episode": ep["episode"], "start_id": sid, "end_id": best[1]}],
            "hook_id": sid,
            "hook": lines[sid]["text"][:120],
            "cliffhanger": lines[best[1]]["text"][:120],
            "why_it_works": "rule-based: strongest keyword/emotion cue at the start, most unresolved ending",
            "hook_score": round(1 + 9 * hs), "cliffhanger_score": round(1 + 9 * best[0]), "standalone_score": 5,
            "spoiler_risk": "unknown",
        })
        used.append((s0, lines[best[1]]["end"]))
        if len(promos) == k:
            break
    return promos
