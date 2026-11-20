"""
ranker.py - one score per promo, then a posting order for the whole show. Pure code: weights are
visible and easy to tune, instead of hidden inside a prompt.

final = 0.35*hook + 0.30*cliffhanger + 0.25*standalone          (AI scores, 1-10)
      + 0.10*10*avg(rule hook signal, rule cliff signal)          (code check of the same thing)
      - spoiler penalty (high 1.5, medium 0.5)                    (don't burn the big twists)
      - short-clip penalty (< 25s: 0.5)
      + early-episode bonus (ep 1-10: +0.5)                       (a new viewer lands on ep 1, so
                                                                   promos from the opening convert best)
Then: drop near-duplicates (same moment found twice), max 2 promos per episode in the list.
"""
W_HOOK, W_CLIFF, W_STAND, W_RULES = 0.35, 0.30, 0.25, 0.10
SPOILER_PENALTY = {"high": 1.5, "medium": 0.5}
EARLY_EPISODES, EARLY_BONUS = 10, 0.5
MAX_PER_EPISODE = 2
SHORT_CLIP_SEC, SHORT_PENALTY = 25, 0.5                         # short clips rarely set up AND pay off


def _num(x, default=5.0):
    try:
        return max(1.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def score(p: dict) -> dict:
    hook, cliff, stand = _num(p.get("hook_score")), _num(p.get("cliffhanger_score")), _num(p.get("standalone_score"))
    rules = 10 * (p.get("rule_hook_signal", 0) + p.get("rule_cliff_signal", 0)) / 2
    s = W_HOOK * hook + W_CLIFF * cliff + W_STAND * stand + W_RULES * rules
    reasons = [f"hook {hook:g}, cliffhanger {cliff:g}, standalone {stand:g}"]
    risk = str(p.get("spoiler_risk", "")).lower().split()[0] if p.get("spoiler_risk") else ""
    if risk in SPOILER_PENALTY:
        s -= SPOILER_PENALTY[risk]
        reasons.append(f"{risk} spoiler risk")
    if p.get("duration", SHORT_CLIP_SEC) < SHORT_CLIP_SEC:
        s -= SHORT_PENALTY
        reasons.append(f"short clip (<{SHORT_CLIP_SEC}s)")
    first_ep = p["segments"][0]["episode"]
    if first_ep <= EARLY_EPISODES:
        s += EARLY_BONUS
        reasons.append("from the opening episodes (viewer can start right there)")
    if p.get("cross_episode"):
        reasons.append("joins moments across episodes")
    return {**p, "final_score": round(s, 2), "rank_reason": "; ".join(reasons)}


def _overlap(a: dict, b: dict) -> float:
    """Share of the shorter promo that overlaps the other (same episode only)."""
    total = 0.0
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["episode"] == sb["episode"]:
                total += max(0.0, min(sa["end_sec"], sb["end_sec"]) - max(sa["start_sec"], sb["start_sec"]))
    shorter = min(a["duration"], b["duration"]) or 1
    return total / shorter


def rank(promos: list) -> list:
    good = sorted((score(p) for p in promos if p["status"] != "rejected"),
                  key=lambda p: p["final_score"], reverse=True)
    ranked, per_ep = [], {}
    for p in good:
        # a montage is a different format: cross-episode promos are only de-duplicated against each other
        # and don't use up the 2-per-episode slots of the episodes they borrow from
        if any(_overlap(p, q) > 0.5 for q in ranked if bool(p.get("cross_episode")) == bool(q.get("cross_episode"))):
            continue
        eps = {s["episode"] for s in p["segments"]}
        if not p.get("cross_episode") and any(per_ep.get(e, 0) >= MAX_PER_EPISODE for e in eps):
            continue
        if not p.get("cross_episode"):
            for e in eps:
                per_ep[e] = per_ep.get(e, 0) + 1
        ranked.append(p)
    for i, p in enumerate(ranked, 1):
        p["rank"] = i
    return ranked
