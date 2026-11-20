"""
validator.py - checks every promo the AI proposes. Pure code, no AI.

The AI only returns line ids. Here we:
  1. turn ids into real timestamps from the parsed screenplay (so timestamps can never be invented),
  2. check the hard rules (20-60s, hook in first 3s, segments in order, no cut mid-dialogue),
  3. auto-fix small problems (snap a cut so it does not chop a line of dialogue),
  4. add simple rule-based "signal" scores that we combine with the AI's scores in the ranker.
Every problem found is recorded, so we can report how often the AI gets it wrong.
"""
import re

from parser import to_timestamp

MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0

HOOK_WORDS = re.compile(
    r"\b(slap|slaps|strike|strikes|struck|punch|kiss|kisses|blood|bleed|knife|blade|sword|gun|fire|flames|explo\w*|"
    r"scream\w*|gasp\w*|shatter\w*|crash\w*|grab\w*|seize\w*|chain\w*|poison\w*|arsenic|hang|execut\w*|witch\w*|"
    r"betray\w*|marry|married|wedding|pregnant|dead|die|kill\w*|duke|duchess|arrest\w*|guards?|collaps\w*)\b",
    re.IGNORECASE,
)
STRONG_EMOTIONS = re.compile(
    r"shock|furious|rage|outrag|scream|gloat|smug|cruel|mock|venom|desperat|horrif|devastat|threat|menac|cold|"
    r"accus|defian|seeth|panick|terrif|sneer|vicious",
    re.IGNORECASE,
)


def hook_signal(line: dict) -> float:
    """0-1: how 'stop-the-scroll' a single line looks, from surface cues only."""
    score = 0.0
    text = line["text"]
    if HOOK_WORDS.search(text):
        score += 0.4
    if line["type"] == "DIALOGUE":
        score += 0.2
        if line["emotion"] and STRONG_EMOTIONS.search(line["emotion"]):
            score += 0.3
        if text.rstrip().endswith(("!", "?")):
            score += 0.1
    if line["type"] in ("SOUND FX", "ACTION") and HOOK_WORDS.search(text):
        score += 0.2
    if line["type"] == "CAMERA" and re.search(r"establishing|wide shot of the|exterior", text, re.I):
        score -= 0.3
    return max(0.0, min(1.0, score))


def cliffhanger_signal(lines: list) -> float:
    """0-1: does the last ~5 seconds look unresolved?"""
    tail = " ".join(l["text"] for l in lines)
    score = 0.0
    if re.search(r"cliffhanger|ominous|cut to black|freeze|swells|sting", tail, re.I):
        score += 0.4
    if re.search(r"\?\s*$|\.\.\.|—\s*$", lines[-1]["text"]) or any(l["text"].rstrip().endswith("?") for l in lines):
        score += 0.3
    if HOOK_WORDS.search(tail):
        score += 0.3
    return min(1.0, score)


def _snap_segment(ep: dict, start_id: int, end_id: int, issues: list):
    """Turn a line-id range into (start_sec, end_sec, lines). Extend the cut so no dialogue is chopped."""
    lines = ep["lines"]
    n = len(lines)
    if not (0 <= start_id < n and 0 <= end_id < n):
        issues.append(f"ep{ep['episode']}: line id out of range ({start_id}-{end_id}, episode has {n})")
        return None
    if end_id < start_id:
        issues.append(f"ep{ep['episode']}: end_id before start_id, swapped")
        start_id, end_id = end_id, start_id
    seg = lines[start_id:end_id + 1]
    start = seg[0]["start"]
    end = max(l["end"] for l in seg)
    # a dialogue line that starts inside the clip but finishes after the cut -> extend the cut
    for l in lines[end_id + 1:]:
        if l["start"] >= end:
            break
        if l["type"] == "DIALOGUE" and l["end"] > end:
            issues.append(f"ep{ep['episode']}: cut chopped dialogue at {to_timestamp(end)}, extended to {to_timestamp(l['end'])}")
            end = l["end"]
    # a dialogue line already running when the clip starts -> move start back to its beginning
    for l in lines[:start_id]:
        if l["type"] == "DIALOGUE" and l["start"] < start < l["end"]:
            issues.append(f"ep{ep['episode']}: clip started mid-dialogue, moved start to {to_timestamp(l['start'])}")
            start = l["start"]
    in_clip = [l for l in lines if l["start"] >= start - 0.01 and l["start"] < end]
    return start, end, in_clip


CROSS_TARGET, CROSS_EDGE = 55.0, 18.0  # trimmed cross-episode promo: total length, max first/last segment


def _seg_len(ep: dict, sid: int, eid: int) -> float:
    """Length of a line-id range after the usual snapping (dialogue is never chopped)."""
    snapped = _snap_segment(ep, sid, eid, [])
    return snapped[1] - snapped[0] if snapped else float("inf")


def _keep_from_start(ep, sid, eid, budget):
    """Longest range sid..e (e <= eid) that fits the budget. Always keeps at least one line."""
    best = sid
    for e in range(sid, eid + 1):
        if _seg_len(ep, sid, e) > budget:
            break
        best = e
    return sid, best


def _keep_from_end(ep, sid, eid, budget):
    """Longest range s..eid (s >= sid) that fits the budget. Always keeps at least one line."""
    best = eid
    for s in range(eid, sid - 1, -1):
        if _seg_len(ep, s, eid) > budget:
            break
        best = s
    return best, eid


def fit_cross_episode(promo: dict, episodes_by_num: dict, target: float = CROSS_TARGET, edge: float = CROSS_EDGE) -> dict:
    """
    Cross-episode promos often join whole clips and run far past 60s. Trim an over-long one into ~55s:
      first segment  : keep from its start (the hook is there), max ~18s
      last segment   : keep back from its end (the cliffhanger is there), max ~18s
      middle segments: keep from their start, sharing the time that is left
    Cuts are on line boundaries and measured after snapping, so dialogue is never chopped.
    Anything it can't read is returned unchanged (validate() then rejects it as usual).
    """
    segs = promo.get("segments", [])
    if len(segs) < 2:
        return promo
    try:
        parts = [(episodes_by_num[int(s["episode"])], int(s["start_id"]), int(s["end_id"])) for s in segs]
    except (KeyError, TypeError, ValueError):
        return promo
    if any(not (0 <= a <= b < len(ep["lines"])) for ep, a, b in parts):
        return promo
    before = sum(_seg_len(ep, a, b) for ep, a, b in parts)
    if before <= MAX_LEN:
        return promo
    first = _keep_from_start(*parts[0], edge)
    last = _keep_from_end(*parts[-1], edge)
    kept = [first]
    used = _seg_len(parts[0][0], *first) + _seg_len(parts[-1][0], *last)
    middles = parts[1:-1]
    for i, (ep, a, b) in enumerate(middles):
        share = max(0.0, target - used) / (len(middles) - i)  # split what is left over the remaining middles
        rng = _keep_from_start(ep, a, b, share)
        used += _seg_len(ep, *rng)
        kept.append(rng)
    kept.append(last)
    new_segs = [{**s, "start_id": a, "end_id": b} for s, (a, b) in zip(segs, kept)]
    after = sum(_seg_len(ep, a, b) for (ep, _, _), (a, b) in zip(parts, kept))
    note = f"trimmed cross-episode promo from {round(before, 1)}s to {round(after, 1)}s"
    return {**promo, "segments": new_segs, "pre_issues": list(promo.get("pre_issues", [])) + [note]}


def validate(promo: dict, episodes_by_num: dict, default_episode: int | None = None) -> dict:
    """Return the promo enriched with timestamps, duration, checks, issues and status (ok / fixed / rejected)."""
    promo = dict(promo)
    issues, segments_out, all_lines = promo.pop("pre_issues", []), [], []
    hard_fail = False
    prev = None
    segs = [dict(s) for s in promo.get("segments", []) if isinstance(s, dict)]
    # auto-fix: the AI often starts the clip a few seconds BEFORE its own hook -> start on the hook instead
    hook_id = promo.get("hook_id")
    if segs and isinstance(hook_id, int):
        ep0 = episodes_by_num.get(int(segs[0].get("episode", default_episode) or 0))
        try:
            sid0, eid0 = int(segs[0]["start_id"]), int(segs[0]["end_id"])
        except (KeyError, TypeError, ValueError):
            sid0 = eid0 = None
        if ep0 and sid0 is not None and sid0 < hook_id <= eid0 < len(ep0["lines"]):
            gap = ep0["lines"][hook_id]["start"] - ep0["lines"][sid0]["start"]
            if gap > HOOK_WINDOW:
                issues.append(f"hook was {round(gap, 1)}s in; moved clip start onto the hook")
                segs[0]["start_id"] = hook_id
    for seg in segs:
        ep_num = int(seg.get("episode", default_episode))
        ep = episodes_by_num.get(ep_num)
        if ep is None:
            issues.append(f"unknown episode {ep_num}")
            hard_fail = True
            continue
        try:
            sid, eid = int(seg["start_id"]), int(seg["end_id"])
        except (KeyError, TypeError, ValueError):
            issues.append("segment without valid start_id/end_id")
            hard_fail = True
            continue
        snapped = _snap_segment(ep, sid, eid, issues)
        if snapped is None:
            hard_fail = True
            continue
        start, end, lines = snapped
        if prev and (ep_num, start) < (prev[0], prev[2]):
            issues.append("segments not in story order / overlapping")
            hard_fail = True
        prev = (ep_num, start, end)
        segments_out.append({"episode": ep_num, "start": to_timestamp(start), "end": to_timestamp(end),
                             "start_sec": start, "end_sec": end, "start_id": sid, "end_id": eid})
        all_lines.extend(lines)

    if not segments_out:
        return {**promo, "segments": [], "status": "rejected", "issues": issues or ["no segments"], "duration": 0}

    duration = round(sum(s["end_sec"] - s["start_sec"] for s in segments_out), 2)
    if duration < MIN_LEN:
        issues.append(f"too short: {duration}s (min {MIN_LEN})")
        hard_fail = True
    if duration > MAX_LEN:
        issues.append(f"too long: {duration}s (max {MAX_LEN})")
        hard_fail = True

    # hook must land in the first 3 seconds
    first = segments_out[0]
    first_ep = episodes_by_num[first["episode"]]
    hook_offset = None
    if isinstance(hook_id, int) and 0 <= hook_id < len(first_ep["lines"]):
        hook_offset = round(first_ep["lines"][hook_id]["start"] - first["start_sec"], 2)
        if hook_offset < -0.01 or hook_offset > HOOK_WINDOW:
            issues.append(f"hook line starts {hook_offset}s into the clip (must be 0-{HOOK_WINDOW}s)")
            hard_fail = True
    opening = [l for l in all_lines if l["start"] - first["start_sec"] <= HOOK_WINDOW] or all_lines[:1]
    last = segments_out[-1]
    closing = [l for l in all_lines if l["end"] >= last["end_sec"] - 5] or all_lines[-1:]

    status = "rejected" if hard_fail else ("fixed" if issues else "ok")
    return {
        **promo,
        "segments": segments_out,
        "duration": duration,
        "hook_offset": hook_offset,
        "rule_hook_signal": round(max(hook_signal(l) for l in opening), 2),
        "rule_cliff_signal": round(cliffhanger_signal(closing), 2),
        "opening_text": " | ".join(l["text"] for l in opening if l["type"] != "CAMERA")[:300],
        "closing_text": " | ".join(l["text"] for l in closing if l["type"] != "CAMERA")[-300:],
        "status": status,
        "issues": issues,
    }
