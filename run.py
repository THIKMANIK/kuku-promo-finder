"""
run.py - runs the whole pipeline on a folder of screenplays (or single files).

    python run.py --input data/show                      # all episodes, model from .env
    python run.py --input data/show --episodes 1-5       # quick test
    python run.py --input data/show --provider groq      # swap the model
    python run.py --input data/show --provider mock      # no AI, rule-based baseline

Writes to outputs/<run-name>/:
    raw/                 unedited AI responses (one file per call)
    candidates.json      every candidate with its validation result (incl. rejected ones)
    promos_ranked.json   final ranked promos
    promos_ranked.csv    same, as a spreadsheet
    report.md            readable top list
    stats.json           calls, tokens, time, how often the AI broke the rules
"""
import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from finder import build_series_context, final_posting_order, find_cross_episode_promos, find_promos_in_episode
from llm import LLM, DailyLimitError
from parser import load_show, parse_episode
from ranker import rank
from validator import fit_cross_episode, validate


def parse_range(spec: str | None):
    if not spec:
        return None
    keep = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return keep


def load_inputs(paths: list) -> list:
    episodes = []
    for p in map(Path, paths):
        episodes += load_show(p) if p.is_dir() else [parse_episode(p)]
    return sorted(episodes, key=lambda e: e["episode"])


ISSUE_KINDS = ["trimmed cross-episode", "too short", "too long", "hook line starts", "moved clip start onto the hook", "chopped dialogue",
               "mid-dialogue", "out of range", "not in story order", "end_id before start_id", "unknown episode",
               "without valid"]


def issue_kind(text: str) -> str:
    return next((k for k in ISSUE_KINDS if k in text), "other")


def run(episodes: list, llm: LLM, out_dir: Path, cross: bool = True, log=print, title: str | None = None) -> dict:
    """title: the show's name; only sent in the cross-episode and final posting-order calls."""
    t0 = time.time()
    raw_dir = out_dir / "raw"
    by_num = {e["episode"]: e for e in episodes}

    done, left = [], []  # episodes with / without a promo response (for the daily-limit message)

    def stop_on_daily_limit(e, step):
        left_eps = [ep["episode"] for ep in episodes if ep["episode"] not in done]
        log(f"\nSTOPPED at {step}: {e}. Nothing more is sent today.")
        log(f"  episodes done ({len(done)}): {_ranges(done) or '-'}")
        log(f"  episodes left ({len(left_eps)}): {_ranges(left_eps) or '-'}")
        log("  Re-run the same command after the daily limit resets; saved responses are reused.")
        return left_eps

    log(f"[1/4] series context ({llm.name})")
    try:
        ctx = build_series_context(llm, episodes, raw_dir)
    except DailyLimitError as e:
        stop_on_daily_limit(e, "series context")
        return {"stopped": True}

    log(f"[2/4] finding promos in {len(episodes)} episodes")
    candidates = []
    for ep in episodes:
        try:
            found = find_promos_in_episode(llm, ep, ctx, raw_dir)
        except DailyLimitError as e:
            left = stop_on_daily_limit(e, f"episode {ep['episode']}")
            break
        except Exception as e:
            log(f"  ep {ep['episode']}: FAILED ({e})")
            continue
        done.append(ep["episode"])
        checked = [validate(p, by_num, ep["episode"]) for p in found]
        candidates += checked
        ok = sum(p["status"] != "rejected" for p in checked)
        log(f"  ep {ep['episode']:>3}: {len(found)} found, {ok} passed checks")

    log("[3/4] cross-episode promos")
    if left:  # don't build (and save) cross-episode promos from an incomplete set of episodes
        log("  skipped: not all episodes are done yet")
    elif cross and len(episodes) > 1:
        try:
            for p in find_cross_episode_promos(llm, candidates, ctx, raw_dir, title=title):
                candidates.append(validate(fit_cross_episode(p, by_num), by_num))
        except DailyLimitError as e:
            log(f"  cross-episode step STOPPED ({e}); all episodes are done, re-run tomorrow to add it")
            left = ["cross-episode"]
        except Exception as e:
            log(f"  cross-episode step FAILED ({e})")

    log("[4/4] ranking")
    ranked = rank(candidates)
    stability = None
    for p in ranked:
        p["code_rank"] = p["rank"]
    if left:  # don't build (and save) a posting order from an incomplete set of episodes
        log("  final posting order skipped: not all episodes are done yet")
    else:
        try:
            ranked, stability = final_posting_order(llm, ranked, ctx, raw_dir, title=title)
        except Exception as e:  # never lose the run: keep the code ranking
            log(f"  ! final posting order FAILED ({e}); keeping the code ranking")
    for i, p in enumerate(ranked, 1):
        p["rank"] = i

    issue_types = Counter(issue_kind(i) for p in candidates for i in p.get("issues", []))
    stats = {
        "model": llm.name,
        "title": title,
        "episodes": len(episodes),
        "llm_calls": llm.calls,
        "tokens_in": llm.tokens_in,
        "tokens_out": llm.tokens_out,
        "seconds": round(time.time() - t0, 1),
        "candidates": len(candidates),
        "status_counts": dict(Counter(p["status"] for p in candidates)),
        "issue_counts": dict(issue_types.most_common()),
        "ranked": len(ranked),
        "models": dict(Counter(p.get("model", "unknown") for p in candidates)),
        "posting_stability": stability,
        "complete": not left,
        "left": left,
    }
    write_outputs(out_dir, ctx, candidates, ranked, stats)
    log(f"{'done' if not left else 'PARTIAL (see above)'}: {len(ranked)} ranked promos -> {out_dir}")
    return {"context": ctx, "candidates": candidates, "ranked": ranked, "stats": stats}


def _ranges(nums):
    """[1,2,3,7,8] -> '1-3, 7-8'"""
    out, nums = [], sorted(nums)
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def _seg_str(p):
    return " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])


def write_outputs(out_dir: Path, ctx, candidates, ranked, stats):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "series_context.json").write_text(json.dumps(ctx, indent=2), encoding="utf-8")
    (out_dir / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    (out_dir / "promos_ranked.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    cols = ["rank", "code_rank", "final_score", "posting_reason", "model", "segments", "duration", "hook", "cliffhanger", "why_it_works",
            "hook_score", "cliffhanger_score", "standalone_score", "spoiler_risk", "status", "issues", "rank_reason"]
    with open(out_dir / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in ranked:
            w.writerow([_seg_str(p) if c == "segments" else "; ".join(p.get(c, [])) if c == "issues" else p.get(c, "")
                        for c in cols])
    md = [f"# Promo report\n", f"Model: `{stats['model']}` | episodes: {stats['episodes']} | "
          f"candidates: {stats['candidates']} | ranked: {stats['ranked']} | status: {stats['status_counts']}\n"]
    for p in ranked[:20]:
        md.append(f"## #{p['rank']} - {p.get('title', '')} (score {p['final_score']}, code rank #{p['code_rank']})\n"
                  f"- **Clip:** {_seg_str(p)} ({p['duration']}s)\n- **Hook:** {p.get('hook')}\n"
                  f"- **Cliffhanger:** {p.get('cliffhanger')}\n- **Why it works:** {p.get('why_it_works')}\n"
                  f"- **Why this rank:** {p['rank_reason']}\n"
                  + (f"- **Posting order:** {p['posting_reason']}\n" if p.get("posting_reason") else "")
                  + (f"- **Auto-fixes:** {'; '.join(p['issues'])}\n" if p.get("issues") else ""))
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Find and rank promo clips in microdrama screenplays")
    ap.add_argument("--input", nargs="+", required=True, help="folder(s) or .txt file(s)")
    ap.add_argument("--episodes", help="e.g. 1-5 or 1,3,41-42")
    ap.add_argument("--provider", help="gemini | groq | openai | mock (default: LLM_PROVIDER in .env)")
    ap.add_argument("--model", help="override the model name")
    ap.add_argument("--out", help="output folder (default outputs/<provider>_<time>)")
    ap.add_argument("--no-cross", action="store_true", help="skip cross-episode promos")
    ap.add_argument("--title", help="show title, used in the cross-episode and posting-order calls "
                                    "(default: SHOW_TITLE in .env, else the name of the input folder)")
    args = ap.parse_args()

    episodes = load_inputs(args.input)
    keep = parse_range(args.episodes)
    if keep:
        episodes = [e for e in episodes if e["episode"] in keep]
    if not episodes:
        raise SystemExit("no episodes found")
    llm = LLM(args.provider, args.model)
    out = Path(args.out or f"outputs/{llm.provider}_{datetime.now():%Y%m%d_%H%M}")
    first = Path(args.input[0])
    title = args.title or os.getenv("SHOW_TITLE") or (first if first.is_dir() else first.parent).resolve().name
    print(f"show title: {title}")
    run(episodes, llm, out, cross=not args.no_cross, title=title)


if __name__ == "__main__":
    main()
