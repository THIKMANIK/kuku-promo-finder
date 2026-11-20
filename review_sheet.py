"""
review_sheet.py - makes a CSV for checking the top promos by hand. Pure code, no AI.

    python review_sheet.py outputs/full_groq                # top 15 -> outputs/full_groq/review_top15.csv
    python review_sheet.py outputs/full_groq --top 20 --input data/show

One row per promo: the top N, plus every ranked cross-episode promo that is not already in the top N.
Shows the actual screenplay lines in the first and last ~5 seconds (CAMERA lines left out) and, for
cross-episode promos, the lines on both sides of each join, so you can judge whether the jumps make sense.
hook_ok / cliffhanger_ok / standalone_ok / note are left empty to fill in with Y/N.
"""
import argparse
import csv
import json
from pathlib import Path

from parser import load_show

EDGE_SEC = 5.0


def _fmt(line: dict) -> str:
    if line["type"] == "DIALOGUE":
        who = line["speaker"] + (f" ({line['emotion']})" if line["emotion"] else "")
        return f'{who}: "{line["text"]}"'
    return f"{line['type']}: {line['text']}"


def _lines_in(ep: dict, start: float, end: float) -> list:
    return [l for l in ep["lines"] if l["type"] != "CAMERA" and l["start"] < end and l["end"] > start]


def main():
    ap = argparse.ArgumentParser(description="CSV of the top promos for manual review")
    ap.add_argument("run_dir")
    ap.add_argument("--input", default="data/show", help="screenplay folder the run used")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    all_ranked = json.loads((run_dir / "promos_ranked.json").read_text(encoding="utf-8"))
    ranked = all_ranked[:args.top] + [p for p in all_ranked[args.top:] if p.get("cross_episode")]
    eps = {e["episode"]: e for e in load_show(args.input)}
    out = run_dir / f"review_top{args.top}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "type", "clip", "duration", "opening_lines", "join_lines", "closing_lines",
                    "hook_ok", "cliffhanger_ok", "standalone_ok", "note"])
        for p in ranked:
            first, last = p["segments"][0], p["segments"][-1]
            opening = _lines_in(eps[first["episode"]], first["start_sec"], first["start_sec"] + EDGE_SEC)
            closing = _lines_in(eps[last["episode"]], last["end_sec"] - EDGE_SEC, last["end_sec"])
            clip = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
            joins = []
            for n, (a, b) in enumerate(zip(p["segments"], p["segments"][1:]), 1):
                before = _lines_in(eps[a["episode"]], a["end_sec"] - EDGE_SEC, a["end_sec"])
                after = _lines_in(eps[b["episode"]], b["start_sec"], b["start_sec"] + EDGE_SEC)
                joins.append(f"JOIN {n} (Ep{a['episode']} ends {a['end']} -> Ep{b['episode']} starts {b['start']}): "
                             f"{' | '.join(map(_fmt, before))}  >>>  {' | '.join(map(_fmt, after))}")
            kind = "cross-episode" if p.get("cross_episode") else "single"
            w.writerow([p["rank"], kind, clip, p["duration"], " | ".join(map(_fmt, opening)), "\n".join(joins),
                        " | ".join(map(_fmt, closing)), "", "", "", ""])
    extra = len(ranked) - min(args.top, len(all_ranked))
    print(f"wrote {out} ({len(ranked)} promos: top {args.top} + {extra} cross-episode)")


if __name__ == "__main__":
    main()
