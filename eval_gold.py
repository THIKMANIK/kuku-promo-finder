"""
eval_gold.py - compares the tool's promos with promos a human picked by hand (gold.json).

    python eval_gold.py outputs/<run-name>

gold.json format:
[
  {"episode": 1, "start": "00:33.70", "end": "01:18.50", "note": "altar betrayal"}
]
For each gold pick we report the best-matching tool promo and the time overlap (IoU: 1.0 = identical,
0 = no overlap). A gold pick counts as "found" if IoU >= 0.5.
"""
import json
import sys
from pathlib import Path

from parser import to_seconds


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def main():
    run_dir = Path(sys.argv[1])
    gold_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("gold.json")
    gold = json.loads(gold_path.read_text())
    candidates = json.loads((run_dir / "candidates.json").read_text())
    ranked = json.loads((run_dir / "promos_ranked.json").read_text())
    rank_of = {(p["segments"][0]["episode"], p["segments"][0]["start"]): p["rank"] for p in ranked if p["segments"]}
    found = 0
    for g in gold:
        gs = (to_seconds(g["start"]), to_seconds(g["end"]))
        best, best_iou = None, 0.0
        for p in candidates:
            for s in p["segments"]:
                if s["episode"] == g["episode"]:
                    v = iou(gs, (s["start_sec"], s["end_sec"]))
                    if v > best_iou:
                        best, best_iou = p, v
        hit = best_iou >= 0.5
        found += hit
        where = ""
        if best:
            seg = best["segments"][0]
            where = (f"tool: Ep{seg['episode']} {seg['start']}-{seg['end']} status={best['status']} "
                     f"rank={rank_of.get((seg['episode'], seg['start']), '-')}")
        print(f"{'FOUND' if hit else 'MISSED':6s} Ep{g['episode']} {g['start']}-{g['end']} ({g.get('note', '')}) "
              f"IoU={best_iou:.2f}  {where}")
    print(f"\n{found}/{len(gold)} gold promos found (IoU >= 0.5)")


if __name__ == "__main__":
    main()
