"""
parser.py - turns one screenplay .txt file into structured data. Pure code, no AI.

Output for one episode (a dict):
{
  "episode": 1,
  "logline": "...",
  "continuity": "...",
  "duration": 69.0,                   # seconds
  "lines": [                          # every timestamped block, in order
     {"id": 0, "type": "CAMERA", "speaker": None, "emotion": None,
      "start": 0.0, "end": 4.0, "text": "Slow wide shot of a narrow street ..."},
     {"id": 4, "type": "DIALOGUE", "speaker": "THEO", "emotion": "Smug, Cold",
      "start": 8.0, "end": 10.5, "text": "You have until Friday, Nora."},
     ...
  ]
}
"""
import re
from pathlib import Path

# "**NORA (Shocked, Whispering):**"  or  "**CAMERA:**"  or  "**VOICE ON PHONE (O.S.) (Urgent):**"
BLOCK_HEADER = re.compile(r"^\*\*(?P<label>.+?):\*\*\s*$")
# "Start: (00:11.30) | End: (00:13.10)"
TIME_LINE = re.compile(
    r"^Start:\s*\((?P<s>\d+:\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((?P<e>\d+:\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)
SCENE_END = re.compile(r"^scene end:\s*(\d+:\d+(?:\.\d+)?)", re.IGNORECASE)  # case-insensitive: ep 6 uses CAPS
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "VO", "TEXT"}


def to_seconds(ts: str) -> float:
    """'01:14.60' -> 74.6"""
    minutes, seconds = ts.split(":")
    return round(int(minutes) * 60 + float(seconds), 2)


def to_timestamp(sec: float) -> str:
    """74.6 -> '01:14.60' (same format as the screenplays)"""
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"


def _split_label(label: str):
    """'NORA (Shocked, Whispering)' -> ('DIALOGUE', 'NORA', 'Shocked, Whispering')"""
    label = label.strip()
    if label.upper() in NON_DIALOGUE:
        return label.upper(), None, None
    parens = re.findall(r"\(([^)]*)\)", label)
    speaker = re.sub(r"\s*\([^)]*\)", "", label).strip().upper()
    emotion = next((p for p in reversed(parens) if p.upper() not in {"O.S.", "V.O.", "CONT'D", "O.C."}), None)
    return "DIALOGUE", speaker, emotion


def _header_value(text: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_episode(path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    ep_raw = _header_value(text, "EPISODE NUMBER")
    ep_num = int(re.search(r"\d+", ep_raw).group()) if re.search(r"\d+", ep_raw) else None
    if ep_num is None:  # fall back to the file name: episode-007-...
        m = re.search(r"episode-(\d+)", Path(path).name, re.IGNORECASE)
        ep_num = int(m.group(1)) if m else 0

    lines, current, warnings = [], None, []
    scene_ends = []
    for raw in text.splitlines():
        line = raw.strip()
        if m := SCENE_END.match(line):
            scene_ends.append(to_seconds(m.group(1)))
        if m := BLOCK_HEADER.match(line):
            btype, speaker, emotion = _split_label(m.group("label"))
            current = {"type": btype, "speaker": speaker, "emotion": emotion,
                       "start": None, "end": None, "text": ""}
            lines.append(current)
            continue
        if current is None:
            continue
        if m := TIME_LINE.match(line):
            current["start"], current["end"] = to_seconds(m.group("s")), to_seconds(m.group("e"))
        elif line and not line.startswith("---") and not line.upper().startswith("SCENE"):
            current["text"] = (current["text"] + " " + line.strip('"')).strip()

    clean = []
    for blk in lines:
        if blk["start"] is None:
            warnings.append(f"block without timestamp skipped: {blk['type']} {blk['speaker'] or ''}")
            continue
        blk["id"] = len(clean)
        clean.append(blk)

    duration = max(scene_ends + [b["end"] for b in clean] + [0.0])
    return {
        "episode": ep_num,
        "file": Path(path).name,
        "logline": _header_value(text, "EPISODE LOGLINE"),
        "continuity": _header_value(text, "CONTINUITY NOTES"),
        "duration": duration,
        "lines": clean,
        "warnings": warnings,
    }


def load_show(folder) -> list:
    """Parse every .txt file in a folder, sorted by episode number."""
    eps = [parse_episode(p) for p in sorted(Path(folder).glob("*.txt"))]
    return sorted(eps, key=lambda e: e["episode"])


def render_for_llm(ep: dict, include_camera: bool = False) -> str:
    """Compact text the LLM reads: one line per block, with timestamps and line ids.
    CAMERA lines are left out unless include_camera=True; the other lines keep their original ids
    (so ids have gaps), and the validator still works on the full list of lines."""
    out = []
    for b in ep["lines"]:
        if b["type"] == "CAMERA" and not include_camera:
            continue
        who = b["speaker"] + (f" ({b['emotion']})" if b["emotion"] else "") if b["type"] == "DIALOGUE" else b["type"]
        out.append(f"[{b['id']}] {to_timestamp(b['start'])}-{to_timestamp(b['end'])} {who}: {b['text']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys, json
    ep = parse_episode(sys.argv[1])
    print(json.dumps({k: v for k, v in ep.items() if k != "lines"}, indent=2))
    print(f"{len(ep['lines'])} blocks")
    print(render_for_llm(ep)[:1500])
