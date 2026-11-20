"""Fast tests on the synthetic sample show. No API calls (the only model used is the offline 'mock')."""
import json
import re
from pathlib import Path

import pytest

from llm import LLM
from parser import load_show, parse_episode, render_for_llm
from run import run
from validator import validate

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_show"


@pytest.fixture(scope="module")
def eps():
    return {e["episode"]: e for e in load_show(SAMPLE)}


def line_id(ep, text):
    """id of the first line whose text contains `text`."""
    return next(l["id"] for l in ep["lines"] if text in l["text"])


def promo(*segments, hook_id=None):
    segs = [{"episode": e, "start_id": a, "end_id": b} for e, a, b in segments]
    return {"title": "test", "segments": segs, "hook_id": segs[0]["start_id"] if hook_id is None else hook_id}


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("path", sorted(SAMPLE.glob("*.txt")), ids=lambda p: p.name)
def test_parser_reads_every_timestamped_block(path):
    ep = parse_episode(path)
    expected = len(re.findall(r"^Start: \(", path.read_text(encoding="utf-8"), re.MULTILINE))
    assert len(ep["lines"]) == expected
    assert [l["id"] for l in ep["lines"]] == list(range(expected))
    assert all(l["start"] is not None and l["end"] >= l["start"] for l in ep["lines"])
    assert ep["warnings"] == []
    assert ep["logline"]


def test_parser_header_speakers_and_emotions(eps):
    ep = eps[1]
    assert ep["duration"] == 69.0
    line = ep["lines"][line_id(ep, "You have until Friday")]
    assert (line["type"], line["speaker"], line["emotion"]) == ("DIALOGUE", "THEO", "Smug, Cold")
    off_screen = ep["lines"][line_id(ep, "who she really is")]
    assert off_screen["speaker"] == "VOICE ON PHONE" and off_screen["emotion"] == "Urgent"  # (O.S.) is not an emotion


def test_render_drops_camera_lines_but_keeps_ids(eps):
    ep = eps[1]
    text = render_for_llm(ep)
    assert "CAMERA" not in text
    ids = [int(m) for m in re.findall(r"^\[(\d+)\]", text, re.MULTILINE)]
    assert ids == [l["id"] for l in ep["lines"] if l["type"] != "CAMERA"]  # original ids, with gaps
    assert "CAMERA" in render_for_llm(ep, include_camera=True)


# ---------------------------------------------------------------- validator
def test_valid_promo_passes(eps):
    ep = eps[1]
    p = validate(promo((1, line_id(ep, "You have until Friday"), line_id(ep, "all I have left of him"))), eps)
    assert p["status"] == "ok", p["issues"]
    assert p["duration"] == pytest.approx(23.0)
    assert p["segments"][0]["start"] == "00:08.00"


def test_rejects_too_long(eps):
    p = validate(promo((1, 0, len(eps[1]["lines"]) - 1)), eps)
    assert p["status"] == "rejected"
    assert any("too long" in i for i in p["issues"])


def test_rejects_too_short(eps):
    ep = eps[1]
    p = validate(promo((1, line_id(ep, "You have until Friday"), line_id(ep, "Until Friday for what"))), eps)
    assert p["status"] == "rejected"
    assert any("too short" in i for i in p["issues"])


def test_rejects_bad_line_id(eps):
    p = validate(promo((1, 4, 999)), eps)
    assert p["status"] == "rejected"
    assert any("out of range" in i for i in p["issues"])


def test_rejects_segments_out_of_story_order(eps):
    p = validate(promo((2, line_id(eps[2], "You knew my father"), line_id(eps[2], "second will")),
                       (1, line_id(eps[1], "You have until Friday"), line_id(eps[1], "Promises are not paperwork"))), eps)
    assert p["status"] == "rejected"
    assert any("not in story order" in i for i in p["issues"])


def test_moves_clip_start_onto_the_hook(eps):
    ep = eps[1]
    hook = line_id(ep, "You have until Friday")  # starts 8s into the episode: too late for a clip that starts at 0
    p = validate(promo((1, 0, line_id(ep, "better daughter")), hook_id=hook), eps)
    assert p["status"] == "fixed"
    assert any("moved clip start onto the hook" in i for i in p["issues"])
    assert p["segments"][0]["start_sec"] == 8.0
    assert p["hook_offset"] == 0


def test_never_chops_dialogue(eps):
    ep = eps[1]
    # the clip ends on an ACTION line (31.0-34.0) while Theo's next line (33.2-36.5) is already being spoken
    end_on_action = line_id(ep, "turns toward the door")
    p = validate(promo((1, line_id(ep, "You have until Friday"), end_on_action)), eps)
    assert any("chopped dialogue" in i for i in p["issues"])
    assert p["segments"][0]["end_sec"] == 36.5  # extended to the end of the dialogue line
    assert p["status"] == "fixed"


# ---------------------------------------------------------------- whole pipeline, offline
def test_mock_run_writes_all_outputs(eps, tmp_path):
    result = run(list(eps.values()), LLM("mock"), tmp_path, log=lambda m: None, title="The Flower Shop")
    for name in ["candidates.json", "promos_ranked.json", "promos_ranked.csv", "report.md", "stats.json"]:
        assert (tmp_path / name).exists(), name
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["llm_calls"] == 0 and stats["complete"] is True
    assert result["ranked"] and all(p["status"] != "rejected" for p in result["ranked"])
