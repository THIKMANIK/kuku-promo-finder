# Microdrama Promo Finder

Reads the screenplays of a vertical microdrama series and finds 20–60 second promo clips for TikTok/Reels/Shorts.
Every clip opens on a hook in its first 3 seconds, ends on a cliffhanger, and makes sense to a stranger.
The output is a ranked posting order with exact timestamps, plus a sheet for an editor to check the top picks.

| Who | Does what | Why |
|---|---|---|
| **AI** | Summarises the series (premise, characters, big reveals); proposes 2–4 promo candidates per episode as line-id ranges with hook / cliffhanger / standalone scores; joins moments across episodes; orders the top 20 for posting (3 shuffled runs, averaged) | Judging drama, hooks and cliffhangers needs language understanding |
| **Code** | Parses screenplays into numbered, timestamped lines; turns line ids into real timestamps (the AI can't invent them); enforces 20–60s, hook in the first 3s, no cut mid-dialogue, story order; auto-fixes small mistakes; scores with visible weights; removes duplicates; max 2 promos per episode | Exact, free, testable, and the rules are checked instead of trusted |
| **Editor** | Reviews the top 15 (plus cross-episode montages) in `review_top15.csv` and makes the final call | Reviewing 15 clips instead of watching every episode |

## Requirements

- Python 3.12 (tested with 3.12.3)
- Tested on Ubuntu on WSL2 (Windows). Plain Linux and macOS should work the same.
- For real runs: a free API key from [Groq](https://console.groq.com/keys) (default) or [Google AI Studio](https://aistudio.google.com/apikey) (Gemini), or any OpenAI-compatible API. Not needed for the offline `mock` mode.

## Quick start

```bash
git clone https://github.com/THIKMANIK/kuku-promo-finder.git
cd kuku-promo-finder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# open .env and paste your key after GROQ_API_KEY=   (or set LLM_PROVIDER=gemini and fill GEMINI_API_KEY=)
```

### Try it without any API key

```bash
python run.py --input examples/sample_show --provider mock
```

`mock` is a rule-based baseline (keywords and emotion cues, no AI). It runs offline in about a second and writes
the same output files as a real run, here to `outputs/mock_<date>_<time>/`. `examples/sample_show/` holds two short synthetic episodes.

### Run on a real show

```bash
python run.py --input path/to/show_folder --title "Show Title"
```

`--input` takes a folder of episode `.txt` files (or individual files) in the screenplay format of
`examples/sample_show/`. Other options:

| Option | Example | Meaning |
|---|---|---|
| `--episodes` | `--episodes 1-3` or `1,3,41-42` | only these episodes (quick tests) |
| `--provider` | `--provider gemini` | `groq` · `gemini` · `openai` · `mock` (default: `LLM_PROVIDER` in `.env`) |
| `--model` | `--model openai/gpt-oss-20b` | override the provider's default model |
| `--out` | `--out outputs/my_run` | output folder (default `outputs/<provider>_<date>_<time>`); re-using a folder resumes the run |
| `--no-cross` | | skip cross-episode montages |
| `--title` | `--title "Show Title"` | show title, used only when comparing promos across the show (default: `SHOW_TITLE` in `.env`, else the input folder name) |

A 50-episode show takes about 52 AI calls (≈4.4K tokens each) and about 40 minutes on Groq's free tier.

## Swap the model

`llm.py` is the only file that talks to a model. Everything is set in `.env`, with no code changes:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` · `gemini` · `openai` · `mock` |
| `LLM_MODEL` | `openai/gpt-oss-120b` (groq), `gemini-3.8-flash` (gemini), `gpt-4o-mini` (openai) | model name |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` | | key for the chosen provider |
| `LLM_BASE_URL` | the provider's URL | any OpenAI-compatible endpoint |
| `LLM_MIN_SECONDS_BETWEEN_CALLS` | `4` | pause between calls; use `45` on Groq's free tier (8K tokens/minute) |
| `LLM_REASONING_EFFORT` | unset | `low` / `medium` / `high`, only sent to Groq `gpt-oss` models |
| `LLM_INCLUDE_CAMERA` | `false` | `true` also sends CAMERA lines to the AI (more tokens) |
| `SHOW_TITLE` | unset | default for `--title` |

Example: any OpenAI-compatible endpoint (a local Ollama server here):

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
OPENAI_API_KEY=anything
```

## Web app

```bash
streamlit run app.py
```

Upload episode `.txt` files, pick the provider/model and show title in the sidebar, and get the ranked promos with the screenplay lines of each clip.

## Outputs (`outputs/<run>/`)

| File | Contents |
|---|---|
| `report.md` | readable top 20: clip timestamps, hook, cliffhanger, why it works, score, code rank, posting reason, auto-fixes |
| `promos_ranked.json` / `.csv` | every ranked promo with all scores, `code_rank`, `posting_reason` and the `model` that found it |
| `candidates.json` | every candidate the AI proposed, with its check result (`ok` / `fixed` / `rejected`) and the issues found |
| `stats.json` | model, AI calls, tokens, time, status and issue counts, posting-order stability (`top5_agreement`), whether the run is complete |
| `series_context.json` | the AI's series summary: premise, characters and aliases, big reveals |
| `raw/` | every AI response, unedited (`ep_001.json`, `cross_episode.json`, `final_rank_1..3.json`, ...), each with a `.meta.json` naming the model that produced it. Saved responses are reused when you re-run into the same folder |

## Quality tools

```bash
python eval_gold.py outputs/<run>                  # compare with hand-picked promos in gold.json (time overlap, IoU >= 0.5 = found)
python review_sheet.py outputs/<run>               # review_top15.csv: top 15 + cross-episode montages, with opening/closing
                                                   # lines and the lines around each join; empty Y/N columns for the editor
pip install -r requirements-dev.txt && pytest      # fast offline tests on examples/sample_show (no API calls)
```

`gold.json` holds hand-picked promos as `{"episode", "start", "end", "note"}`. The included one refers to the
confidential show, so `eval_gold.py` needs your own gold file (2nd argument) for other shows.

## Troubleshooting

| Problem | What to do |
|---|---|
| **429 / rate limit** | Per-minute limits are retried automatically with growing waits. If it keeps happening, raise `LLM_MIN_SECONDS_BETWEEN_CALLS`. |
| **Daily limit reached** (`STOPPED ... daily limit`) | The run stops cleanly and prints which episodes are done and which are left. Re-run **the same command** after the limit resets: saved responses are reused and only the missing episodes are sent. |
| **404 / model does not exist** | The model name was retired. List your models (e.g. in the provider's console) and set `LLM_MODEL` in `.env`. |
| **503 / overloaded** | Retried automatically. If a single episode still fails, re-run the same command: only that episode is sent again. |
| **Slow** | Free tiers are slow and limited: Groq ≈ 45s between calls by design (tokens/minute), Gemini free calls can take minutes each. Test with `--episodes 1-3` first. |

## Confidential data

The Kuku input pack (the screenplays of the assignment's show) is confidential and **not included**: `data/`,
`outputs/` and `*.zip` are git-ignored. `examples/sample_show/` is synthetic data written for this repo.

## Project structure

```
run.py              command line: runs the whole pipeline, writes outputs/<run>/
parser.py           screenplay .txt -> numbered, timestamped lines (no AI)
finder.py           the AI steps: series context, per-episode promos, cross-episode montages, posting order; resume logic
validator.py        checks and auto-fixes every promo: 20-60s, hook in 3s, no chopped dialogue, trims montages (no AI)
ranker.py           score with visible weights, remove duplicates, max 2 per episode (no AI)
llm.py              the only file that talks to a model; providers, retries, daily-limit detection
app.py              Streamlit web page
eval_gold.py        compare a run with hand-picked promos
review_sheet.py     CSV of the top promos for manual review
gold.json           hand-picked promos (timestamps only)
prompts/            the prompts, as plain text
examples/sample_show/  two synthetic episodes for tests and the mock demo
tests/              pytest tests (offline)
.env.example        copy to .env and fill in
requirements.txt    pinned runtime dependencies (requirements-dev.txt adds pytest)
```
