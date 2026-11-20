"""
llm.py - the ONLY place that talks to an AI model.

Swapping the model = changing .env, no code changes:
    LLM_PROVIDER=gemini   (default)  uses GEMINI_API_KEY
    LLM_PROVIDER=groq                uses GROQ_API_KEY
    LLM_PROVIDER=openai              uses OPENAI_API_KEY (or any OpenAI-compatible API via LLM_BASE_URL)
    LLM_PROVIDER=mock                no AI at all: rule-based baseline (works offline, used for tests)
    LLM_MODEL=...                    optional, overrides the default model of the provider
    LLM_REASONING_EFFORT=low         optional (low|medium|high), only sent to Groq gpt-oss models

Gemini and Groq both offer an OpenAI-compatible endpoint, so one client covers all of them.
"""
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    #  name     base_url                                                     key env var       default model
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.8-flash"),
    "groq":   ("https://api.groq.com/openai/v1",                          "GROQ_API_KEY",   "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1",                               "OPENAI_API_KEY", "gpt-4o-mini"),
}


class DailyLimitError(RuntimeError):
    """The provider's DAILY quota (tokens or requests per day) is used up. Retrying today won't help."""


DAILY_LIMIT = re.compile(r"per day|\b(TPD|RPD)\b", re.IGNORECASE)


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        if self.provider == "mock":
            self.model = "rule-based-baseline"
            self.client = None
            return
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM_PROVIDER '{self.provider}'. Use one of {list(PROVIDERS)} or 'mock'.")
        base_url, key_var, default_model = PROVIDERS[self.provider]
        base_url = os.getenv("LLM_BASE_URL", base_url)
        api_key = os.getenv(key_var)
        if not api_key:
            raise RuntimeError(f"{key_var} is not set. Add it to your .env file.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("LLM_MODEL") or default_model
        self.min_gap = float(os.getenv("LLM_MIN_SECONDS_BETWEEN_CALLS", "4"))  # free tiers limit requests/minute
        self._last_call = 0.0
        # optional: LLM_REASONING_EFFORT=low|medium|high, only sent to models that accept it (Groq gpt-oss)
        effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
        if effort and effort not in {"low", "medium", "high"}:
            raise ValueError(f"LLM_REASONING_EFFORT must be low, medium or high, not '{effort}'")
        self.extra = {"reasoning_effort": effort} if effort and self.provider == "groq" and "gpt-oss" in self.model else {}

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def ask_json(self, system: str, user: str, retries: int = 4, temperature: float = 0.3) -> tuple[dict, str]:
        """Send a prompt, get back (parsed JSON dict, raw text). Retries on rate limits and bad JSON."""
        if self.client is None:
            raise RuntimeError("mock provider has no model; callers use the rule-based path instead")
        last_err = None
        for attempt in range(retries):
            wait = self.min_gap - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    **self.extra,
                )
                self.calls += 1
                if resp.usage:
                    self.tokens_in += resp.usage.prompt_tokens or 0
                    self.tokens_out += resp.usage.completion_tokens or 0
                raw = resp.choices[0].message.content or ""
                return parse_json(raw), raw
            except Exception as e:  # rate limit, network, invalid JSON ...
                if "429" in str(e) and DAILY_LIMIT.search(str(e)):
                    raise DailyLimitError(f"daily limit reached for {self.name}") from e
                last_err = e
                transient = any(s in str(e) for s in ("429", "503")) or "rate" in str(e).lower()
                backoff = 2 ** attempt * 5 if transient else 2  # 503 = model overloaded, wait like a rate limit
                print(f"  ! LLM call failed ({type(e).__name__}: {str(e)[:120]}), retry in {backoff}s")
                time.sleep(backoff)
        raise RuntimeError(f"LLM failed after {retries} attempts: {last_err}")


def parse_json(raw: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add text around it. Be forgiving."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise
