"""OpenAI-compatible LLM client (the platform passes OPENAI_API_KEY / OPENAI_MODEL /
OPENAI_BASE_URL). Every failure returns None: the agent must work without the LLM.

Budget knobs (env): AGENT_LLM_MAX_CALLS (default 30), AGENT_LLM_TIMEOUT_SEC (default 180),
AGENT_LLM_MAX_TOKENS (default 8000; doubled once if reasoning eats it), AGENT_LLM_MAX_SECONDS (default 1200, total LLM wait per task),
AGENT_LLM_RETRIES (default 2; the openai client retries
429/5xx with backoff and honours Retry-After), AGENT_LLM_MIN_INTERVAL_SEC (default 0; pause between
calls, e.g. 20 for Groq's free 8k tokens/minute).
"""
import json
import os
import re
import time


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def extract_json(text: str):
    """Parse the first JSON object/array in a model reply (handles <think> and ``` fences)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    for chunk in fenced + [text]:
        chunk = chunk.strip()
        try:
            return json.loads(chunk)
        except ValueError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = chunk.find(opener)
            end = chunk.rfind(closer)
            if 0 <= start < end:
                try:
                    return json.loads(chunk[start:end + 1])
                except ValueError:
                    continue
    return None


MAX_CONSECUTIVE_FAILURES = 4
LLM_DEFAULT_MAX_TOKENS = 8000  # Qwen3 reasons before answering: 2000 left the analyst with an empty reply
DEFAULT_MODEL = "qwen/qwen3.8-27b"  # organizers' model; their proxy substitutes the task model anyway


class LLM:
    """Thin wrapper with call/token accounting. `available` is False without credentials,
    after max_calls, or after MAX_CONSECUTIVE_FAILURES failed calls in a row.
    Every call (final prompt, reply, parse result) goes to `trace_path` as JSONL if set."""

    def __init__(self, log=print, trace_path=None):
        self.log = log
        self.trace_path = trace_path
        self.model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.max_calls = _env_int("AGENT_LLM_MAX_CALLS", 30)
        self.timeout = _env_int("AGENT_LLM_TIMEOUT_SEC", 180)  # one reasoning reply of 8-16k tokens
        self.max_tokens = _env_int("AGENT_LLM_MAX_TOKENS", LLM_DEFAULT_MAX_TOKENS)
        self.max_seconds = _env_int("AGENT_LLM_MAX_SECONDS", 1200)  # total wall time spent waiting for the LLM
        self.min_interval = _env_int("AGENT_LLM_MIN_INTERVAL_SEC", 0)
        self._last_call_end = 0.0
        self.calls = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self._client = None
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self._client = OpenAI(timeout=self.timeout, max_retries=_env_int("AGENT_LLM_RETRIES", 2))
            except Exception as error:
                log(f"LLM client unavailable: {error!r}")

    @property
    def available(self) -> bool:
        return (self._client is not None and self.calls < self.max_calls
                and self.consecutive_failures < MAX_CONSECUTIVE_FAILURES and self.seconds < self.max_seconds)

    def _trace(self, record: dict) -> None:
        if self.trace_path is None:
            return
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as error:
            self.log(f"LLM trace not written: {error!r}")
            self.trace_path = None

    def _complete(self, system: str, user: str) -> str | None:
        if "qwen" in self.model.lower() and os.environ.get("AGENT_LLM_NO_THINK", "1") == "1":
            user += "\n/no_think"
        self._last_prompt = user
        max_tokens = self.max_tokens
        while True:
            response = self._client.chat.completions.create(
                model=self.model, temperature=0.4, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.prompt_tokens += usage.prompt_tokens or 0
                self.completion_tokens += usage.completion_tokens or 0
            choice = response.choices[0]
            content = choice.message.content
            # Reasoning models (Qwen3 behind the organizers' proxy) may spend the whole limit on hidden
            # reasoning and return no content at all: retry once with twice the limit.
            if content or getattr(choice, "finish_reason", None) != "length" or max_tokens > self.max_tokens:
                return content
            self.log(f"LLM reply cut at {max_tokens} tokens before any content; retrying with {2 * max_tokens}")
            max_tokens *= 2

    def ask_json(self, role: str, system: str, user: str):
        """One call; parsed JSON or None. Logged with role, time and token totals."""
        return self._ask(role, system, user, extract_json)

    def ask_text(self, role: str, system: str, user: str) -> str | None:
        """One call; the raw reply (e.g. a code block) or None on failure / empty reply."""
        return self._ask(role, system, user, lambda text: text if text and text.strip() else None)

    def _ask(self, role: str, system: str, user: str, parse):
        if not self.available:
            return None
        wait = self._last_call_end + self.min_interval - time.time()
        if wait > 0:  # stay under a tokens-per-minute limit
            time.sleep(wait)
        self.calls += 1
        started = time.time()
        self._last_prompt = user
        try:
            text = self._complete(system, user)
        except Exception as error:
            text, parsed = None, None
            self.log(f"LLM {role}: call failed {error!r}"[:300])
            self._trace({"role": role, "prompt": self._last_prompt[:6000], "error": repr(error)[:500]})
        else:
            parsed = parse(text or "")
            self.log(f"LLM {role}: {time.time() - started:.1f}s, reply={'ok' if parsed is not None else 'BAD'}, "
                     f"calls={self.calls}, tokens={self.prompt_tokens}+{self.completion_tokens}")
            self._trace({"role": role, "seconds": round(time.time() - started, 1), "parsed_ok": parsed is not None,
                         "prompt": self._last_prompt[:6000], "reply": (text or "")[:6000]})
        finally:
            self.seconds += time.time() - started
            self._last_call_end = time.time()
        if parsed is None:
            self.failures += 1
            self.consecutive_failures += 1
            if self.consecutive_failures == MAX_CONSECUTIVE_FAILURES:
                self.log(f"LLM switched off after {MAX_CONSECUTIVE_FAILURES} failures in a row; rules only")
        else:
            self.consecutive_failures = 0
        if self.calls == self.max_calls:
            self.log(f"LLM call budget ({self.max_calls}) used up; rules only from now on")
        if self.seconds >= self.max_seconds:
            self.log(f"LLM time budget ({self.max_seconds}s) used up; rules only from now on")
        return parsed

    def stats(self) -> dict:
        return {"calls": self.calls, "failures": self.failures, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 1)}
