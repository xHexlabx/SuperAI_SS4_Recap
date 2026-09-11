"""One thin OpenAI-compatible chat client, so local Qwen and hosted Qwen are the same code.

    local  : vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ   -> http://localhost:8000/v1
    ollama : ollama serve                               -> http://localhost:11434/v1
    remote : OpenRouter / DashScope                     -> https://.../v1 + $LLM_API_KEY

Answers are cached on disk by hash of (model, messages), so re-running `make compile` after a
prompt tweak only pays for the clauses whose prompt actually changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from .utils import ensure_dir

log = logging.getLogger("legal_act.llm")

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.llm
        self.base_url = self.cfg.base_url.rstrip("/")
        self.api_key = os.environ.get(self.cfg.api_key_env, "") or "EMPTY"
        self.cache_dir = ensure_dir(cfg.resolve(self.cfg.cache_dir)) if self.cfg.cache else None
        self._client = httpx.Client(timeout=self.cfg.timeout)

    # ---------------------------------------------------------------- cache

    def _key(self, messages: Messages, schema: dict | None = None) -> str:
        blob = json.dumps(
            {"model": self.cfg.model, "t": self.cfg.temperature,
             "think": self.cfg.thinking, "extra": self.cfg.extra_body,
             "schema": schema, "messages": messages},
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> str | None:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["content"]
        return None

    def _store(self, key: str, content: str) -> None:
        if self.cache_dir is None:
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8"
        )

    # ---------------------------------------------------------------- call

    def chat(self, messages: Messages, schema: dict | None = None) -> str:
        key = self._key(messages, schema)
        hit = self._cached(key)
        if hit is not None:
            return hit

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.thinking in ("on", "off"):
            payload["chat_template_kwargs"] = {"enable_thinking": self.cfg.thinking == "on"}
        if self.cfg.extra_body:
            payload.update(self.cfg.extra_body)
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rule", "schema": schema},
            }
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if r.status_code == 429 or r.status_code >= 500:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                self._store(key, content)
                return content
            except httpx.ConnectError as exc:
                raise LLMError(
                    f"cannot reach {self.base_url} — start a server with `make serve`, or point "
                    f"llm.base_url at a hosted endpoint and export ${self.cfg.api_key_env}."
                ) from exc
            except Exception as exc:  # noqa: BLE001 — retry anything transient
                last = exc
                sleep = min(30.0, 2.0 ** attempt) + random.random()
                log.warning("llm call failed (%s/%s): %s — retrying in %.1fs",
                            attempt + 1, self.cfg.max_retries, exc, sleep)
                time.sleep(sleep)
        raise LLMError(f"giving up after {self.cfg.max_retries} attempts: {last}")

    def map(self, items: Sequence[Any], build: Callable[[Any], Messages],
            desc: str = "llm", schema: dict | None = None) -> list[str | None]:
        """Run one prompt per item, `llm.concurrency` at a time, keeping input order."""
        from tqdm import tqdm

        results: list[str | None] = [None] * len(items)

        def work(i: int) -> None:
            try:
                results[i] = self.chat(build(items[i]), schema)
            except LLMError as exc:
                log.error("%s[%d] failed: %s", desc, i, exc)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as pool:
            list(tqdm(pool.map(work, range(len(items))), total=len(items), desc=desc))
        return results

    def map_json(self, items: Sequence[Any], build: Callable[[Any], Messages],
                 schema: dict, desc: str = "llm") -> list[dict | None]:
        """Same as `map`, but parse the replies — and re-ask under the grammar if one is broken.

        Free-form decoding wins on quality here, so the schema is only paid for on the replies
        that actually came back malformed (Qwen3-14B: ~5% of clauses, all of them the same
        mistake of closing the `branches` array early and starting a second one).
        """
        replies = self.map(items, build, desc=desc)
        out: list[dict | None] = []
        retry: list[int] = []
        for i, reply in enumerate(replies):
            try:
                out.append(extract_json(reply) if reply is not None else None)
            except ValueError:
                out.append(None)
            if out[i] is None and reply is not None:
                retry.append(i)

        if retry:
            log.info("%s: %d/%d replies unparsable — re-asking under the JSON grammar",
                     desc.strip(), len(retry), len(items))
            fixed = self.map([items[i] for i in retry], build, desc=f"{desc}*", schema=schema)
            for i, reply in zip(retry, fixed):
                try:
                    out[i] = extract_json(reply) if reply is not None else None
                except ValueError as exc:
                    log.warning("%s[%d] still unparsable after grammar retry: %s", desc, i, exc)
        return out

    def ping(self) -> str:
        """Fail fast with a readable message before a long compile run."""
        try:
            r = self._client.get(f"{self.base_url}/models",
                                 headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            served = [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"cannot reach {self.base_url} ({exc}). Start Qwen locally with `make serve`, "
                f"or set llm.base_url + ${self.cfg.api_key_env} for a hosted endpoint."
            ) from exc
        if served and self.cfg.model not in served:
            log.warning("llm.model=%r is not in the served list %s — calling it anyway",
                        self.cfg.model, served[:5])
        return f"{self.cfg.model} @ {self.base_url}"


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (handles ```json fences and prose)."""
    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        # The one malformation Qwen3 actually makes: closing `branches` and opening a second
        # array instead of continuing the first. No schema here nests array-in-array, so
        # stitching `], [` back into `, ` can only ever repair, never corrupt.
        repaired = re.sub(r"\]\s*,\s*\[", ", ", str(text))
        return _extract_json(repaired)


def _extract_json(text: str) -> dict[str, Any]:
    if text is None:
        raise ValueError("empty reply")
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                s = part
                break
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError(f"unbalanced JSON in reply: {text[:200]!r}")
