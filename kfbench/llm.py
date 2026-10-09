"""Minimal OpenAI-compatible chat client (text + image parts), synchronous `requests` wrapped for asyncio.

Every text step of the pipeline goes through `LLM.complete`; `read_media` goes through `LLM.complete` on the
[vision] endpoint with an `image_url` content part.  Temperature 0 everywhere, as in the paper.
"""
from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import requests

from .config import Endpoint


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, ep: Endpoint):
        self.ep = ep
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.ep.api_key:
            h["Authorization"] = f"Bearer {self.ep.api_key}"
        return h

    def complete_sync(self, messages: list[dict[str, Any]], *, temperature: float = 0.0,
                      max_tokens: int = 2048, retries: int = 2) -> str:
        body: dict[str, Any] = {"model": self.ep.model, "messages": messages,
                                "temperature": temperature, "max_tokens": max_tokens, **(self.ep.extra_body or {})}
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(f"{self.ep.base_url}/chat/completions", json=body, headers=self._headers(),
                                  timeout=self.ep.timeout_s)
                if r.status_code >= 500 or r.status_code == 429:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                j = r.json()
                self.calls += 1
                u = j.get("usage") or {}
                self.prompt_tokens += int(u.get("prompt_tokens") or 0)
                self.completion_tokens += int(u.get("completion_tokens") or 0)
                msg = (j.get("choices") or [{}])[0].get("message") or {}
                return str(msg.get("content") or "")
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM call failed after {retries + 1} attempts: {last}")

    async def complete(self, messages: list[dict[str, Any]], **kw) -> str:
        return await asyncio.to_thread(self.complete_sync, messages, **kw)

    @staticmethod
    def image_message(prompt: str, data: bytes, mime: str) -> list[dict[str, Any]]:
        b64 = base64.b64encode(data).decode("ascii")
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}]
