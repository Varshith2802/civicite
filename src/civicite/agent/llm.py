"""Minimal OpenAI-compatible chat client with tool calling (works with OpenAI, Ollama, vLLM, LM Studio ...)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.1, timeout: float = 120.0, max_tokens: int = 700):
        self.base_url = (base_url or os.environ.get("CIVICITE_LLM_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        self.model = model or os.environ.get("CIVICITE_LLM_MODEL") or "llama3.1:8b"
        self.api_key = api_key if api_key is not None else os.environ.get("CIVICITE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"{self.model} @ {self.base_url}"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature,
                                   "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"LLM unreachable at {self.base_url}: {e}") from e
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response: {str(data)[:300]}") from e


class ScriptedLLM:
    """Test double: a function decides the next assistant message from the conversation so far."""

    def __init__(self, script: Callable[[list[dict[str, Any]]], dict[str, Any]], name: str = "scripted"):
        self.script = script
        self.name = name
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> dict[str, Any]:
        self.calls += 1
        return self.script(messages)
