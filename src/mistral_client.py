"""Mistral API client helper used by the bot.

This module provides a small, project-focused wrapper around an HTTP
Mistral-like inference API. It is intentionally conservative: the
endpoint URL and API key are configurable via constructor args or
environment variables (`MISTRAL_API_URL`, `MISTRAL_API_KEY`).

The `MistralClient.ask` method accepts a user prompt and returns a
string formatted as Markdown suitable for sending to Telegram
(standard Markdown, not MarkdownV2). The implementation attempts to
handle a few common JSON response shapes so it's resilient to API
variations.

Example:
    client = MistralClient()
    md = client.ask("Explain mindfulness in simple terms")
    # send `md` to Telegram using bot API with parse_mode='Markdown'
"""
from __future__ import annotations

import os
import json
from typing import Optional

import requests


class MistralClient:
    """Lightweight client for calling a Mistral-style text generation API.

    Args:
        api_key: API key/token for authorization. Falls back to
            `MISTRAL_API_KEY` env var.
        api_url: Base URL for the API. Falls back to `MISTRAL_API_URL`
            env var or `https://api.mistral.ai`.
        timeout: Request timeout in seconds.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.api_url = (api_url or os.getenv("MISTRAL_API_URL") or "https://api.mistral.ai").rstrip("/")
        self.timeout = timeout

    def ask(self, prompt: str, system_promt: str, model: str = "mistral-large-latest", max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Send `prompt` to the Mistral API and return Markdown text for Telegram.

        The returned string is plain Markdown (suitable for Telegram's
        `parse_mode='Markdown'`). The method performs minimal post-processing
        and will raise a `RuntimeError` on API or network errors.

        The request payload is intentionally generic; adapt keys if your
        deployment expects a different schema.
        """
        if not prompt:
            raise ValueError("prompt must be a non-empty string")

        if not self.api_key:
            raise RuntimeError("Mistral API key is not configured (MISTRAL_API_KEY)")

        url = f"{self.api_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            # Common fields used by several hosted inference APIs - change
            # these if your provider expects different names.
            "model": model,
            "messages": [{ "role": "user", "content": prompt }, {"role": "system", "content": system_promt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"network error when calling Mistral API: {exc}") from exc

        if resp.status_code != 200:
            # try to include any helpful message from the API body
            msg = resp.text
            try:
                j = resp.json()
                # common error shapes
                msg = j.get("error") or j.get("message") or json.dumps(j, ensure_ascii=False)
            except Exception:
                pass
            raise RuntimeError(f"Mistral API returned {resp.status_code}: {msg}")

        try:
            data = resp.json()
        except ValueError:
            # response was not JSON, return raw text as markdown-safe content
            return self._to_markdown(resp.text)

        text = self._extract_text_from_response(data)
        return self._to_markdown(text)

    def _extract_text_from_response(self, data: dict) -> str:
        """Try several common response shapes and return the generated text."""
        # 1) Some APIs return {'output': '...'}
        if isinstance(data, dict):
            if "output" in data and isinstance(data["output"], str):
                return data["output"]
            # 2) OpenAI-like: {'choices': [{ 'text': '...' }]} or {'choices':[{'message':{'content':'...'}}]}
            choices = data.get("choices")
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    if "text" in first and isinstance(first["text"], str):
                        return first["text"]
                    if "message" in first and isinstance(first["message"], dict):
                        for key in ("content", "text"):
                            if key in first["message"] and isinstance(first["message"][key], str):
                                return first["message"][key]
            # 3) Some hosted APIs use {'generated_text': '...'}
            if "generated_text" in data and isinstance(data["generated_text"], str):
                return data["generated_text"]
            # 4) Some use {'results':[{'content':'...'}]}
            results = data.get("results")
            if results and isinstance(results, list):
                r0 = results[0]
                if isinstance(r0, dict):
                    for key in ("content", "text", "output"):
                        if key in r0 and isinstance(r0[key], str):
                            return r0[key]

        # As a last resort try to stringify the whole payload in readable form
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

    def _to_markdown(self, text: str) -> str:
        """Minimal conversion to Markdown suitable for Telegram messages.

        Notes:
        - We keep content as Markdown (not MarkdownV2). The caller should
          pass `parse_mode='Markdown'` when sending via Telegram.
        - Avoid aggressive escaping here — let the bot sender choose the
          exact parse mode.
        """
        if text is None:
            return ""

        # Trim excessive whitespace
        out = text.strip()

        # Ensure code blocks are fenced: if the content contains multiple lines
        # and includes typical code markers, leave as-is; otherwise return plain
        # markdown text. Do not force any wrapping to preserve generated formatting.
        return out


__all__ = ["MistralClient"]
