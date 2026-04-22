"""OpenAI-compatible LLM client based on requests.

Used as a fallback when SDK-based clients are blocked by relay providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import json

import requests


@dataclass
class LLMTextResponse:
    """Simple response object with a content field for compatibility."""

    content: str


class OpenAICompatRequestsLLM:
    """Lightweight chat-completions client with invoke/stream methods."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _normalize_messages(self, prompt: Any) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        if isinstance(prompt, list):
            normalized: List[Dict[str, str]] = []
            for item in prompt:
                if isinstance(item, dict) and "role" in item and "content" in item:
                    normalized.append(
                        {"role": str(item["role"]), "content": str(item["content"])}
                    )
                elif hasattr(item, "type") and hasattr(item, "content"):
                    role = "assistant" if str(getattr(item, "type", "")).lower() == "ai" else "user"
                    normalized.append(
                        {"role": role, "content": str(getattr(item, "content", ""))}
                    )
                else:
                    normalized.append({"role": "user", "content": str(item)})
            if normalized:
                return normalized

        return [{"role": "user", "content": str(prompt)}]

    def invoke(self, prompt: Any) -> LLMTextResponse:
        payload = {
            "model": self.model,
            "messages": self._normalize_messages(prompt),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(
            self._endpoint(),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM request failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        content = ""
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")

        return LLMTextResponse(content=content)

    def stream(self, prompt: Any) -> Iterable[str]:
        payload = {
            "model": self.model,
            "messages": self._normalize_messages(prompt),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        response = requests.post(
            self._endpoint(),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM stream request failed ({response.status_code}): {response.text}"
            )

        # Some relay providers omit charset in streaming responses.
        # Decode bytes explicitly as UTF-8 to avoid mojibake for Chinese text.
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line).strip()
            if not line.startswith("data: "):
                continue

            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                break

            try:
                chunk = json.loads(payload_str)
            except Exception:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield str(piece)

            # Some providers may return full message content in stream chunks.
            message = choices[0].get("message") or {}
            full_piece = message.get("content")
            if full_piece and not piece:
                yield str(full_piece)
