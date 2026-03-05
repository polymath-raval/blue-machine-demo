from __future__ import annotations

import re

from ollama import AsyncClient

_client: AsyncClient | None = None
_history: list[dict] = []
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and conversational — "
    "they will be spoken aloud. Avoid markdown, bullet points, or special characters. "
    "Answer in plain natural sentences."
)


def get_client() -> AsyncClient:
    global _client
    if _client is None:
        _client = AsyncClient()
    return _client


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def chat(user_message: str) -> str:
    client = get_client()
    _history.append({"role": "user", "content": user_message})

    # Keep bounded conversation history (pairs of user+assistant)
    recent = _history[-(MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

    response = await client.chat(
        model="qwen3.5:2b",
        messages=messages,
        think=False,  # disable chain-of-thought for faster voice responses
    )

    content = _strip_thinking(response.message.content)
    _history.append({"role": "assistant", "content": content})
    return content


def clear_history() -> None:
    _history.clear()
