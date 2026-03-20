from __future__ import annotations

import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Shared ────────────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and conversational — "
    "they will be spoken aloud. Avoid markdown, bullet points, or special characters. "
    "Answer in plain natural sentences."
)

_search_tool = DuckDuckGoSearchRun()

# ── Ollama (via OpenAI-compatible /v1 endpoint) ───────────────────────────────

OLLAMA_MODEL = "llama3.1:8b"

_ollama_history: list = []
_ollama_llm: ChatOpenAI | None = None


def _get_ollama_llm() -> ChatOpenAI:
    global _ollama_llm
    if _ollama_llm is None:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        _ollama_llm = ChatOpenAI(
            base_url=f"{ollama_host}/v1",
            api_key="ollama",          # Ollama ignores this but the field is required
            model=OLLAMA_MODEL,
        ).bind_tools([_search_tool])
    return _ollama_llm


async def chat_ollama(user_message: str) -> str:
    """Chat with local Ollama using LangChain tool calling (DuckDuckGo search)."""
    llm = _get_ollama_llm()

    _ollama_history.append(HumanMessage(content=user_message))
    recent = _ollama_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent

    while True:
        response: AIMessage = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            query = tc["args"].get("query", "")
            print(f"[Ollama Tool Call] Searching: {query}")
            result = _search_tool.run(query)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    content = response.content
    _ollama_history.append(AIMessage(content=content))
    return content


# ── OpenRouter + DuckDuckGo tool ──────────────────────────────────────────────

OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

_or_history: list = []
_or_llm: ChatOpenAI | None = None


def _get_or_llm() -> ChatOpenAI:
    global _or_llm
    if _or_llm is None:
        _or_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=OPENROUTER_MODEL,
        ).bind_tools([_search_tool])
    return _or_llm


async def chat_openrouter(user_message: str) -> str:
    llm = _get_or_llm()

    _or_history.append(HumanMessage(content=user_message))
    recent = _or_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent

    while True:
        response: AIMessage = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            result = _search_tool.run(tc["args"].get("query", ""))
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    content = response.content
    _or_history.append(AIMessage(content=content))
    return content


# ── Public streaming API ──────────────────────────────────────────────────────

async def stream_chat(user_message: str, engine: str = "ollama") -> AsyncGenerator[str, None]:
    """Resolve tool calls then yield the final response for SSE streaming."""
    if engine == "openrouter":
        result = await chat_openrouter(user_message)
    else:
        result = await chat_ollama(user_message)
    yield result


# ── Non-streaming fallback ────────────────────────────────────────────────────

async def chat(user_message: str, engine: str = "ollama") -> str:
    tokens = []
    async for token in stream_chat(user_message, engine=engine):
        tokens.append(token)
    return "".join(tokens)


def clear_history() -> None:
    _ollama_history.clear()
    _or_history.clear()
