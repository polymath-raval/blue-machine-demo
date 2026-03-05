from __future__ import annotations

import os
import re
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ollama import AsyncClient

load_dotenv()

# ── Shared ────────────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and conversational — "
    "they will be spoken aloud. Avoid markdown, bullet points, or special characters. "
    "Answer in plain natural sentences."
)


# ── Ollama ────────────────────────────────────────────────────────────────────

_ollama_client: AsyncClient | None = None
_ollama_history: list[dict] = []
_search_tool = DuckDuckGoSearchRun()

OLLAMA_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + "\n\n"
    "When you need current information or want to search for something, use this format:\n"
    "<search>your search query here</search>\n"
    "I will provide search results, then you can incorporate them into your response."
)


def _get_ollama_client() -> AsyncClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = AsyncClient()
    return _ollama_client


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def chat_ollama(user_message: str) -> str:
    """Chat with Ollama with tool calling support via search markers."""
    client = _get_ollama_client()
    
    _ollama_history.append({"role": "user", "content": user_message})
    recent = _ollama_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": OLLAMA_SYSTEM_PROMPT}] + recent

    while True:
        response_text = ""
        async for chunk in await client.chat(
            model="qwen3.5:4b",
            messages=messages,
            think=False,
            stream=True,
        ):
            token = chunk.message.content or ""
            response_text += token

        # Check for search tool marker
        search_match = re.search(r'<search>(.*?)</search>', response_text, re.DOTALL)
        
        if search_match:
            query = search_match.group(1).strip()
            print(f"[Tool Call] Searching for: {query}")
            result = _search_tool.run(query)
            
            # Add response and search result to conversation
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Search results:\n{result}\n\nNow provide your response based on these results."})
        else:
            _ollama_history.append({"role": "assistant", "content": response_text.strip()})
            return response_text.strip()


async def stream_ollama(user_message: str) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama one by one. Appends to history on completion."""
    client = _get_ollama_client()
    _ollama_history.append({"role": "user", "content": user_message})

    recent = _ollama_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

    full_response = ""
    async for chunk in await client.chat(
        model="qwen3.5:4b",
        messages=messages,
        think=False,
        stream=True,
    ):
        token = chunk.message.content or ""
        full_response += token
        yield token

    _ollama_history.append({"role": "assistant", "content": full_response.strip()})


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
    """
    Async generator that yields LLM tokens.
    - Ollama: uses chat_ollama with tool calling support
    - OpenRouter: tool calls resolved first, then full response yielded at once
    """
    if engine == "openrouter":
        result = await chat_openrouter(user_message)
        yield result
    else:
        result = await chat_ollama(user_message)
        yield result


# ── Non-streaming fallback (kept for compatibility) ───────────────────────────

async def chat(user_message: str, engine: str = "ollama") -> str:
    tokens = []
    async for token in stream_chat(user_message, engine=engine):
        tokens.append(token)
    return "".join(tokens)


def clear_history() -> None:
    _ollama_history.clear()
    _or_history.clear()
