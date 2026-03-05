from __future__ import annotations

import os
import re

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


def _get_ollama_client() -> AsyncClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = AsyncClient()
    return _ollama_client


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def chat_ollama(user_message: str) -> str:
    client = _get_ollama_client()
    _ollama_history.append({"role": "user", "content": user_message})

    recent = _ollama_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

    response = await client.chat(
        model="qwen3.5:2b",
        messages=messages,
        think=False,
    )

    content = _strip_thinking(response.message.content)
    _ollama_history.append({"role": "assistant", "content": content})
    return content


# ── OpenRouter + DuckDuckGo tool ──────────────────────────────────────────────

OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

_or_history: list = []  # list of LangChain message objects
_search_tool = DuckDuckGoSearchRun()
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

    # Agentic tool-calling loop
    while True:
        response: AIMessage = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute each tool call and feed results back
        for tc in response.tool_calls:
            result = _search_tool.run(tc["args"].get("query", ""))
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    content = response.content
    _or_history.append(AIMessage(content=content))
    return content


# ── Public API ────────────────────────────────────────────────────────────────

async def chat(user_message: str, engine: str = "ollama") -> str:
    if engine == "openrouter":
        return await chat_openrouter(user_message)
    return await chat_ollama(user_message)


def clear_history() -> None:
    _ollama_history.clear()
    _or_history.clear()
