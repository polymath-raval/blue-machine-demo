#!/usr/bin/env python3
"""Test script to verify Ollama tool calling support with qwen3.5:4b"""

import asyncio
from voice_bot.llm import chat_ollama, clear_history

async def test_tool_calling():
    """Test cases that should trigger tool calling"""
    
    test_queries = [
        "What's the current weather in San Francisco?",
        "Who won the latest NBA championship?",
        "Tell me about recent AI developments",
        "What's happening in the stock market today?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        response = await chat_ollama(query)
        print(f"Response: {response}\n")
        
    clear_history()

async def test_regular_chat():
    """Test regular chat without tool calling"""
    
    test_queries = [
        "Hello, what's your name?",
        "Tell me a joke",
        "What is 2+2?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        response = await chat_ollama(query)
        print(f"Response: {response}\n")
        
    clear_history()

async def main():
    print("Testing Ollama Tool Calling (qwen3.5:4b)")
    print("=" * 60)
    
    print("\n[TEST 1] Regular chat (should NOT use search tool)")
    await test_regular_chat()
    
    print("\n\n[TEST 2] Queries that might trigger search tool")
    await test_tool_calling()

if __name__ == "__main__":
    asyncio.run(main())
