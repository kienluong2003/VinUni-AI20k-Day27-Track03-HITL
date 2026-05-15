# """LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

# import os

# from langchain_openai import ChatOpenAI


# def get_llm(temperature: float = 0.2) -> ChatOpenAI:
#     api_key = os.environ.get("OPENROUTER_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
#     return ChatOpenAI(
#         model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
#         base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
#         api_key=api_key,
#         temperature=temperature,
#     )


"""LLM factory. Returns an OpenAI chat model (no OpenRouter)."""

import os
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set — please add it to your .env")

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=temperature,
    )