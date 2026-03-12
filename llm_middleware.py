"""
Vendor-neutral LLM middleware backed by LiteLLM.

Swap providers by changing two env vars — no code changes required anywhere:

    ACTIVE_LLM_VENDOR   REQUIRED KEY(S)              EXAMPLE MODELS
    ─────────────────   ─────────────────────────    ─────────────────────────────────────────
    OPENAI  (default)   OPENAI_API_KEY               gpt-4o, gpt-4o-mini
                                                      text-embedding-ada-002, text-embedding-3-small
    AZURE               AZURE_OPENAI_API_KEY          gpt-4o-mini  (prefix added automatically)
                        AZURE_OPENAI_ENDPOINT         text-embedding-ada-002
                        AZURE_API_VERSION
    ANTHROPIC           ANTHROPIC_API_KEY             claude-3-5-sonnet-20241022
                        (pair with OPENAI for         (no native embeddings — set EMBEDDING_MODEL
                         embeddings)                   to an OpenAI model and keep OPENAI_API_KEY)
    GROQ                GROQ_API_KEY                  groq/llama3-70b-8192
                        (same embedding caveat)
    OLLAMA              OLLAMA_API_BASE               ollama/llama3
                        (default http://localhost:    ollama/nomic-embed-text
                         11434, no key needed)

LiteLLM picks up all provider keys from standard environment variables
automatically.  Only AZURE requires explicit mapping because it uses
non-standard variable names.
"""

import os
from typing import Any

import litellm

import config

# Silence LiteLLM's verbose success logs; warnings and errors still surface.
litellm.success_callback = []
litellm.set_verbose = False


# ---------------------------------------------------------------------------
# Internal namespace objects that mirror the AsyncOpenAI client surface
# ---------------------------------------------------------------------------

class _Completions:
    def __init__(self, mw: "LLMMiddleware") -> None:
        self._mw = mw

    async def create(self, model: str, messages: list, **kwargs: Any):
        """Route chat completions through LiteLLM."""
        return await litellm.acompletion(
            model=self._mw._resolve_model(model),
            messages=messages,
            **kwargs,
        )


class _Chat:
    def __init__(self, mw: "LLMMiddleware") -> None:
        self.completions = _Completions(mw)


class _Embeddings:
    def __init__(self, mw: "LLMMiddleware") -> None:
        self._mw = mw

    async def create(self, model: str, input: Any, **kwargs: Any):
        """Route embedding requests through LiteLLM."""
        return await litellm.aembedding(
            model=self._mw._resolve_model(model),
            input=input,
            **kwargs,
        )


class LiteLLMClient:
    """
    Drop-in async client that mirrors the AsyncOpenAI interface:

        client.chat.completions.create(model, messages, **kwargs)
        client.embeddings.create(model, input, **kwargs)

    All calls are routed through LiteLLM, which translates them to the
    correct provider API and returns OpenAI-format response objects.
    """

    def __init__(self, mw: "LLMMiddleware") -> None:
        self.chat = _Chat(mw)
        self.embeddings = _Embeddings(mw)

    async def close(self) -> None:
        pass  # LiteLLM is stateless; nothing to clean up.


# ---------------------------------------------------------------------------
# Public middleware class
# ---------------------------------------------------------------------------

class LLMMiddleware:
    """
    Vendor-neutral LLM middleware.  Call initialize_client() to receive a
    LiteLLMClient that is a drop-in replacement for AsyncOpenAI /
    AsyncAzureOpenAI — no changes required in any caller.
    """

    def __init__(self) -> None:
        self.active_vendor: str = config.ACTIVE_LLM_VENDOR.upper()
        self.embedding_model: str = config.EMBEDDING_MODEL
        self._configure_litellm()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_litellm(self) -> None:
        """
        Push provider credentials into the environment variables that
        LiteLLM reads at call time.  Only Azure requires explicit mapping;
        all other providers are configured via their native env vars
        (ANTHROPIC_API_KEY, GROQ_API_KEY, OLLAMA_API_BASE, etc.).
        """
        if self.active_vendor == "OPENAI":
            if config.OPENAI_API_KEY:
                os.environ.setdefault("OPENAI_API_KEY", config.OPENAI_API_KEY)

        elif self.active_vendor == "AZURE":
            if config.AZURE_OPENAI_API_KEY:
                os.environ["AZURE_API_KEY"] = config.AZURE_OPENAI_API_KEY
            if config.AZURE_OPENAI_ENDPOINT:
                os.environ["AZURE_API_BASE"] = config.AZURE_OPENAI_ENDPOINT
            if config.AZURE_API_VERSION:
                os.environ["AZURE_API_VERSION"] = config.AZURE_API_VERSION

        # ANTHROPIC / GROQ / OLLAMA / others: users set the native env
        # vars in .env and LiteLLM picks them up automatically.

    def _resolve_model(self, model: str) -> str:
        """
        Add the provider prefix that LiteLLM requires when the caller
        passes a bare model name:

            AZURE     "gpt-4o-mini"  →  "azure/gpt-4o-mini"
            others    pass through unchanged (caller should already
                      include any non-OpenAI prefix, e.g. "groq/llama3-…")
        """
        if self.active_vendor == "AZURE" and "/" not in model:
            return f"azure/{model}"
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_client(self) -> LiteLLMClient:
        """
        Return a vendor-neutral async client compatible with the
        AsyncOpenAI interface.  Change ACTIVE_LLM_VENDOR in .env to
        switch providers without touching application code.
        """
        return LiteLLMClient(self)
