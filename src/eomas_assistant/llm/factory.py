# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import ssl

import httpx
from langchain.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from eomas_assistant.config.settings import AppSettings


def create_ollama_client(settings: AppSettings) -> ChatOllama:
    """Create a configured ChatOllama client from application settings."""

    client_kwargs: dict[str, object] = {"timeout": settings.request_timeout_seconds}
    client_kwargs["verify"] = ssl.create_default_context(
        cafile=settings.llm_ca_bundle_path
    )

    model = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=0.0,
        client_kwargs=client_kwargs,
    )

    return model


def create_vllm_openai_client(settings: AppSettings) -> ChatOpenAI:
    """Create a configured ChatOpenAI client for the EVE-Instruct model."""

    verify_value = ssl.create_default_context(cafile=settings.llm_ca_bundle_path)

    model = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.vllm_api_key,
        timeout=settings.request_timeout_seconds,
        temperature=0.0,
        http_client=httpx.Client(verify=verify_value),
    )

    return model


def create_llm_client(settings: AppSettings) -> BaseChatModel:
    """Create a provider-agnostic chat model client.

    Provider selection is controlled via `LLM_PROVIDER` environment variable:
    - `ollama`
    - `vllm_openai` (default)
    """

    provider = settings.llm_provider

    if provider == "ollama":
        return create_ollama_client(settings)
    if provider == "vllm_openai":
        return create_vllm_openai_client(settings)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'ollama' or 'vllm_openai'."
    )
