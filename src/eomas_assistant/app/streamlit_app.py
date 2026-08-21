# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import logging
import os
import ssl
from datetime import UTC, datetime
from time import perf_counter

import httpx
import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from eomas_assistant.config.settings import get_settings
from eomas_assistant.graph.workflow import AgentWorkflow
from eomas_assistant.models.response_models import (
    AgentResponse,
    ErrorResponseItem,
    MapResponseItem,
    TextResponseItem,
)
from eomas_assistant.ui.renderers import render_agent_response

SHORT_TERM_CONTEXT_MESSAGES = 6


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger("eomas_assistant").setLevel(level)


_configure_logging()


@st.cache_resource
def get_workflow() -> AgentWorkflow:
    return AgentWorkflow()


@st.cache_resource
def get_llm_connection_status() -> dict[str, object]:
    """Check LLM availability once at app startup."""

    settings = get_settings()
    endpoint = (
        f"{settings.llm_base_url.rstrip('/')}/api/tags"
        if settings.llm_provider == "ollama"
        else f"{settings.llm_base_url.rstrip('/')}/models"
    )
    verify = ssl.create_default_context(cafile=settings.llm_ca_bundle_path)

    headers = {}
    if settings.llm_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"

    started = perf_counter()
    try:
        with httpx.Client(
            timeout=settings.request_timeout_seconds,
            verify=verify,
            headers=headers,
        ) as client:
            response = client.get(endpoint)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        is_ok = response.status_code == 200
        detail = (
            "Connection successful."
            if is_ok
            else f"Unexpected status code: {response.status_code}."
        )
        return {
            "ok": is_ok,
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "detail": detail,
        }
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return {
            "ok": False,
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "status_code": None,
            "latency_ms": latency_ms,
            "detail": str(exc),
        }


def _init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(message["content"])
            else:
                render_agent_response(message["content"])


def _append_user_message(content: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": content})


def _append_assistant_message(content: AgentResponse) -> None:
    st.session_state.chat_history.append({"role": "assistant", "content": content})


def _build_messages(
    chat_history: list[dict[str, object]],
    max_messages: int = SHORT_TERM_CONTEXT_MESSAGES,
) -> list[AnyMessage]:
    """Return recent native chat messages from the latest UI history entries."""

    messages: list[AnyMessage] = []
    for message in chat_history[-max_messages:]:
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")

        if role == "user" and isinstance(content, str) and content.strip():
            messages.append(HumanMessage(content=content.strip()))
            continue

        if role == "assistant" and isinstance(content, AgentResponse):
            summary = _summarize_agent_response(content)
            if summary:
                messages.append(AIMessage(content=summary))

    return messages


def _summarize_agent_response(response: AgentResponse) -> str:
    """Render an assistant response into a compact single-string context snippet."""

    snippets: list[str] = []
    for item in response.items:
        if isinstance(item, TextResponseItem) and item.content.strip():
            snippets.append(item.content.strip())
        elif isinstance(item, ErrorResponseItem) and item.message.strip():
            snippets.append(f"Error: {item.message.strip()}")
        elif isinstance(item, MapResponseItem):
            snippets.append(
                f"Map: {item.title}"
                f" (center {item.center_latitude:.4f}, {item.center_longitude:.4f})."
            )

    compact = " ".join(snippets)
    compact = " ".join(compact.split())
    return compact[:600]


def _render_sidebar(llm_status: dict[str, object]) -> None:
    settings = get_settings()

    with st.sidebar:
        st.header("Debug & Settings")

        st.subheader("Runtime settings")
        st.json(
            {
                "app_name": settings.app_name,
                "debug": settings.debug,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "llm_base_url": settings.llm_base_url,
                "nominatim_base_url": settings.nominatim_base_url,
                "request_timeout_seconds": settings.request_timeout_seconds,
                "sentinel_hub_wmts_base_url": settings.sentinel_hub_wmts_base_url,
                "sentinel_hub_instance_configured": bool(settings.sentinel_hub_instance_id),
                "sentinel_hub_tile_matrix_set": settings.sentinel_hub_tile_matrix_set,
                "titiler_base_url": settings.titiler_base_url,
                "stac_cache_root": settings.stac_cache_root,
            }
        )

        st.subheader("LLM connectivity")
        if llm_status.get("ok"):
            st.success("LLM server reachable")
        else:
            st.error("LLM server unreachable")
        st.json(llm_status)

        st.subheader("Session debug")
        history = st.session_state.chat_history
        user_messages = sum(1 for message in history if message.get("role") == "user")
        assistant_messages = sum(1 for message in history if message.get("role") == "assistant")

        st.metric("Total messages", len(history))
        st.caption(f"User: {user_messages} | Assistant: {assistant_messages}")

        last_assistant = next(
            (message for message in reversed(history) if message.get("role") == "assistant"),
            None,
        )

        if last_assistant is None:
            st.info("No assistant response yet.")
        else:
            response = last_assistant.get("content")
            if isinstance(response, AgentResponse):
                st.write(f"Last agent: {response.agent_name}")
                route = response.metadata.get("route", "n/a")
                st.write(f"Last route: {route}")
                attempt_count = response.metadata.get("attempt_count")
                max_attempts = response.metadata.get("max_attempts")
                evaluation_status = response.metadata.get("evaluation_status")
                if isinstance(attempt_count, int) and isinstance(max_attempts, int):
                    st.write(f"Attempts: {attempt_count}/{max_attempts}")
                if isinstance(evaluation_status, str) and evaluation_status.strip():
                    st.write(f"Evaluation: {evaluation_status}")
                with st.expander("Last response metadata", expanded=False):
                    st.json(response.metadata)

        if st.button("Clear chat history", width="stretch"):
            st.session_state.chat_history = []
            st.rerun()


def main() -> None:
    settings = get_settings()
    llm_status = get_llm_connection_status()

    st.set_page_config(page_title=settings.app_name, page_icon="🛰️", layout="centered")
    st.title(f"🛰️ {settings.app_name}")
    st.caption("Extensible agentic earth observation chat assistant")

    st.markdown(
        """
        <style>
            .stChatMessage {
                border-radius: 12px;
                padding: 0.6rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _init_session_state()
    if settings.debug:
        _render_sidebar(llm_status)
    _render_chat_history()

    user_prompt = st.chat_input(
        "Ask a geographical question (e.g., Show me Bremen in January 2023"
        " with less than 20% cloud cover in the blue band)"
    )
    if not user_prompt:
        return

    messages = _build_messages(st.session_state.chat_history)
    messages.append(HumanMessage(content=user_prompt))

    _append_user_message(user_prompt)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    workflow = get_workflow()
    with st.chat_message("assistant"):
        status_box = st.status("Analyzing request...", expanded=True)
        response: AgentResponse | None = None

        for event in workflow.stream(messages):
            event_type = event.get("type")

            if event_type == "status":
                message = event.get("message")
                if isinstance(message, str) and message.strip():
                    status_box.write(message)
                continue

            if event_type == "final_response":
                streamed_response = event.get("response")
                if isinstance(streamed_response, AgentResponse):
                    response = streamed_response

        status_box.update(label="Analysis complete", state="complete", expanded=False)

        if response is None:
            response = AgentResponse(
                agent_name="system",
                items=[TextResponseItem(content="No response generated.")],
                metadata={"status": "empty"},
            )

        render_agent_response(response)
    _append_assistant_message(response)


if __name__ == "__main__":
    main()
