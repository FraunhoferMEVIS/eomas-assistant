# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from time import sleep
from typing import TypeVar

import httpx
from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

SchemaModel = TypeVar("SchemaModel", bound=BaseModel)
logger = logging.getLogger(__name__)


_MAX_RETRY_ATTEMPTS = 2
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def get_latest_user_message(messages: Sequence[AnyMessage]) -> str:
    """Return the latest user-authored message content from message history."""

    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = get_plain_text(message)
        return content
    return ""


def build_system_prompt_with_current_date(
    system_prompt: str,
) -> str:
    """Append current UTC date context to a system prompt."""

    current_date = datetime.now(UTC).date().isoformat()
    return f"The current date is: {current_date}\n{system_prompt}"


def build_model_input_messages(
    system_prompt: str,
    messages: Sequence[AnyMessage],
    supplemental_prompt: str | None = None,
) -> list[AnyMessage]:
    """Build model input from native chat history plus optional task metadata."""

    input_messages: list[AnyMessage] = [
        SystemMessage(content=build_system_prompt_with_current_date(system_prompt)),
        *list(messages),
    ]
    if supplemental_prompt and supplemental_prompt.strip():
        input_messages.append(SystemMessage(content=supplemental_prompt.strip()))
    for message in input_messages:
        logger.debug(
            "LLM input %s: %s",
            message.__class__.__name__,
            describe_message_content(message),
        )
    return input_messages


def call_llm_with_schema(
    llm: BaseChatModel,
    system_prompt: str,
    messages: Sequence[AnyMessage],
    schema_model: type[SchemaModel],
    call_site: str = "unknown",
    supplemental_prompt: str | None = None,
) -> SchemaModel:
    """Run a structured LLM call and validate the result against a Pydantic schema.
    The `call_site` is used for logging only.
    """

    llm_input = build_model_input_messages(
        system_prompt=system_prompt,
        messages=messages,
        supplemental_prompt=supplemental_prompt,
    )
    # Ask LangChain to constrain the model output to the JSON schema generated
    # from the requested Pydantic model.
    payload: object | None = None
    for attempt in range(_MAX_RETRY_ATTEMPTS):
        try:
            logger.info(
                "Calling structured LLM service (call_site=%s, schema=%s, attempt=%s/%s)",
                call_site,
                schema_model.__name__,
                attempt + 1,
                _MAX_RETRY_ATTEMPTS,
            )
            payload = llm.with_structured_output(schema_model.model_json_schema()).invoke(
                input=llm_input
            )
            break
        except Exception as exc:
            retryable = _is_retryable_structured_call_error(exc)
            logger.warning(
                "Structured LLM call failed (call_site=%s, schema=%s,"
                " attempt=%s/%s, retryable=%s, error_type=%s, error=%s)",
                call_site,
                schema_model.__name__,
                attempt + 1,
                _MAX_RETRY_ATTEMPTS,
                retryable,
                type(exc).__name__,
                exc,
            )
            if attempt + 1 >= _MAX_RETRY_ATTEMPTS or not retryable:
                raise
            # Small fixed backoff for short-lived provider overloads.
            sleep(0.25 * (attempt + 1))

    if payload is None:
        raise ValueError("Model output must be a JSON object.")

    # Different providers/adapters may return either a raw dict or a Pydantic-like
    # object, so normalize both forms before validating against the target schema.
    if isinstance(payload, dict):
        return schema_model.model_validate(payload)
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()
        if isinstance(dumped, dict):
            return schema_model.model_validate(dumped)

    # Fail fast if the model response cannot be interpreted as a JSON object.
    raise ValueError("Model output must be a JSON object.")


def _is_retryable_structured_call_error(exc: Exception) -> bool:
    """Return True for transient transport/status failures worth retrying once."""

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES

    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "502",
            "503",
            "504",
            "service temporarily unavailable",
            "gateway timeout",
            "too many requests",
        )
    )


def describe_message_content(ai_message: AnyMessage) -> str:
    if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
        return (
            f"[tool call(s): {', '.join(tool_call['name'] for tool_call in ai_message.tool_calls)}]"
        )
    return get_plain_text(ai_message) or "[no content]"


def get_plain_text(ai_message: AnyMessage) -> str:
    """Format heterogeneous message content as plain text."""

    content = getattr(ai_message, "content", ai_message)

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""
