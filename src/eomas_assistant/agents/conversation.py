# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage

from eomas_assistant.llm.llm_helper import LLMHelper
from eomas_assistant.models.response_models import AgentResponse, TextResponseItem
from eomas_assistant.graph.state import AgentState

SYSTEM_PROMPT = (
    "You are the general conversation agent for an Earth observation assistant. "
    "Respond directly in plain text. "
    "Handle greetings, short meta questions, capability questions, general chat about the assistant, and conceptual Earth observation questions that can be answered directly in text. "
    "Examples include defining NDVI, explaining cloud cover filtering, or describing the difference between imagery layers. "
    "Do not fabricate geocoding results, imagery availability, or map data. "
    "Keep answers concise and useful."
)


class ConversationAgent:
    """Responds directly to general conversation and assistant meta questions."""

    def __init__(self, llm_client: BaseChatModel) -> None:
        self._llm_client = llm_client

    def __call__(self, state: AgentState) -> dict:
        """Graph node: answer general conversation directly."""

        response = self.run(state.messages)
        plan = state.plan
        if plan is not None:
            response.metadata["route_reason"] = plan.reason
            response.metadata["outputs"] = plan.expected_response_items

        return dict(response=response)

    def run(
        self,
        messages: Sequence[AnyMessage],
    ) -> AgentResponse:
        """Return a direct plain-text response for general conversation."""

        try:
            llm_input = LLMHelper.build_model_input_messages(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
            )
            payload = self._llm_client.invoke(input=llm_input)
            content = LLMHelper.get_plain_text(payload)
        except Exception:
            content = None

        if not content:
            content = "I can help with Earth observation, geospatial questions, and general questions about this assistant."

        return AgentResponse(
            agent_name="conversation_agent",
            items=[TextResponseItem(content=content)],
            metadata={
                "route": "conversation",
                "outputs": ["text"],
                "reasoning_trace": [
                    "Orchestrator routed the request to the general conversation agent.",
                ],
            },
        )
