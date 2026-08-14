# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from langchain_core.messages import HumanMessage
import pytest

from eomas_assistant.config.settings import AppSettings
from eomas_assistant.graph.workflow import AgentWorkflow
from eomas_assistant.llm import create_llm_client
from eomas_assistant.models.response_models import TextResponseItem


@pytest.mark.integration
def test_live_llm_workflow_answers_ndvi_purpose_as_conversation() -> None:
    settings = AppSettings()
    llm_client = create_llm_client(settings)
    workflow = AgentWorkflow(settings=settings, llm_client=llm_client)

    state = workflow._graph.invoke(
        workflow._build_initial_state(
            [HumanMessage(content="what is the purpose of the NDVI?")],
        )
    )

    response = state.get("response")
    assert response is not None

    assert response.agent_name == "conversation_agent"
    assert response.metadata.get("route") == "conversation"

    text_items = [item for item in response.items if isinstance(item, TextResponseItem)]
    assert text_items

    answer_text = " ".join(item.content for item in text_items).lower()
    assert "ndvi" in answer_text
    assert "vegetation" in answer_text
