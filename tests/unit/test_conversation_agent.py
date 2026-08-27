# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from langchain_core.messages import AIMessage, HumanMessage

from eomas_assistant.models.response_models import TextResponseItem
from eomas_assistant.nodes.conversation import ConversationAgent


def test_conversation_agent_answers_ndvi_purpose_question(mocker) -> None:
    llm_client = mocker.Mock()
    llm_client.invoke.return_value = AIMessage(
        content=(
            "NDVI helps evaluate vegetation health and activity by comparing "
            "near-infrared and red reflectance."
        )
    )

    agent = ConversationAgent(llm_client=llm_client)
    response = agent.run(messages=[HumanMessage(content="what is the purpose of the NDVI?")])

    assert response.agent_name == "conversation_agent"
    assert response.metadata["route"] == "conversation"
    assert len(response.items) == 1
    text_item = response.items[0]
    assert isinstance(text_item, TextResponseItem)
    assert "ndvi" in text_item.content.lower()
    assert "vegetation" in text_item.content.lower()
