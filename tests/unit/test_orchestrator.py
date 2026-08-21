# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from langchain_core.messages import AIMessage, HumanMessage

from eomas_assistant.agents.orchestrator import OrchestratorAgent
from eomas_assistant.models.schemas import OrchestratorPlan


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyStructuredModel:
    def __init__(self, answer: str, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error

    def invoke(self, input: list[object]) -> dict[str, object]:
        _ = input
        if self._error is not None:
            raise self._error
        import json

        payload = json.loads(self._answer)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload


class DummyChatOllama:
    def __init__(self, answer: str, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error

    def invoke(self, input: list[object]) -> DummyResponse:
        _ = input
        return DummyResponse(self._answer)

    def with_structured_output(self, json_schema: dict[str, object]) -> DummyStructuredModel:
        _ = json_schema
        return DummyStructuredModel(self._answer, self._error)


def test_orchestrator_routes_geography_by_keyword() -> None:
    agent = OrchestratorAgent(
        llm_client=DummyChatOllama(
            '{"route":"geography","reason":"llm classified keyword query","confidence":0.90}'
        )  # type: ignore
    )
    plan = agent.run(messages=[HumanMessage(content="Where is Bremen, Germany?")])
    assert plan.route == "geography"
    assert plan.expected_response_items == ["text", "map"]


def test_orchestrator_routes_geography_by_coordinates() -> None:
    agent = OrchestratorAgent(
        llm_client=DummyChatOllama(
            '{"route":"geography","reason":"llm classified coordinates","confidence":0.88}'
        )  # type: ignore
    )
    plan = agent.run(messages=[HumanMessage(content="51.0504, 13.7373")])
    assert plan.route == "geography"


def test_orchestrator_routes_by_llm_json_fallback() -> None:
    agent = OrchestratorAgent(
        llm_client=DummyChatOllama(
            '{"route":"geography","reason":"classified by llm","confidence":0.84}'
        )  # type: ignore
    )
    plan = agent.run(messages=[HumanMessage(content="Need position details for city center")])
    assert plan.route == "geography"


def test_orchestrator_returns_error_for_invalid_llm_output() -> None:
    agent = OrchestratorAgent(llm_client=DummyChatOllama("geography"))  # type: ignore
    plan = agent.run(messages=[HumanMessage(content="Need position details for city center")])
    assert plan.route == "error"
    assert plan.expected_response_items == ["text"]


def test_orchestrator_sanitizes_503_html_error_reason() -> None:
    agent = OrchestratorAgent(
        llm_client=DummyChatOllama(
            "{}",
            error=RuntimeError(
                "<html><head><title>503 Service Temporarily Unavailable</title></head></html>"
            ),
        )  # type: ignore
    )

    plan = agent.run(messages=[HumanMessage(content="Show me Bremen")])

    assert plan.route == "error"
    assert plan.reason == "Router service is temporarily unavailable (HTTP 503)."


def test_orchestrator_includes_conversation_context_in_prompt(mocker) -> None:
    captured: dict[str, object] = {}

    def _fake_call_llm_with_schema(**kwargs):
        captured["messages"] = kwargs["messages"]
        captured["supplemental_prompt"] = kwargs["supplemental_prompt"]
        return OrchestratorPlan(
            route="geography",
            reason="context-aware route",
            confidence=0.91,
        )

    mocker.patch(
        "eomas_assistant.agents.orchestrator.llm_helper.call_llm_with_schema",
        side_effect=_fake_call_llm_with_schema,
    )
    agent = OrchestratorAgent(llm_client=DummyChatOllama("{}"))  # type: ignore

    plan = agent.run(
        messages=[
            HumanMessage(content="Show me Bremen."),
            AIMessage(content="Bremen was resolved near 53.08, 8.80."),
            HumanMessage(content="Same area, but for January 2020."),
        ],
    )

    assert plan.route == "geography"
    captured_messages = captured["messages"]
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 3
    assert captured_messages[0].content == "Show me Bremen."
    assert captured_messages[1].content == "Bremen was resolved near 53.08, 8.80."
    assert captured_messages[2].content == "Same area, but for January 2020."
