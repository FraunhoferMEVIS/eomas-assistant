# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from eomas_assistant.agents.conversation import ConversationAgent
from eomas_assistant.agents.eo_imagery import EO_IMAGERY_TOOLS, EOImageryAgent
from eomas_assistant.agents.evaluator import EvaluatorAgent
from eomas_assistant.agents.geography import GeographyAgent
from eomas_assistant.agents.orchestrator import OrchestratorAgent
from eomas_assistant.config.settings import AppSettings, get_settings
from eomas_assistant.graph.state import AgentState
from eomas_assistant.graph.workflow_streamer import WorkflowStreamer
from eomas_assistant.llm.llm_helper import LLMHelper
from eomas_assistant.llm import create_llm_client
from eomas_assistant.models.response_models import AgentResponse, TextResponseItem
from eomas_assistant.tools.geocoding import Geocoding


class AgentWorkflow:
    """LangGraph workflow that orchestrates routing and agent execution."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        llm_client: BaseChatModel | None = None,
    ) -> None:
        """Wire dependencies and compile the workflow graph."""

        self._settings = settings or get_settings()
        self._llm_client = llm_client or create_llm_client(self._settings)

        geocoding_tool = Geocoding(
            base_url=self._settings.nominatim_base_url,
            user_agent=self._settings.nominatim_user_agent,
            timeout_seconds=self._settings.request_timeout_seconds,
        )

        self._orchestrator_agent = OrchestratorAgent(self._llm_client)
        self._conversation_agent = ConversationAgent(self._llm_client)
        self._geography_agent = GeographyAgent(geocoding_tool, self._llm_client)
        self._eo_imagery_agent = EOImageryAgent(self._llm_client)
        self._evaluator_agent = EvaluatorAgent(self._llm_client)

        self._graph = self._build_graph()
        self._streamer = WorkflowStreamer(self._graph)

    def run(
        self,
        messages: Sequence[AnyMessage],
    ) -> AgentResponse:
        """Run the compiled graph for a message history and return agent output."""

        state = self._graph.invoke(
            self._build_initial_state(messages)
        )
        response = WorkflowStreamer.extract_response_from_state(state)
        if response is not None:
            return response

        return WorkflowStreamer.empty_response()

    def stream(
        self,
        messages: Sequence[AnyMessage],
    ):
        """Yield incremental workflow status events and the final agent response."""

        yield from self._streamer.stream(
            self._build_initial_state(messages),
        )

    def _build_initial_state(
        self,
        messages: Sequence[AnyMessage],
    ) -> AgentState:
        """Create the initial workflow state for a new user request."""

        return AgentState(
            messages=list(messages),
            max_attempts=self._settings.workflow_max_attempts,
        )

    def _build_graph(self) -> CompiledStateGraph:
        """Create and compile the graph with routing and terminal nodes."""

        graph = StateGraph(AgentState)

        # The nodes automatically receive the full state dict and can return partial updates.
        graph.add_node("orchestrator", self._orchestrator_agent)
        graph.add_node("conversation", self._conversation_agent)
        graph.add_node("geography", self._geography_agent)
        graph.add_node("eo_imagery", self._eo_imagery_agent)
        graph.add_node("eo_imagery_tools", ToolNode(EO_IMAGERY_TOOLS))
        graph.add_node("evaluator", self._evaluator_agent)
        graph.add_node("unsupported", self._unsupported_node)
        graph.add_node("error", self._error_node)

        graph.add_edge(START, "orchestrator")
        graph.add_conditional_edges(
            "orchestrator",
            self._route_selector,
            {
                "conversation": "conversation",
                "geography": "geography",
                "unsupported": "unsupported",
                "error": "error",
            },
        )
        graph.add_conditional_edges(
            "geography",
            self._geography_outcome_selector,
            {
                "eo_imagery": "eo_imagery",
                "done": "evaluator",
            },
        )
        graph.add_edge("conversation", "evaluator")
        graph.add_conditional_edges(
            "eo_imagery",
            self._eo_imagery_outcome_selector,
            {
                "tools": "eo_imagery_tools",
                "done": "evaluator",
            },
        )
        graph.add_edge("eo_imagery_tools", "eo_imagery")
        graph.add_edge("unsupported", "evaluator")
        graph.add_edge("error", "evaluator")
        graph.add_conditional_edges(
            "evaluator",
            self._evaluation_selector,
            {
                "approved": END,
                "retry": "orchestrator",
                "done": END,
            },
        )

        result = graph.compile()
        if False:
            png = result.get_graph().draw_mermaid_png()
            print("saving workflow graph to graph_diagram.png")
            with open("graph_diagram.png", "wb") as f:
                f.write(png)
        return result

    def _unsupported_node(self, state: AgentState) -> dict:
        """Graph node: return a static message for non-geography intents."""

        plan = state.plan
        routing_reason = (
            plan.reason
            if plan is not None
            else "Orchestrator could not map the request to an implemented domain agent."
        )
        response = AgentResponse(
            agent_name="orchestrator_agent",
            items=[
                TextResponseItem(
                    content=(
                        "Sorry, I can only help with Earth observation, geospatial, "
                        "and geology-related requests."
                    )
                )
            ],
            metadata={
                "route": "unsupported",
                "user_query": LLMHelper.get_latest_user_message(state.messages),
                "route_reason": routing_reason,
                "response_source": "static_unsupported_message",
                "reasoning_trace": [
                    routing_reason,
                    "Static unsupported-response message was returned.",
                ],
            },
        )

        return dict(response=response)

    def _error_node(self, state: AgentState) -> dict:
        """Graph node: return a server/runtime error message."""

        plan = state.plan
        error_message = (
            plan.reason if plan is not None else "Unknown orchestrator error"
        )
        response = AgentResponse(
            agent_name="orchestrator_agent",
            items=[
                TextResponseItem(content=f"A server error occurred: {error_message}")
            ],
            metadata={
                "route": "error",
                "user_query": LLMHelper.get_latest_user_message(state.messages),
                "error_message": error_message,
                "response_source": "error_message",
                "reasoning_trace": [
                    error_message,
                    "Error-response message was returned.",
                ],
            },
        )

        return dict(response=response)

    def _route_selector(self, state: AgentState) -> str:
        """Return the routing key used by LangGraph conditional edges."""

        plan = state.plan
        if plan is None:
            return "error"
        return plan.route

    def _geography_outcome_selector(self, state: AgentState) -> str:
        """Continue to EO extraction whenever the planned response includes a map."""

        plan = state.plan
        if state.geo_location is None:
            return "done"
        if plan is not None and "map" in plan.expected_response_items:
            return "eo_imagery"
        return "done"

    def _eo_imagery_outcome_selector(self, state: AgentState) -> str:
        """Continue to EO imagery tools whenever the AI response contains tool calls."""

        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
            )
        if not last_message.tool_calls:
            return "done"
        return "tools"

    def _evaluation_selector(self, state: AgentState) -> str:
        """Approve, retry, or stop after the evaluator reviewed the latest output."""

        evaluation = state.evaluation
        if evaluation is None:
            return "done"
        if evaluation.approved:
            return "approved"

        if evaluation.retryable and state.attempt_count < state.max_attempts:
            return "retry"
        return "done"
