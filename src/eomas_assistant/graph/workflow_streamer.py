# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

from langgraph.graph.state import CompiledStateGraph

from eomas_assistant.graph.state import AgentState
from eomas_assistant.models.response_models import AgentResponse, TextResponseItem
from eomas_assistant.models.schemas import (
    AssetCatalog,
    DataRequest,
    EvaluationResult,
    GeoLocation,
    OrchestratorPlan,
)

logger = logging.getLogger(__name__)


class WorkflowStreamer:
    """Translate LangGraph events into UI-friendly streaming updates."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    def stream(
        self,
        initial_state: AgentState,
    ) -> Iterator[dict[str, object]]:
        """Yield incremental workflow status events and the final agent response."""

        event_stream = self._graph.astream_events(initial_state)
        final_response: AgentResponse | None = None

        with asyncio.Runner() as runner:
            try:
                while True:
                    try:
                        event = runner.run(event_stream.__anext__())
                    except StopAsyncIteration:
                        break

                    yield from self._stream_updates_from_event(event)

                    response = self._extract_response_from_event(event)
                    if response is not None:
                        final_response = response
            finally:
                runner.run(event_stream.aclose())

        yield {
            "type": "final_response",
            "response": final_response or self.empty_response(),
        }

    @staticmethod
    def empty_response() -> AgentResponse:
        """Return a deterministic fallback response for empty graph outputs."""

        return AgentResponse(
            agent_name="system",
            items=[TextResponseItem(content="No response generated.")],
            metadata={"status": "empty"},
        )

    @staticmethod
    def extract_response_from_state(state: object) -> AgentResponse | None:
        """Read a typed response from a graph state dictionary when available."""

        if not isinstance(state, dict):
            return None

        response = state.get("response")
        if isinstance(response, AgentResponse):
            return response
        return None

    def _extract_response_from_event(self, event: dict[str, object]) -> AgentResponse | None:
        """Read a typed response from a LangGraph stream event when available."""

        data = event.get("data")
        if not isinstance(data, dict):
            return None

        return self.extract_response_from_state(data.get("output"))

    def _stream_updates_from_event(self, event: dict[str, object]) -> Iterator[dict[str, object]]:
        """Translate raw LangGraph events into compact UI status updates."""

        event_name = event.get("event")
        node_name = event.get("name")
        if not isinstance(node_name, str) or node_name == "LangGraph":
            return

        if event_name == "on_chain_start":
            message = self._node_start_message(node_name)
            if message is not None:
                yield {"type": "status", "node": node_name, "message": message}
            return

        if event_name == "on_chain_end":
            for message in self._node_complete_messages(node_name, event.get("data")):
                yield {"type": "status", "node": node_name, "message": message}

    def _node_start_message(self, node_name: str) -> str | None:
        """Return a user-facing status message when a workflow node starts."""

        messages = {
            "orchestrator": "Routing request...",
            "conversation": "Preparing direct response...",
            "geography": "Resolving geographic context...",
            # "eo_imagery": "Discovering available EO assets...",
            "data_download": "Selecting and downloading EO imagery...",
            "evaluator": "Reviewing result quality...",
            "unsupported": "Preparing fallback response...",
            "error": "Preparing error response...",
        }
        return messages.get(node_name)

    def _node_complete_messages(self, node_name: str, data: object) -> list[str]:
        """Return a user-facing status message when a workflow node completes."""

        output = data.get("output") if isinstance(data, dict) else None
        output_state = output if isinstance(output, dict) else {}

        if node_name == "orchestrator":
            plan: OrchestratorPlan | None = output_state.get("plan")
            if plan is None:
                return ["Routing complete."]
            return [f"Routing complete: {plan.route}."]

        elif node_name == "geography":
            geo_location: GeoLocation | None = output_state.get("geo_location")
            if geo_location is None:
                return ["No location was resolved for this request."]
            return [f"Location resolved: {geo_location.display_name}."]

        elif node_name == "conversation":
            return ["Direct response ready."]

        elif node_name == "eo_imagery":
            asset_catalog: AssetCatalog | None = output_state.get("asset_catalog")
            result = []
            if asset_catalog is not None:
                result.append(
                    "Asset discovery complete: "
                    f"{len(asset_catalog.available_asset_keys)} key(s) "
                    f"from {len(asset_catalog.available_stac_items)} item(s) "
                    f"over {len(asset_catalog.available_stac_items_by_date)} dates."
                )

            tool_calls = output_state["messages"][0].tool_calls
            if tool_calls:
                logger.debug(f"Tool calls: {tool_calls}")
                result.extend(
                    [f"EO imagery agent calls tool {tool_call['name']}" for tool_call in tool_calls]
                )

            return result

        elif node_name == "data_download":
            data_request: DataRequest | None = output_state.get("data_request")
            if data_request is None:
                return ["EO download did not produce a structured data request."]
            selection_reasons = ", ".join(data_request.selection_reasons)
            return [
                f"EO download selected asset {data_request.wmts_layer},"
                f" cloud coverage: {data_request.max_cloud_cover}.",
                f" Selection reasons: {selection_reasons}",
            ]

        elif node_name == "evaluator":
            evaluation: EvaluationResult | None = output_state.get("evaluation")
            if evaluation is None:
                return ["Review complete."]
            if evaluation.approved:
                return ["Response approved."]
            if evaluation.retryable:
                return ["Response needs revision. Replanning..."]
            return ["Response review complete."]

        elif node_name == "unsupported":
            return ["Fallback response ready."]

        elif node_name == "error":
            return ["Error response ready."]

        return []
