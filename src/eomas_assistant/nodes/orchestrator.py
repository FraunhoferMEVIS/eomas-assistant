# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage

from eomas_assistant.graph.state import AgentState
from eomas_assistant.llm import llm_helper
from eomas_assistant.models.schemas import (
    EvaluationResult,
    OrchestratorPlan,
)

SYSTEM_PROMPT = (
    "You are the orchestration agent for an Earth observation assistant,"
    " and your responsibility is to assess the *intent* of the user query. "
    "Use the `conversation` route for greetings, capability questions, general"
    " meta-conversation, simple assistant chat, and conceptual Earth observation"
    " questions that can be answered directly in text without resolving a"
    " location or retrieving imagery. "
    "Treat the `geography` route as the supported domain route for all Earth observation,"
    " geospatial, mapping, remote-sensing, satellite-imagery, and geology-adjacent requests, "
    " including locations, coordinates, bounding boxes, time ranges, NDVI, spectral bands, "
    " cloud cover, Sentinel imagery, vegetation, land cover, terrain, and map-based"
    " follow-up questions. "
    "When the user primarily wants explanation, discussion, or interpretation,"
    " prefer `outputs`=[text]. "
    "When the user primarily wants a visual result, map display, or imagery layer,"
    " prefer `outputs`=[map]. "
    "When the user wants both explanation and visualization, set `outputs`=[text, map]. "
    "For follow-up questions such as 'Now in NDVI', 'Use the same area in 2021',"
    " or 'Show the same scene with less cloud cover', use the conversation"
    " context to resolve omitted references. "
    "Return `unsupported` only when the current request is clearly outside that domain. "
)


class OrchestratorAgent:
    """Plans which agent should handle the request and how it should respond."""

    def __init__(self, llm_client: BaseChatModel) -> None:
        """Initialize the orchestrator with a chat model instance."""

        self._llm_client = llm_client

    def __call__(self, state: AgentState) -> dict:
        """Graph node: classify request and store plan in state."""

        attempt_count = state.attempt_count + 1
        plan = self.run(
            messages=state.messages,
            evaluation=state.evaluation,
        )

        return dict(
            attempt_count=attempt_count,
            plan=plan,
        )

    def run(
        self,
        messages: Sequence[AnyMessage],
        evaluation: EvaluationResult | None = None,
    ) -> OrchestratorPlan:
        """Return a workflow plan for the incoming user query."""

        supplemental_prompt = None
        if evaluation is not None and not evaluation.approved:
            supplemental_prompt = (
                "Previous attempt review:\n"
                f"- critique: {evaluation.critique}\n"
                f"- replanning instructions: {evaluation.replanning_instructions or 'none'}"
            )

        try:
            plan = llm_helper.call_llm_with_schema(
                llm=self._llm_client,
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                schema_model=OrchestratorPlan,
                call_site="orchestrator.route_plan",
                supplemental_prompt=supplemental_prompt,
            )
            plan.reason = plan.reason.strip() or "LLM router plan"
        except Exception as exc:
            plan = OrchestratorPlan(
                route="error",
                expected_response_items=["text"],
                reason=self._format_router_error_reason(exc),
                confidence=0.0,
            )
        return plan

    @staticmethod
    def _format_router_error_reason(exc: Exception) -> str:
        """Return a concise user-facing router error reason."""

        message = str(exc).strip()
        lowered = message.lower()
        if "503" in lowered or "service temporarily unavailable" in lowered:
            return "Router service is temporarily unavailable (HTTP 503)."
        if "429" in lowered or "too many requests" in lowered:
            return "Router is rate-limited right now (HTTP 429)."
        if "<html" in lowered:
            return "Router returned an invalid non-JSON response."

        first_line = next((line for line in message.splitlines() if line.strip()), "")
        cleaned_line = re.sub(r"<[^>]*>", "", first_line).strip()
        if cleaned_line:
            return f"Router failed or returned invalid output: {cleaned_line[:180]}"
        return "Router failed or returned invalid output."
