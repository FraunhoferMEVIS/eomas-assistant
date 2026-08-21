# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage

from eomas_assistant.graph.state import AgentState
from eomas_assistant.graph.workflow_streamer import WorkflowStreamer
from eomas_assistant.llm import llm_helper
from eomas_assistant.models.response_models import (
    AgentResponse,
    ErrorResponseItem,
    MapResponseItem,
    TextResponseItem,
)
from eomas_assistant.models.schemas import EvaluationResult, OrchestratorPlan

SYSTEM_PROMPT = (
    "You review whether the assistant's latest output adequately answers the user's request. "
    "Interpret dates relative to the current date given above. "
    "Only judge success against the latest user request shown above. "
    "Only older conversation context as background for disambiguation / missing information. "
    "Approve when the current output is a reasonable final answer for the current request. "
    "Evaluate the rendered map and EO imagery evidence, not just the text summary. "
    "rendered_map_eo_image is the tiled EO image actually shown on the map. "
    # "stac_downloaded_images are the STAC frames downloaded as overlays. "
    "When the user asks for a specific EO layer or band such as NDVI or the red band,"
    " confirm that these entries match the request when they are present. "
    "Decide layer compatibility from the raw asset_key and asset_title values. "
    "You should infer common EO aliases and provider naming conventions yourself,"
    " for example WMTS RED can match STAC B04_10m and TRUE_COLOR can match TCI. "
    "Reject outputs that claim the right layer but only show/download mismatched imagery. "
    "Reject when the output misses the user's main intent, chooses the wrong output mode,"
    " or follows an obviously wrong route. "
    "Set retryable=true only when another orchestration attempt could improve the result. "
    "Set retryable=false for already-acceptable outputs, clear out-of-domain requests,"
    " or failures that should not loop further. "
    "The critique must be short and specific. "
    "The replanning_instructions must tell the orchestrator what to do differently"
    " on the next attempt when approved=false. "
    "The score must be a number between 0 and 1."
)


class EvaluatorAgent:
    """Review the current workflow result and decide whether replanning is needed."""

    def __init__(self, llm_client: BaseChatModel) -> None:
        self._llm_client = llm_client

    def __call__(
        self,
        state: AgentState,
    ) -> dict:
        """Graph node: review the latest workflow output and decide on replanning."""

        response = state.response or WorkflowStreamer.empty_response()
        evaluation = self.run(
            messages=state.messages,
            response=response,
            plan=state.plan,
            attempt_count=state.attempt_count,
            max_attempts=state.max_attempts,
        )

        if evaluation.approved:
            evaluation_status = "approved"
        elif state.attempt_count >= state.max_attempts:
            evaluation_status = "max_attempts_reached"
        elif not evaluation.retryable:
            evaluation_status = "rejected_no_retry"
        else:
            evaluation_status = "retry_requested"

        return dict(
            response=dict(
                evaluation=evaluation.model_dump(),
                attempt_count=state.attempt_count,
                max_attempts=state.max_attempts,
                evaluation_status=evaluation_status,
            ),
            evaluation=evaluation,
        )

    def run(
        self,
        messages: Sequence[AnyMessage],
        response: AgentResponse,
        plan: OrchestratorPlan | None = None,
        attempt_count: int = 1,
        max_attempts: int = 1,
    ) -> EvaluationResult:
        """Return a structured evaluation for the latest workflow output."""

        supplemental_prompt = (
            "Evaluation task metadata:\n"
            f"- current_attempt: {attempt_count} of {max_attempts}\n"
            f"Current plan:\n{self._format_plan(plan)}\n"
            f"Current output:\n{self._format_response(response)}\n"
            "Judge adequacy against the latest user message in the conversation above."
        )
        try:
            return llm_helper.call_llm_with_schema(
                llm=self._llm_client,
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                schema_model=EvaluationResult,
                call_site="evaluator.review_response",
                supplemental_prompt=supplemental_prompt,
            )
        except Exception:
            return self._fallback_evaluation(response)

    def _format_plan(self, plan: OrchestratorPlan | None) -> str:
        if plan is None:
            return "- none"
        return (
            f"- route: {plan.route}\n"
            f"- outputs: {plan.expected_response_items}\n"
            f"- reason: {plan.reason}"
        )

    def _format_response(self, response: AgentResponse) -> str:
        lines = [f"- agent_name: {response.agent_name}"]
        map_items: list[MapResponseItem] = []
        for item in response.items:
            if isinstance(item, TextResponseItem):
                compact = " ".join(item.content.split())
                lines.append(f"- text: {compact}")
            elif isinstance(item, MapResponseItem):
                map_items.append(item)
                lines.append(
                    "- map: "
                    f"title={item.title}, center=({item.center_latitude},"
                    f" {item.center_longitude}), zoom={item.zoom}"
                )
            elif isinstance(item, ErrorResponseItem):
                compact = " ".join(item.message.split())
                lines.append(f"- error: {compact}")

        route = response.metadata.get("route")
        if isinstance(route, str) and route:
            lines.append(f"- metadata.route: {route}")
        lines.extend(self._format_imagery_metadata(map_items, response.metadata))
        return "\n".join(lines)

    def _format_imagery_metadata(
        self,
        map_items: Sequence[MapResponseItem],
        metadata: dict[str, Any],
    ) -> list[str]:
        eo_images = self._tiled_images_as_dicts(map_items)
        stac_images = self._image_dicts(metadata.get("stac_images"))

        # The map renderer uses the first tiled EO image; STAC frames stay separate overlays.
        rendered_map_image = next(
            (image for image in eo_images if self._is_tiled_image(image)), None
        )
        return [
            f"- rendered_map_eo_image: {self._json_dump(rendered_map_image)}",
            f"- stac_downloaded_images: {self._json_dump(stac_images)}",
        ]

    def _tiled_images_as_dicts(self, map_items: Sequence[MapResponseItem]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for map_item in map_items:
            for image in map_item.eo_images:
                images.append(image.model_dump())
        return images

    def _image_dicts(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [image for image in value if isinstance(image, dict)]

    def _json_dump(self, value: Any) -> str:
        return json.dumps(value, default=str, ensure_ascii=True, separators=(",", ":"))

    def _is_tiled_image(self, image: dict[str, Any]) -> bool:
        tiles_url = image.get("tiles_url_template")
        tilejson_url = image.get("tilejson_url")
        return (
            isinstance(tiles_url, str)
            and bool(tiles_url.strip())
            or isinstance(tilejson_url, str)
            and bool(tilejson_url.strip())
        )

    def _fallback_evaluation(self, response: AgentResponse) -> EvaluationResult:
        has_error = any(isinstance(item, ErrorResponseItem) for item in response.items)
        if has_error:
            return EvaluationResult(
                approved=False,
                retryable=False,
                score=0.0,
                critique="The workflow returned an error response.",
                replanning_instructions="",
            )

        return EvaluationResult(
            approved=True,
            retryable=False,
            score=0.5,
            critique="Structured evaluation failed, so the current response was accepted.",
            replanning_instructions="",
        )
