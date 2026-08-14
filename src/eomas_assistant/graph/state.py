# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import Field
from typing import Annotated

from eomas_assistant.models.response_models import AgentResponse
from eomas_assistant.models.schemas import (
    StrictBaseModel,
    AssetCatalog,
    DataRequest,
    EvaluationResult,
    GeoLocation,
    OrchestratorPlan,
)


class AgentState(StrictBaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )

    attempt_count: int = 0  # Number of orchestration attempts already executed.
    max_attempts: int = 1  # Hard stop for evaluation-triggered replanning (overwritten from settings).

    plan: OrchestratorPlan | None = None  # Full orchestration plan with routing and output plan.
    geo_location: GeoLocation | None = None  # Extracted geo location for data retrieval.
    asset_catalog: AssetCatalog | None = None  # Discovered assets for current geo/time query.
    data_request: DataRequest | None = None  # EO retrieval parameters extracted by data agent.
    response: Annotated[AgentResponse | None, AgentResponse.modify] = None  # Final agent response returned to the caller.
    evaluation: EvaluationResult | None = None  # Review result for the latest workflow output.
