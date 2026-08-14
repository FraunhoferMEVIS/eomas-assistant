# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from eomas_assistant.models.schemas import TiledEOImage


class TextResponseItem(BaseModel):
    """Represents a plain text message returned by an agent."""

    type: Literal["text"] = "text"
    content: str = Field(min_length=1)


class MapPoint(BaseModel):
    """Represents a single point marker shown on a map output."""

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    label: str | None = None


class MapResponseItem(BaseModel):
    """Represents a map visualization including center, zoom, and overlays."""

    type: Literal["map"] = "map"
    title: str
    center_latitude: float = Field(ge=-90.0, le=90.0)
    center_longitude: float = Field(ge=-180.0, le=180.0)
    zoom: int = Field(default=10, ge=1, le=20)
    points: list[MapPoint] = Field(default_factory=list)
    geojson: dict[str, Any] | None = None
    eo_images: list[TiledEOImage] = Field(default_factory=list)


class ErrorResponseItem(BaseModel):
    """Represents an error message produced during request handling."""

    type: Literal["error"] = "error"
    message: str


class AgentResponse(BaseModel):
    """Top-level response envelope containing ordered agent outputs and
    metadata. This is used for rendering in the UI.  (Therefore, some
    information is duplicated here, since the rendering only gets the response,
    not other parts of the state.)
    """

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    items: list[TextResponseItem | MapResponseItem | ErrorResponseItem]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def modify(
        response: AgentResponse,
        updates_or_response: dict[str, Any] | AgentResponse | None = None,
    ) -> AgentResponse | None:
        """LangGraph state reducer for modifying agent response metadata.  This
        is much more convenient than having to look up and modify the nested
        metadata everywhere, and we special-case the "reasoning_trace" key to
        append to the existing list rather than overwriting it.
        """
        if not isinstance(updates_or_response, dict):
            return updates_or_response
        new_metadata = dict(response.metadata)

        # for some elements, we want to apply custom inner reducers, like
        # appending to lists:
        reasoning_trace = updates_or_response.pop("reasoning_trace", None)
        if reasoning_trace:
            existing_trace = new_metadata.setdefault("reasoning_trace", [])
            existing_trace.extend(reasoning_trace)

        roi_cloud_cover = updates_or_response.pop("roi_cloud_cover", None)
        if roi_cloud_cover is not None:
            acquisition_date, roi_cc = roi_cloud_cover
            available_stac_images = new_metadata.get("available_stac_images")
            if isinstance(available_stac_images, list):
                for stac_image in available_stac_images:
                    if stac_image.get("acquisition_date") == acquisition_date:
                        stac_image["roi_cc"] = roi_cc
                        break

        new_metadata.update(updates_or_response)

        return AgentResponse(
            request_id=response.request_id,
            agent_name=response.agent_name,
            items=response.items,
            metadata=new_metadata,
        )
