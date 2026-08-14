# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from math import log2
from typing import Any
from pydantic import Field

from langchain.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage

from eomas_assistant.llm.llm_helper import LLMHelper
from eomas_assistant.graph.state import AgentState
from eomas_assistant.models.response_models import (
    AgentResponse,
    ErrorResponseItem,
    MapPoint,
    MapResponseItem,
    TextResponseItem,
)
from eomas_assistant.models.schemas import (
    BoundingBox,
    GeographySummary,
    GeoLocation,
    ResponseItemType,
    StrictBaseModel,
    TimeRange,
)
from eomas_assistant.tools.geocoding import Geocoding

logger = logging.getLogger(__name__)


GEOGRAPHY_SUMMARY_SYSTEM_PROMPT = (
    "You are a geography assistant for earth observability workflows. "
    "Produce a summary that consists of 2-4 concise sentences and includes location and coordinates."
)

GEOGRAPHY_EXTRACTION_SYSTEM_PROMPT = (
    "You should extract candidates for geographical locations and a desired time range from earth-observation requests. "
    "Return ONLY a JSON object according to the provided schema. "
    "`candidates` must be a list of strings for OpenStreetMap Nominatim geocoding, "
    "ordered with most likely candidates first. "
    "Do not include non-location parts such as action phrases and temporal filters in `candidates`. "
    "If no location is mentioned, return `candidates` as an empty array."
    "\n"
    "Interpret relative time expressions using the provided current date. "
    "For phrases like 'last N years', return the last N complete calendar years ending with the most recently completed year, "
    "unless the user explicitly asks to include the current partial year. "
    "If the user gives a full date, return the same ISO-8601 UTC date-time in both fields. "
    "If the user gives a month, return the first instant of the month as start_timepoint "
    "and the last instant of the month as end_timepoint. "
    "If the user gives only a year, return January 1st at 00:00:00 UTC as start_timepoint "
    "and December 31st at 23:59:59 UTC as end_timepoint. "
    "Only if no time information is specified at all, return null for start_timepoint and end_timepoint."
    "\n"
    "Do not include explanations."
)

DEFAULT_ZOOM_LEVEL = 11  # Default zoom level for point locations without area geometry
MIN_ZOOM_LEVEL = 1
MAX_ZOOM_LEVEL = 20


class GeographyExtraction(StrictBaseModel):
    """Combined location-candidate and time-range from user intent analysis."""

    candidates: list[str] = Field(
        default_factory=list,
        description="Ordered candidate place-name strings extracted before geocoding.",
    )
    start_timepoint: datetime | None = Field(
        default=None,
        description="Inclusive normalized start timestamp when present in the user request.",
    )
    end_timepoint: datetime | None = Field(
        default=None,
        description="Inclusive normalized end timestamp when present in the user request.",
    )


class GeographyAgent:
    """Resolves place-related queries into validated text and map outputs."""

    def __init__(self, geocoding_tool: Geocoding, llm_client: BaseChatModel) -> None:
        """Initialize with geocoding and LLM dependencies."""

        self._geocoding_tool = geocoding_tool
        self._llm_client = llm_client

    def __call__(self, state: AgentState) -> dict:
        """Graph node: extract geographical information (location, time, etc.) from the user query."""

        plan = state.plan
        response, geo_location = self.run(
            messages=state.messages,
            outputs=(
                plan.expected_response_items if plan is not None else ["text", "map"]
            ),
            prior_geo_location=state.geo_location,
        )

        if isinstance(geo_location, GeoLocation):
            return dict(geo_location=geo_location, response=response)
        else:
            return dict(response=response)

    def run(
        self,
        messages: Sequence[AnyMessage],
        outputs: list[ResponseItemType] | None = None,
        prior_geo_location: GeoLocation | None = None,
    ) -> tuple[AgentResponse, GeoLocation | None]:
        """Execute geography workflow and return response plus extracted geo context."""

        user_query = LLMHelper.get_latest_user_message(messages)
        logger.debug("GeographyAgent received query: %s", user_query)
        normalized_outputs = outputs or ["text", "map"]

        candidates, extracted_time_range = self._extract_location_and_time(
            user_query,
            messages,
        )
        location, attempts = self._resolve_location(candidates)
        if location is None and prior_geo_location is not None:
            location = prior_geo_location.model_copy(deep=True)

        if location is None:
            return (
                self._build_not_found_response(
                    user_query=user_query,
                    candidates=candidates,
                    attempts=attempts,
                ),
                None,
            )

        if extracted_time_range is not None:
            location.time_range = extracted_time_range
        elif (
            prior_geo_location is not None and prior_geo_location.time_range is not None
        ):
            location.time_range = prior_geo_location.time_range

        location.bbox_wgs84_lat_lon = self._extract_bbox_wgs84(location)

        return (
            self._build_success_response(
                user_query=user_query,
                candidates=candidates,
                attempts=attempts,
                location=location,
                messages=messages,
                outputs=normalized_outputs,
            ),
            location,
        )

    def _extract_location_and_time(
        self,
        user_query: str,
        messages: Sequence[AnyMessage],
    ) -> tuple[list[str], TimeRange | None]:
        """Extract location candidates and time range in a single model call."""

        normalized = user_query.strip()
        cleaned = normalized.strip(" ?!.,;:\t\n\r")
        if not cleaned:
            return [], None

        try:
            model_output = LLMHelper.call_llm_with_schema(
                llm=self._llm_client,
                system_prompt=GEOGRAPHY_EXTRACTION_SYSTEM_PROMPT,
                messages=messages,
                schema_model=GeographyExtraction,
                call_site="geography.extract_location_and_time",
                supplemental_prompt=(
                    "Extract location candidates and time range if given in the latest "
                    "user message in the conversation above."
                ),
            )
        except Exception:
            return [cleaned], None

        candidates: list[str] = []
        for candidate in model_output.candidates:
            value = candidate.strip(" ?!.,;:\t\n\r")
            if value and value not in candidates:
                candidates.append(value)

        if model_output.start_timepoint is None:
            start_timepoint = datetime.now(UTC) - timedelta(
                days=180
            )  # default to last 6 months
        else:
            start_timepoint = (
                model_output.start_timepoint.replace(tzinfo=UTC)
                if model_output.start_timepoint.tzinfo is None
                else model_output.start_timepoint
            )

        if model_output.end_timepoint is None:
            end_timepoint = datetime.now(UTC)
        else:
            end_timepoint = (
                model_output.end_timepoint.replace(tzinfo=UTC)
                if model_output.end_timepoint.tzinfo is None
                else model_output.end_timepoint
            )

        return (
            candidates,
            TimeRange(
                start_timepoint=start_timepoint,
                end_timepoint=end_timepoint,
            ),
        )

    def _extract_bbox_wgs84(self, location: GeoLocation) -> BoundingBox:
        """Derive a WGS84 bounding box from location geometry or fallback to point bounds."""

        if location.geojson is not None:
            coordinates = self._collect_coordinates(location.geojson)
            if coordinates:
                longitudes = [lon for lon, _ in coordinates]
                latitudes = [lat for _, lat in coordinates]
                return BoundingBox(
                    min_latitude=min(latitudes),
                    min_longitude=min(longitudes),
                    max_latitude=max(latitudes),
                    max_longitude=max(longitudes),
                )

        return BoundingBox(
            min_latitude=location.latitude,
            min_longitude=location.longitude,
            max_latitude=location.latitude,
            max_longitude=location.longitude,
        )

    def _collect_coordinates(self, value: Any) -> list[tuple[float, float]]:
        """Recursively collect [lon, lat] coordinate pairs from a GeoJSON-like object."""

        if isinstance(value, dict):
            coordinates: list[tuple[float, float]] = []
            for nested in value.values():
                coordinates.extend(self._collect_coordinates(nested))
            return coordinates

        if isinstance(value, (list, tuple)):
            if (
                len(value) >= 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))
            ):
                return [(float(value[0]), float(value[1]))]

            coordinates: list[tuple[float, float]] = []
            for nested in value:
                coordinates.extend(self._collect_coordinates(nested))
            return coordinates

        return []

    def _build_not_found_response(
        self,
        user_query: str,
        candidates: list[str],
        attempts: list[str],
    ) -> AgentResponse:
        """Build the response when no location match is found."""

        reasoning_trace = [
            "Geography agent received the request.",
            f"Tried geocoding {len(attempts)} candidate query/candidates.",
            "No OpenStreetMap match was found.",
        ]
        return AgentResponse(
            agent_name="geography_agent",
            items=[
                ErrorResponseItem(
                    message=f"No location result found for: {user_query}"
                ),
                TextResponseItem(
                    content="I could not find a matching place in OpenStreetMap. "
                    f"Try a more specific location name. Candidates: {', '.join(candidates)}"
                ),
            ],
            metadata={
                "source": "OpenStreetMap Nominatim",
                "geocode_attempts": attempts,
                "reasoning_trace": reasoning_trace,
            },
        )

    def _build_success_response(
        self,
        user_query: str,
        candidates: list[str],
        attempts: list[str],
        location: GeoLocation,
        messages: Sequence[AnyMessage],
        outputs: list[ResponseItemType],
    ) -> AgentResponse:
        """Build the response payload for a resolved location."""

        summary = self._create_summary(
            user_query=user_query,
            location=location,
            messages=messages,
        )
        zoom = (
            self._compute_zoom_from_bbox(location.bbox_wgs84_lat_lon)
            if location.bbox_wgs84_lat_lon
            else DEFAULT_ZOOM_LEVEL
        )
        response_items = self._build_response_items_for_mode(
            outputs=outputs,
            summary=summary,
            location=location,
            zoom=zoom,
        )
        return AgentResponse(
            agent_name="geography_agent",
            items=response_items,
            metadata={
                "source": location.source,
                "query": user_query,
                "resolved_query": location.query,
                "display_name": location.display_name,
                "outputs": outputs,
                "has_area_geometry": location.geojson is not None,
                "geocode_attempts": attempts,
                "reasoning_trace": [
                    "Geography agent received the request.",
                    f"Generated {len(candidates)} geocoding candidate query/candidates.",
                    f"Selected location: {location.display_name}.",
                    (
                        "Area boundary geometry found and attached to the map output."
                        if location.geojson is not None
                        else "No area boundary geometry found; using point marker only."
                    ),
                ],
            },
        )

    def _build_response_items_for_mode(
        self,
        outputs: list[ResponseItemType],
        summary: str,
        location: GeoLocation,
        zoom: int,
    ) -> list[TextResponseItem | MapResponseItem | ErrorResponseItem]:
        """Build user-visible outputs for the requested response mode."""

        map_response = MapResponseItem(
            title=f"Location map: {location.name}",
            center_latitude=location.latitude,
            center_longitude=location.longitude,
            zoom=zoom,
            points=[
                MapPoint(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    label=location.display_name,
                )
            ],
            geojson=location.geojson,
        )

        rendered_responses: list[
            TextResponseItem | MapResponseItem | ErrorResponseItem
        ] = []
        if "text" in outputs:
            rendered_responses.append(TextResponseItem(content=summary))
        if "map" in outputs:
            rendered_responses.append(map_response)
        return rendered_responses

    def _compute_zoom_from_bbox(self, bbox: BoundingBox) -> int:
        """Estimate a deck.gl-friendly zoom level from a WGS84 bounding box."""

        min_lon, min_lat, max_lon, max_lat = bbox.as_lon_lat_tuple()
        lon_span = max(0.0, max_lon - min_lon)
        lat_span = max(0.0, max_lat - min_lat)

        # Point locations should still render at a meaningful city-level zoom.
        if lon_span == 0.0 and lat_span == 0.0:
            return DEFAULT_ZOOM_LEVEL

        epsilon = 1e-6
        lon_zoom = log2(360.0 / max(lon_span, epsilon))
        lat_zoom = log2(180.0 / max(lat_span, epsilon))
        fitted_zoom = min(lon_zoom, lat_zoom) - 0.5

        return max(MIN_ZOOM_LEVEL, min(MAX_ZOOM_LEVEL, round(fitted_zoom)))

    def _resolve_location(
        self, candidates: list[str]
    ) -> tuple[GeoLocation | None, list[str]]:
        """Try multiple geocoding candidates and return the first successful match."""

        attempts: list[str] = []
        for candidate in candidates:
            attempts.append(candidate)
            location = self._geocoding_tool.geocode(candidate)
            if location is not None:
                return location, attempts
        return None, attempts

    def _create_summary(
        self,
        user_query: str,
        location: GeoLocation,
        messages: Sequence[AnyMessage],
    ) -> str:
        """Generate a short geographic summary for chat output.

        Falls back to a deterministic response when the configured LLM is not
        reachable or returns an error.
        """

        supplemental_prompt = (
            f"Question: {user_query}\n"
            f"Resolved place: {location.display_name}\n"
            f"Latitude: {location.latitude}\n"
            f"Longitude: {location.longitude}\n"
            "Provide a concise answer suitable for a chat UI."
        )
        try:
            summary = LLMHelper.call_llm_with_schema(
                llm=self._llm_client,
                system_prompt=GEOGRAPHY_SUMMARY_SYSTEM_PROMPT,
                messages=messages,
                schema_model=GeographySummary,
                call_site="geography.create_summary",
                supplemental_prompt=supplemental_prompt,
            )
        except Exception:
            summary = GeographySummary(
                summary=(
                    f"{location.name} is located at latitude {location.latitude:.5f} and "
                    f"longitude {location.longitude:.5f}. "
                    f"Resolved place: {location.display_name}."
                )
            )
        return summary.summary.strip()
