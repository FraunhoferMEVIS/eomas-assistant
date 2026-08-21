# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import collections
from datetime import datetime
from typing import Any, Literal

import pystac
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Output kinds that can be requested by the orchestrator.
ResponseItemType = Literal["text", "map"]


class StrictBaseModel(BaseModel):
    """BaseModel with strict validation and no extra fields allowed"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


### AGENT RESPONSE CLASSES


# Output schema for Orchestrator agent
class OrchestratorPlan(StrictBaseModel):
    """
    Captures the selected workflow plan returned by the orchestrator agent.
    """

    route: Literal["conversation", "geography", "unsupported", "error"] = Field(
        description="Selected workflow branch that should handle the current user request"
    )
    expected_response_items: list[ResponseItemType] = Field(
        default_factory=lambda: ["text", "map"],
        min_length=1,
        description="Requested user-visible outputs for the selected route, in display order",
    )
    reason: str = Field(description="Short explanation of why the orchestrator selected this plan")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Planner confidence in the selected route and requested outputs",
    )

    # TODO: The orchestrator should actually extract the user's intent along more dimensions:
    # - dates (single, range, selected dates for comparison)
    # - location phrases (single, eventually even multiple)
    # - request type / planned response (including diagrams/tables in addition to text/map)
    # - constraints (cloud cover, resolution, etc.)
    # - layers / values (NDVI, true color, etc.)

    @model_validator(mode="after")
    def validate_outputs(self) -> OrchestratorPlan:
        """Ensure outputs are unique while preserving the declared order"""

        deduplicated_outputs = list(dict.fromkeys(self.expected_response_items))
        if not deduplicated_outputs:
            raise ValueError("OrchestratorPlan requires at least one output type")
        object.__setattr__(self, "expected_response_items", deduplicated_outputs)
        return self


# Output schema for evaluation agent
class EvaluationResult(StrictBaseModel):
    """Structured review of whether the current workflow output satisfies the user request"""

    approved: bool = Field(
        description="Whether the current workflow output is acceptable as the final answer"
    )
    retryable: bool = Field(
        default=True,
        description="Whether another orchestration attempt could plausibly improve the output",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized quality score assigned by the evaluator",
    )
    critique: str = Field(
        min_length=1,
        description="Specific explanation of what is okay or insufficient about the current output",
    )
    replanning_instructions: str = Field(
        default="",
        description="Specific guidance the orchestrator should use on the next retry attempt",
    )


# FIXME: We really don't need structured output for a single text summary
class GeographySummary(StrictBaseModel):
    """Stores a concise generated summary for a geography answer"""

    summary: str = Field(
        min_length=1,
        description="Short natural-language summary of the resolved geography result",
    )


class AssetCatalog(StrictBaseModel):
    """Structured information on discovered EO assets for a resolved geo/time request"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    available_stac_items: list[pystac.Item] = Field(
        description="Discovered STAC items for the current query constraints",
    )

    available_asset_keys: list[str] = Field(
        default_factory=list,
        description="Deduplicated provider asset keys available for the current query constraints",
    )
    available_stac_items_by_date: dict[str, list[pystac.Item]] = Field(
        default_factory=lambda: collections.defaultdict(list),
        description="STAC items organized by acquisition date",
    )
    available_stac_images: list[StacImageCatalogEntry] = Field(
        default_factory=list,
        description="Per-item STAC catalog rows with acquisition date, asset key, and cloud cover"
        " for the current query constraints",
    )


class StacImageCatalogEntry(StrictBaseModel):
    """Compact STAC catalog row used for rendering availability tables"""

    acquisition_date: str = Field(
        default="unknown",
        description="Acquisition date/time in ISO-8601 UTC format when available",
    )
    stac_cc: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Scene cloud cover percentage when provided by STAC metadata",
    )


# Output schema for the geography agent
class GeoLocation(StrictBaseModel):
    """
    Represents a geocoded location with coordinates and metadata.
    Returned by the Geography Agent.
    """

    query: str = Field(
        description="Original location phrase or search text that produced this geocoding result"
    )
    name: str = Field(description="Compact canonical place name used internally by the workflow")
    latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Latitude of the representative point for the resolved location in WGS84",
    )  # TODO: Rather use bounding box than lat/long
    longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Longitude of the representative point for the resolved location in WGS84",
    )
    display_name: str = Field(description="Human-readable geocoder label shown to the user")
    geojson: dict[str, Any] | None = Field(
        default=None,
        description="GeoJSON geometry returned by the geocoder for map rendering",
    )
    bbox_wgs84_lat_lon: BoundingBox | None = Field(
        default=None,
        description="WGS84 bounding box describing the spatial extent of the location",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Time range of interest for downstream EO queries",
    )
    source: str = Field(
        default="OpenStreetMap Nominatim",
        description="Name of the geocoding provider that produced this location result",
    )


class DataRequest(StrictBaseModel):
    """Image parameters for a given location. Used by the data extract node"""

    wmts_layer: str | None = Field(
        default=None,
        description="Currently selected WMTS layer used for map visualization",
    )
    acquired_at: datetime | None = Field(
        default=None,
        description="Acquisition date of the selected scene",
    )
    max_cloud_cover: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable cloud coverage percentage (range 0..100)"
        " for candidate EO scenes",
    )
    selection_reasons: list[str] = Field(
        default_factory=list,
        description="Short justifications "
        "explaining why the selected imagery parameters were chosen",
    )


### GENERAL CLASSES


class TimeRange(StrictBaseModel):
    """Contains a normalized time range extracted from user query"""

    start_timepoint: datetime | None = Field(
        default=None,
        description="Inclusive normalized start timestamp of the requested temporal range",
    )
    end_timepoint: datetime | None = Field(
        default=None,
        description="Inclusive normalized end timestamp of the requested temporal range",
    )


class BoundingBox(StrictBaseModel):
    """Represents a geographic bounding box (by default in WGS84 coordinates)"""

    type: Literal["WGS84"] = (
        "WGS84"  # We can potentially extend this to support other coordinate systems in the future
    )
    min_latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Southern latitude boundary of the bounding box in WGS84",
    )
    min_longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Western longitude boundary of the bounding box in WGS84",
    )
    max_latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Northern latitude boundary of the bounding box in WGS84",
    )
    max_longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Eastern longitude boundary of the bounding box in WGS84",
    )

    def as_lon_lat_tuple(self) -> tuple[float, float, float, float]:
        """Return bounds ordered as longitude/latitude values for mapping and tile APIs"""

        return (
            self.min_longitude,
            self.min_latitude,
            self.max_longitude,
            self.max_latitude,
        )


class EOImage(StrictBaseModel):
    """Data class for EO images"""

    bbox_wgs84_lat_lon: BoundingBox = Field(description="Image bounding box in WGS84 coordinates")
    asset_key: str = Field(
        description="Provider-specific asset identifier used to retrieve the imagery"
    )
    asset_title: str = Field(description="Human-readable asset name shown in the user interface")
    acquired_at: datetime | None = Field(
        default=None,
        description="Acquisition timestamp of the selected scene when available",
    )


class LocalEOImage(EOImage):
    """An EO image with locally saved/cached image data"""

    source_path: str | None = Field(
        default=None,
        description="Filesystem path to a local image file",
    )


class TiledEOImage(EOImage):
    """An EO image that can be accessed via XYZ tiles or TileJSON"""

    tiles_url_template: str | None = Field(
        default=None,
        description="XYZ tile URL template used to request map tiles",
    )
    tilejson_url: str | None = Field(
        default=None,
        description="TileJSON document URL describing a tiled image source",
    )
    min_zoom: int | None = Field(
        default=None,
        ge=0,
        description="Minimum supported zoom level for the tiled source",
    )
    max_zoom: int | None = Field(
        default=None,
        ge=0,
        description="Maximum supported zoom level for the tiled source",
    )
    tile_size: int | None = Field(
        default=None,
        gt=0,
        description="Tile edge length in pixels for tiled imagery responses",
    )

    @model_validator(mode="after")
    def validate_tile_source(self) -> TiledEOImage:
        has_template = bool(self.tiles_url_template and self.tiles_url_template.strip())
        has_tilejson = bool(self.tilejson_url and self.tilejson_url.strip())
        if not (has_template or has_tilejson):
            raise ValueError("TiledEOImage requires either tiles_url_template or tilejson_url")
        return self
