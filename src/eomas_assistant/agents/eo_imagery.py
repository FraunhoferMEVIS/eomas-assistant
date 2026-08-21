# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations
from functools import lru_cache

import traceback
from collections.abc import Sequence
from typing import cast
from datetime import datetime

from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolMessage,
)
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
import numpy as np

from eomas_assistant.llm import llm_helper
from eomas_assistant.models.response_models import MapResponseItem
from eomas_assistant.models.schemas import (
    AssetCatalog,
    DataRequest,
    GeoLocation,
    StacImageCatalogEntry,
)
from eomas_assistant.graph.state import AgentState
from eomas_assistant.tools.rasterize_geojson import rasterize_geojson
from eomas_assistant.tools.downloader import EOImageDownloader
from eomas_assistant.tools.find_available_satellite_data import (
    find_sentinel2_assets_in_time_range,
    format_acquisition_date,
)
from eomas_assistant.tools.geography import ensure_minimal_bbox_span
from eomas_assistant.tools.image_processing import (
    compute_roi_statistics as _compute_roi_statistics,
)
from eomas_assistant.tools.wmts_retrieval import (
    construct_tiled_eo_image_with_wmts_metadata,
    request_available_wmts_layers,
)

SYSTEM_PROMPT = (
    "You are the earth observation (EO) imagery agent as part of an EO assistant. "
    "Your tasks are (in this order):\n"
    "* Based on the chat history, set the maximum cloud coverage. "
    "If the user has not specified otherwise, the maximum acceptable cloud coverage should be 5%.\n"
    # TODO: "most recent" in the given time range is a temporary heuristic
    "* Choose a date for the imagery, preferring the most recent acceptably cloud-free acquisition. "
    "Therefore, you should check the available STAC dates and call the compute_cloud_coverage_for_date tool "
    "to find a suitable date.\n"
    "* Find out which WMTS layer the user wants to see and to select the WMTS layer with the found date using the provided tools. "
    "Default to the TRUE_COLOR layer if the user does not declare a preference.\n"
    "Before selecting a WMTS layer, do not forget to search for a suitably cloud-free date.\n"
    "When asked about statistics of a specific STAC asset, "
    "take into account that the STAC assets have different keys than the WMTS layer names. "
    "You can use the list_available_stac_asset_keys tool to find out which STAC asset keys are available for the current geo/time query. "
    "There is also not a one-to-one mapping between all WMTS layers and STAC assets, but many correspond directly, "
    "and for the others you should use your best judgment to find the most appropriate WMTS layer to use as illustration "
    "while computing statistics for the requested STAC asset key. "
)


# cached, since it would only change if the Sentinel Hub WMTS instance changes,
# which is not expected during a single session:
@tool
@lru_cache(maxsize=1)
def list_available_wmts_layers() -> dict[str, str]:
    """List available WMTS layers from the configured Sentinel Hub instance."""
    return request_available_wmts_layers()


@tool
def count_available_stac_assets(
    runtime: ToolRuntime[None, AgentState],
) -> int:
    """Count the number of available STAC assets for the current geo/time query."""

    asset_catalog = runtime.state.asset_catalog
    if asset_catalog is None:
        return 0
    return len(asset_catalog.available_stac_items)


@tool
def get_selected_wmts_layer(
    runtime: ToolRuntime[None, AgentState],
) -> str:
    """Get the ID of the currently selected WMTS layer.

    CURRENTLY UNUSED, since the LLM is prompted with the current selection.
    """
    if not runtime.state.data_request:
        return ""
    selected_layer = runtime.state.data_request.wmts_layer
    return selected_layer or "(not set)"


@tool
def set_selected_wmts_layer(
    runtime: ToolRuntime[None, AgentState],
    selected_layer_id: str,
    acquisition_date: str,
    layer_selection_reason: str,
    date_selection_reason: str,
) -> Command:
    """Select a WMTS layer to be shown to the user by ID."""

    data_request = runtime.state.data_request or DataRequest()

    data_request.wmts_layer = selected_layer_id
    data_request.selection_reasons.append(layer_selection_reason)

    data_request.acquired_at = datetime.fromisoformat(acquisition_date)
    data_request.selection_reasons.append(date_selection_reason)

    update: dict = dict(
        data_request=data_request,
        response=dict(
            reasoning_trace=[
                f"WMTS layer set to {selected_layer_id} ({layer_selection_reason})",
                f"Chose acquisition date {acquisition_date} ({date_selection_reason})",
            ]
        ),
    )

    response = runtime.state.response
    if runtime.state.geo_location is not None and response is not None:
        map_response_items = [
            item for item in response.items if isinstance(item, MapResponseItem)
        ]
        if map_response_items:  # (expected to be exactly one)
            overlay = construct_tiled_eo_image_with_wmts_metadata(
                geo_location=runtime.state.geo_location,
                data_request=data_request,
            )
            if overlay is not None:
                map_response_items[0].eo_images.append(overlay)

    update["messages"] = [
        ToolMessage(
            f"Successfully set the WMTS layer to {selected_layer_id}",
            tool_call_id=runtime.tool_call_id,
        )
    ]

    return Command(update=update)


@tool
def set_maximum_cloud_coverage(
    runtime: ToolRuntime[None, AgentState],
    max_cloud_cover_in_percent: float,
) -> Command:
    """Set the maximum cloud coverage (in percent, range 0..100) which the user allows."""

    if not (0.0 <= max_cloud_cover_in_percent <= 100.0):
        raise ValueError(
            f"max_cloud_cover must be between 0 and 100, got {max_cloud_cover_in_percent}"
        )

    data_request = runtime.state.data_request or DataRequest()
    data_request.max_cloud_cover = max_cloud_cover_in_percent

    update: dict = dict(
        data_request=data_request,
        response=dict(
            reasoning_trace=[
                f"Maximum cloud coverage set to {max_cloud_cover_in_percent}%",
            ]
        ),
        messages=[
            ToolMessage(
                f"Successfully set the maximum cloud coverage to {max_cloud_cover_in_percent}%",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    )

    return Command(update=update)


@tool
def list_available_imagery_dates(
    runtime: ToolRuntime[None, AgentState],
) -> list[str]:
    """List available acquisition dates for the current geo/time query."""

    asset_catalog = runtime.state.asset_catalog
    if asset_catalog is None:
        raise RuntimeError("Asset catalog is not available in the current state.")

    return sorted(asset_catalog.available_stac_items_by_date.keys())


@tool
def compute_roi_cloud_coverage_for_date(
    runtime: ToolRuntime[None, AgentState],
    acquisition_date: str,
) -> Command:
    """Compute the average cloud coverage over the exact geographic region of
    interest for a given acquisition date (in ISO 8601 YYYY-MM-DD format).
    """

    asset_catalog = runtime.state.asset_catalog
    if asset_catalog is None:
        raise RuntimeError("Asset catalog is not available in the current state.")

    if runtime.state.geo_location is None:
        raise RuntimeError("GeoLocation is not available in the current state.")

    roi = runtime.state.geo_location.geojson
    if roi is None:
        raise RuntimeError(
            "GeoLocation does not contain a GeoJSON in the current state."
        )

    date_items = asset_catalog.available_stac_items_by_date.get(acquisition_date)
    if not date_items:
        raise KeyError(f"No STAC items found for acquisition date {acquisition_date}.")

    downloader = EOImageDownloader()
    cld_prob, cld_crs, cld_transform = downloader.download_and_merge_cloud_probability(
        stac_items=date_items
    )

    roi_mask = rasterize_geojson(
        roi, cld_crs, cld_transform, cld_prob.shape[-2:], invert=False
    )
    masked_cld_prob = np.ma.masked_array(cld_prob, mask=roi_mask)

    result = float(masked_cld_prob.mean())
    return Command(
        update=dict(
            response=dict(roi_cloud_cover=(acquisition_date, result)),
            messages=[
                ToolMessage(
                    str(result),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        )
    )


@tool
def list_available_stac_asset_keys() -> dict:
    """Describe all available STAC asset keys in the sentinel-2-l2a collection."""

    # TODO: this is a RAG-like method that returns the exact data available through the STAC API;
    # it should be replaced with a dynamic download
    return {
        "id": "sentinel-2-l2a",
        "title": "Sentinel-2 Level-2A",
        "assets": [
            {"key": "AOT_10m", "title": "Aerosol optical thickness (AOT) - 10m"},
            {"key": "B01_20m", "title": "Coastal aerosol (band 1) - 20m"},
            {"key": "B02_10m", "title": "Blue (band 2) - 10m"},
            {"key": "B03_10m", "title": "Green (band 3) - 10m"},
            {"key": "B04_10m", "title": "Red (band 4) - 10m"},
            {"key": "B05_20m", "title": "Red edge 1 (band 5) - 20m"},
            {"key": "B06_20m", "title": "Red edge 2 (band 6) - 20m"},
            {"key": "B07_20m", "title": "Red edge 3 (band 7) - 20m"},
            {"key": "B08_10m", "title": "NIR 1 (band 8) - 10m"},
            {"key": "B11_20m", "title": "SWIR 1 (band 11) - 20m"},
            {"key": "B12_20m", "title": "SWIR 2 (band 12) - 20m"},
            {"key": "B8A_20m", "title": "NIR 2 (band 8A) - 20m"},
            {"key": "CLD_20m", "title": "Cloud probability (CLD) - 20m"},
            {"key": "SCL_20m", "title": "Scene classification map (SCL) - 20m"},
            {"key": "SNW_20m", "title": "Snow probability (SNW) - 20m"},
            {"key": "TCI_10m", "title": "True color image"},
            {"key": "WVP_10m", "title": "Water vapour (WVP) - 10m"},
        ],
        "description": "The Sentinel-2 Level-2A Collection 1 product provides orthorectified Surface Reflectance (Bottom-Of-Atmosphere: BOA), with sub-pixel multispectral and multitemporal registration accuracy. Scene Classification (including Clouds and Cloud Shadows), AOT (Aerosol Optical Thickness) and WV (Water Vapour) maps are included in the product.",
    }


@tool
def compute_roi_statistics(
    runtime: ToolRuntime[None, AgentState],
    acquisition_date: str,
    stac_asset_key_or_index_name: str,
) -> dict[str, float]:
    """Compute statistics (mean, min, max, stddev) of the pixel values of the
    specified asset type inside the region of interest for a given acquisition
    date (in ISO 8601 YYYY-MM-DD format).
    """
    asset_catalog = runtime.state.asset_catalog
    if asset_catalog is None:
        raise RuntimeError("Asset catalog is not available in the current state.")

    if runtime.state.geo_location is None:
        raise RuntimeError("GeoLocation is not available in the current state.")

    roi = runtime.state.geo_location.geojson
    if roi is None:
        raise RuntimeError(
            "GeoLocation does not contain a GeoJSON in the current state."
        )

    date_items = asset_catalog.available_stac_items_by_date.get(acquisition_date)
    if not date_items:
        raise KeyError(f"No STAC items found for acquisition date {acquisition_date}.")

    return _compute_roi_statistics(
        stac_items=date_items,
        stac_asset_key_or_index_name=stac_asset_key_or_index_name,
        roi_geojson=roi,
    )


EO_IMAGERY_TOOLS = [
    list_available_wmts_layers,
    count_available_stac_assets,
    set_maximum_cloud_coverage,
    set_selected_wmts_layer,
    list_available_imagery_dates,
    compute_roi_cloud_coverage_for_date,
    list_available_stac_asset_keys,
    compute_roi_statistics,
]


class EOImageryAgent:
    """Discover available STAC asset keys for the current geo/time query."""

    def __init__(self, llm_client: BaseChatModel) -> None:
        self._llm_client = llm_client.bind_tools(EO_IMAGERY_TOOLS)

    def __call__(self, state: AgentState) -> dict:
        """Graph node: discover available EO assets for the resolved geography/time."""

        geo_location = state.geo_location
        if geo_location is None:
            return dict(asset_catalog=None)

        result = {}

        if not state.asset_catalog:
            # FIXME: convert to tool as well? (does not have to be done in every LLM run)
            asset_catalog = self.find_assets(geo_location=geo_location)
            reasoning_trace = [
                "Asset discovery inspected "
                f"{len(asset_catalog.available_stac_items)} STAC item(s) "
                f"from {len(asset_catalog.available_stac_items_by_date)} dates and found "
                f"{len(asset_catalog.available_asset_keys)} available asset key(s)."
            ]
            result.update(
                dict(
                    asset_catalog=asset_catalog,
                    response=dict(
                        reasoning_trace=reasoning_trace,
                        # FIXME: remove this table once we have something better
                        available_stac_images=[
                            item.model_dump()
                            for item in asset_catalog.available_stac_images
                        ],
                    ),
                )
            )

        ai_message = self.run_llm(
            data_request=state.data_request,
            messages=state.messages,
        )

        return dict(messages=[ai_message], **result)

    def run_llm(
        self,
        data_request: DataRequest | None,
        messages: Sequence[AnyMessage],
    ) -> AIMessage:
        """Extract normalized EO retrieval parameters from user query."""

        system_prompt = SYSTEM_PROMPT
        supplemental_prompt = (
            f"The WMTS layer currently selected for display is {repr(data_request.wmts_layer)}."
            if data_request
            else "No WMTS layer is currently selected; the set_selected_wmts_layer tool needs to be called."
        )
        # supplemental_prompt = "Extract EO retrieval parameters from the latest user message in the conversation above."
        llm_input = llm_helper.build_model_input_messages(
            system_prompt=system_prompt,
            messages=messages,
            supplemental_prompt=supplemental_prompt,
        )
        return self._llm_client.invoke(input=llm_input)

    def find_assets(
        self,
        geo_location: GeoLocation,
        max_cloud_cover: float | None = None,
        max_items: int = 5000,
    ) -> AssetCatalog:
        if geo_location.bbox_wgs84_lat_lon is None:
            return AssetCatalog(
                available_stac_items=[],
                available_asset_keys=[],
            )

        try:
            search = find_sentinel2_assets_in_time_range(
                bbox_wgs84=ensure_minimal_bbox_span(geo_location.bbox_wgs84_lat_lon),
                datetime_range=geo_location.time_range,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
            )
            items = list(search.items())
        except Exception as exc:
            print(f"Asset catalog lookup failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return AssetCatalog(
                available_stac_items=[],
                available_asset_keys=[],
            )

        result = AssetCatalog(available_stac_items=items)

        available_keys: set[str] = set()
        for item in items:
            available_keys.update(item.assets.keys())
            result.available_stac_items_by_date[format_acquisition_date(item)].append(
                item
            )
        result.available_asset_keys = list(available_keys)

        result.available_stac_images = [
            StacImageCatalogEntry(
                acquisition_date=acquisition_date,
                stac_cc=float(
                    np.mean(
                        [
                            cast(float, item.properties.get("eo:cloud_cover"))
                            for item in date_items
                        ]
                    )
                ),
            )
            for acquisition_date, date_items in sorted(
                result.available_stac_items_by_date.items()
            )
        ]

        return result
