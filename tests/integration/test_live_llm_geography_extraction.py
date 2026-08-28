# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from datetime import date
import re

import pytest
from langchain_core.messages import HumanMessage

from eomas_assistant.config.settings import AppSettings
from eomas_assistant.graph.workflow import AgentWorkflow
from eomas_assistant.llm import create_llm_client
from eomas_assistant.models.response_models import MapResponseItem, TextResponseItem
from eomas_assistant.nodes.geography import GeographyAgent
from eomas_assistant.tools.geocoding import Geocoding


@pytest.mark.integration
@pytest.mark.parametrize(
    ("user_query", "expected_keyword", "expected_start_date", "expected_end_date"),
    [
        (
            "Show me Bremen in February 2017",
            "bremen",
            date(2017, 2, 1),
            date(2017, 2, 28),
        ),
        (
            "Show me Oldenburg on 15.12.2022",
            "oldenburg",
            date(2022, 12, 15),
            date(2022, 12, 15),
        ),
        (
            "Show me Berlin, Germany from 2018 to 2019",
            "berlin",
            date(2018, 1, 1),
            date(2019, 12, 31),
        ),
        (
            "Show me Berlin, Germany in 2022",
            "berlin",
            date(2022, 1, 1),
            date(2022, 12, 31),
        ),
    ],
)
def test_live_llm_extracts_location_bbox_and_time_range(
    user_query: str,
    expected_keyword: str,
    expected_start_date: date,
    expected_end_date: date,
) -> None:
    settings = AppSettings()
    llm_client = create_llm_client(settings)
    geocoding_tool = Geocoding(
        base_url=settings.nominatim_base_url,
        user_agent=settings.nominatim_user_agent,
        timeout_seconds=settings.request_timeout_seconds,
    )
    agent = GeographyAgent(geocoding_tool=geocoding_tool, llm_client=llm_client)

    _, geo_location = agent.run(messages=[HumanMessage(content=user_query)])

    assert geo_location is not None

    location_text = f"{geo_location.name} {geo_location.display_name}".lower()
    assert expected_keyword in location_text

    bbox = geo_location.bbox_wgs84_lat_lon
    assert bbox is not None
    min_lon, min_lat, max_lon, max_lat = bbox.as_lon_lat_tuple()
    assert -180.0 <= min_lon <= 180.0
    assert -180.0 <= max_lon <= 180.0
    assert -90.0 <= min_lat <= 90.0
    assert -90.0 <= max_lat <= 90.0
    assert min_lon <= max_lon
    assert min_lat <= max_lat

    assert geo_location.time_range is not None
    assert geo_location.time_range.start_timepoint is not None
    assert geo_location.time_range.end_timepoint is not None
    start_timepoint = geo_location.time_range.start_timepoint
    end_timepoint = geo_location.time_range.end_timepoint
    assert start_timepoint.date() == expected_start_date
    assert end_timepoint.date() == expected_end_date


@pytest.mark.integration
def test_live_llm_workflow_extracts_bremen_ndvi_cloud_cover_and_returns_map() -> None:
    settings = AppSettings()
    llm_client = create_llm_client(settings)
    workflow = AgentWorkflow(settings=settings, llm_client=llm_client)

    state = workflow._graph.invoke(
        workflow._build_initial_state(
            [
                HumanMessage(
                    content="Show me Bremen in January 2023 with maximum 40% cloud cover in NDVI"
                )
            ],
        )
    )

    geo_location = state.get("geo_location")
    assert geo_location is not None
    location_text = f"{geo_location.name} {geo_location.display_name}".lower()
    assert "bremen" in location_text

    data_request = state.get("data_request")
    assert data_request is not None
    assert data_request.wmts_layer == "NDVI"
    assert data_request.max_cloud_cover == pytest.approx(40.0)

    response = state.get("response")
    assert response is not None
    map_items = [item for item in response.items if isinstance(item, MapResponseItem)]
    assert map_items

    eo_images = [image for item in map_items for image in item.eo_images]
    assert eo_images

    wmts_images = [image for image in eo_images if image.tiles_url_template]
    assert wmts_images
    assert all(image.asset_key == "NDVI" for image in wmts_images)


@pytest.mark.integration
def test_live_vllm_workflow_computes_mean_ndvi_of_oldenburg() -> None:
    settings = AppSettings().model_copy(update={"llm_provider": "openai_api"})
    llm_client = create_llm_client(settings)
    workflow = AgentWorkflow(settings=settings, llm_client=llm_client)

    state = workflow._graph.invoke(
        workflow._build_initial_state(
            [HumanMessage(content="What was the mean NDVI value of Oldenburg in last March?")],
        )
    )

    geo_location = state.get("geo_location")
    assert geo_location is not None
    location_text = f"{geo_location.name} {geo_location.display_name}".lower()
    assert "oldenburg" in location_text

    response = state.get("response")
    assert response is not None

    response_texts = [
        item.content for item in response.items if isinstance(item, TextResponseItem)
    ]
    assert response_texts
    assert re.search(r"(mean|average).*\bNDVI\b.* 0\.[0-9]", " ".join(response_texts), re.IGNORECASE)


@pytest.mark.skip(reason="Implementation of longitudinal analysis not finished yet")
@pytest.mark.integration
def test_live_vllm_workflow_extracts_horn_lehe_and_downloads_yearly_ndvi_images() -> None:
    settings = AppSettings().model_copy(update={"llm_provider": "openai_api"})
    llm_client = create_llm_client(settings)
    workflow = AgentWorkflow(settings=settings, llm_client=llm_client)

    state = workflow._graph.invoke(
        workflow._build_initial_state(
            [
                HumanMessage(
                    content="How has the mean NDVI in Horn-Lehe changed over the last 10 years?"
                )
            ],
        )
    )

    geo_location = state.get("geo_location")
    assert geo_location is not None
    location_text = f"{geo_location.name} {geo_location.display_name}".lower()
    assert "horn-lehe" in location_text
    assert "bremen" in location_text

    response = state.get("response")
    assert response is not None

    stac_images = response.metadata.get("stac_images")
    assert isinstance(stac_images, list)
    assert len(stac_images) >= 10

    years = {
        entry["acquired_at"].year
        for entry in stac_images
        if isinstance(entry, dict) and hasattr(entry.get("acquired_at"), "year")
    }
    assert len(years) >= 10

    # TODO: Check if images get analysed and results get presented properly...
