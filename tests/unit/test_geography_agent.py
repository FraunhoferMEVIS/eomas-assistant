# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from datetime import UTC, datetime, timedelta

from langchain_core.messages import HumanMessage

from eomas_assistant.agents.geography import DEFAULT_ZOOM_LEVEL, GeographyAgent, GeographyExtraction
from eomas_assistant.models.response_models import MapResponseItem
from eomas_assistant.models.schemas import (
    BoundingBox,
    GeographySummary,
    GeoLocation,
    TimeRange,
)


def test_geography_agent_extracts_location_bbox_and_time_range(mocker) -> None:
    user_query = "Show me Bremen in January 2020"

    bremen_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [8.70, 53.00],
                            [8.90, 53.00],
                            [8.90, 53.20],
                            [8.70, 53.20],
                            [8.70, 53.00],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }

    location = GeoLocation(
        query="Bremen",
        name="Bremen",
        latitude=53.0793,
        longitude=8.8017,
        display_name="Bremen, Germany",
        geojson=bremen_geojson,
    )

    geocoding_tool = mocker.Mock()
    geocoding_tool.geocode.return_value = location

    time_range = TimeRange(
        start_timepoint=datetime(2020, 1, 1, tzinfo=UTC),
        end_timepoint=datetime(2020, 1, 31, 23, 59, 59, tzinfo=UTC),
    )

    mocker.patch(
        "eomas_assistant.agents.geography.LLMHelper.call_llm_with_schema",
        side_effect=[
            GeographyExtraction(
                candidates=[user_query, "Bremen, Germany"],
                start_timepoint=time_range.start_timepoint,
                end_timepoint=time_range.end_timepoint,
            ),
            GeographySummary(summary="Bremen summary."),
        ],
    )

    agent = GeographyAgent(geocoding_tool=geocoding_tool, llm_client=mocker.Mock())

    _, geo_location = agent.run(messages=[HumanMessage(content=user_query)])

    assert geo_location is not None
    assert geo_location.display_name == "Bremen, Germany"
    assert geo_location.time_range == time_range
    assert geo_location.bbox_wgs84_lat_lon == BoundingBox(
        min_latitude=53.0,
        min_longitude=8.7,
        max_latitude=53.2,
        max_longitude=8.9,
    )

    geocoding_tool.geocode.assert_called_once_with(user_query)


def test_geography_agent_sets_zoom_from_bbox(mocker) -> None:
    user_query = "Show me Bremen"
    location = GeoLocation(
        query="Bremen",
        name="Bremen",
        latitude=53.0793,
        longitude=8.8017,
        display_name="Bremen, Germany",
        geojson={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [8.70, 53.00],
                                [8.90, 53.00],
                                [8.90, 53.20],
                                [8.70, 53.20],
                                [8.70, 53.00],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        },
    )

    geocoding_tool = mocker.Mock()
    geocoding_tool.geocode.return_value = location
    mocker.patch(
        "eomas_assistant.agents.geography.LLMHelper.call_llm_with_schema",
        side_effect=[
            GeographyExtraction(
                candidates=[user_query],
                start_timepoint=None,
                end_timepoint=None,
            ),
            GeographySummary(summary="Bremen summary."),
        ],
    )

    agent = GeographyAgent(geocoding_tool=geocoding_tool, llm_client=mocker.Mock())
    response, _ = agent.run(messages=[HumanMessage(content=user_query)])

    map_response = next(item for item in response.items if isinstance(item, MapResponseItem))
    assert map_response.zoom == 9


def test_geography_agent_uses_point_zoom_without_area_geometry(mocker) -> None:
    user_query = "Show me Bremen"
    location = GeoLocation(
        query="Bremen",
        name="Bremen",
        latitude=53.0793,
        longitude=8.8017,
        display_name="Bremen, Germany",
        geojson=None,
    )

    geocoding_tool = mocker.Mock()
    geocoding_tool.geocode.return_value = location
    mocker.patch(
        "eomas_assistant.agents.geography.LLMHelper.call_llm_with_schema",
        side_effect=[
            GeographyExtraction(
                candidates=[user_query],
                start_timepoint=None,
                end_timepoint=None,
            ),
            GeographySummary(summary="Bremen summary."),
        ],
    )

    agent = GeographyAgent(geocoding_tool=geocoding_tool, llm_client=mocker.Mock())
    response, _ = agent.run(messages=[HumanMessage(content=user_query)])

    map_response = next(item for item in response.items if isinstance(item, MapResponseItem))
    assert map_response.zoom == DEFAULT_ZOOM_LEVEL
    assert 10 < DEFAULT_ZOOM_LEVEL <= 12


def test_geography_agent_reuses_prior_location_when_location_missing(mocker) -> None:
    user_query = "And in January 2020?"
    previous_time_range = TimeRange(
        start_timepoint=datetime(2019, 1, 1, tzinfo=UTC),
        end_timepoint=datetime(2019, 12, 31, 23, 59, 59, tzinfo=UTC),
    )
    updated_time_range = TimeRange(
        start_timepoint=datetime(2020, 1, 1, tzinfo=UTC),
        end_timepoint=datetime(2020, 1, 31, 23, 59, 59, tzinfo=UTC),
    )
    prior_geo_location = GeoLocation(
        query="Bremen",
        name="Bremen",
        latitude=53.0793,
        longitude=8.8017,
        display_name="Bremen, Germany",
        time_range=previous_time_range,
    )

    geocoding_tool = mocker.Mock()
    geocoding_tool.geocode.return_value = None

    mocker.patch(
        "eomas_assistant.agents.geography.LLMHelper.call_llm_with_schema",
        side_effect=[
            GeographyExtraction(
                candidates=[],
                start_timepoint=updated_time_range.start_timepoint,
                end_timepoint=updated_time_range.end_timepoint,
            ),
            GeographySummary(summary="Bremen summary."),
        ],
    )

    agent = GeographyAgent(geocoding_tool=geocoding_tool, llm_client=mocker.Mock())
    _, geo_location = agent.run(
        messages=[HumanMessage(content=user_query)],
        prior_geo_location=prior_geo_location,
    )

    assert geo_location is not None
    assert geo_location.display_name == "Bremen, Germany"
    assert geo_location.time_range == updated_time_range
    geocoding_tool.geocode.assert_not_called()


def test_geography_agent_defaults_to_recent_time_range_when_time_missing(mocker) -> None:
    user_query = "Show me Hamburg"
    prior_time_range = TimeRange(
        start_timepoint=datetime(2020, 1, 1, tzinfo=UTC),
        end_timepoint=datetime(2020, 1, 31, 23, 59, 59, tzinfo=UTC),
    )
    prior_geo_location = GeoLocation(
        query="Bremen",
        name="Bremen",
        latitude=53.0793,
        longitude=8.8017,
        display_name="Bremen, Germany",
        time_range=prior_time_range,
    )
    new_location = GeoLocation(
        query="Hamburg",
        name="Hamburg",
        latitude=53.5511,
        longitude=9.9937,
        display_name="Hamburg, Germany",
    )

    geocoding_tool = mocker.Mock()
    geocoding_tool.geocode.return_value = new_location

    mocker.patch(
        "eomas_assistant.agents.geography.LLMHelper.call_llm_with_schema",
        side_effect=[
            GeographyExtraction(
                candidates=["Hamburg"],
                start_timepoint=None,
                end_timepoint=None,
            ),
            GeographySummary(summary="Hamburg summary."),
        ],
    )

    agent = GeographyAgent(geocoding_tool=geocoding_tool, llm_client=mocker.Mock())
    _, geo_location = agent.run(
        messages=[HumanMessage(content=user_query)],
        prior_geo_location=prior_geo_location,
    )

    assert geo_location is not None
    assert geo_location.display_name == "Hamburg, Germany"
    assert geo_location.time_range is not None
    # previously, we defaulted to the prior time range, but that behavior is no longer hardcoded
    assert geo_location.time_range != prior_time_range

    now = datetime.now(UTC)
    assert geo_location.time_range.start_timepoint is not None
    assert geo_location.time_range.start_timepoint >= now - timedelta(days=181)
    assert geo_location.time_range.start_timepoint <= now - timedelta(days=179)
    assert geo_location.time_range.end_timepoint is not None
    assert geo_location.time_range.end_timepoint <= now + timedelta(seconds=1)
    assert geo_location.time_range.end_timepoint >= now - timedelta(seconds=1)
