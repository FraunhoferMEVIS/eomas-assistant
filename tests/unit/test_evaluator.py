# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from langchain_core.messages import AIMessage, HumanMessage

from eomas_assistant.agents.evaluator import EvaluatorAgent
from eomas_assistant.models.response_models import AgentResponse, MapResponseItem, TextResponseItem
from eomas_assistant.models.schemas import BoundingBox, EvaluationResult, TiledEOImage


def test_evaluator_includes_wmts_and_stac_band_metadata_in_prompt(mocker) -> None:
    captured: dict[str, object] = {}

    def _fake_call_llm_with_schema(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["messages"] = kwargs["messages"]
        captured["supplemental_prompt"] = kwargs["supplemental_prompt"]
        return EvaluationResult(
            approved=True,
            retryable=False,
            score=0.9,
            critique="Looks correct.",
            replanning_instructions="",
        )

    mocker.patch(
        "eomas_assistant.agents.evaluator.llm_helper.call_llm_with_schema",
        side_effect=_fake_call_llm_with_schema,
    )

    response = AgentResponse(
        agent_name="geography",
        items=[
            TextResponseItem(content="Showing Bremen in the requested red band."),
            MapResponseItem(
                title="Bremen",
                center_latitude=53.0793,
                center_longitude=8.8017,
                zoom=9,
                eo_images=[
                    TiledEOImage(
                        bbox_wgs84_lat_lon=BoundingBox(
                            min_latitude=53.0,
                            min_longitude=8.7,
                            max_latitude=53.2,
                            max_longitude=8.9,
                        ),
                        asset_key="RED",
                        asset_title="RED",
                        tiles_url_template="https://example.com/wmts/{z}/{x}/{y}.png",
                    )
                ],
            ),
        ],
        metadata={
            "route": "geography",
            "stac_images": [
                {
                    "bbox_wgs84_lat_lon": {
                        "min_latitude": 53.0,
                        "min_longitude": 8.7,
                        "max_latitude": 53.2,
                        "max_longitude": 8.9,
                    },
                    "asset_key": "B04_10m",
                    "asset_title": "B04 red",
                    "max_cloud_cover": 20.0,
                    "source_path": "C:/tmp/b04.png",
                }
            ],
        },
    )

    agent = EvaluatorAgent(llm_client=mocker.Mock())
    result = agent.run(
        messages=[HumanMessage(content="Show me Bremen in the red band")],
        response=response,
    )

    assert result.approved is True
    supplemental_prompt = str(captured["supplemental_prompt"])
    assert "rendered_map_eo_image" in supplemental_prompt
    assert '"asset_key":"RED"' in supplemental_prompt
    assert "stac_downloaded_images: [{" in supplemental_prompt
    assert '"asset_key":"B04_10m"' in supplemental_prompt
    captured_messages = captured.get("messages")
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 1
    captured_system_prompt = captured.get("system_prompt")
    assert isinstance(captured_system_prompt, str)
    assert (
        "Decide layer compatibility from the raw asset_key and asset_title values."
        in captured_system_prompt
    )
    assert (
        "infer common EO aliases and provider naming conventions yourself" in captured_system_prompt
    )


def test_evaluator_prompt_prioritizes_latest_request_over_older_context(mocker) -> None:
    captured: dict[str, object] = {}

    def _fake_call_llm_with_schema(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["messages"] = kwargs["messages"]
        captured["supplemental_prompt"] = kwargs["supplemental_prompt"]
        return EvaluationResult(
            approved=True,
            retryable=False,
            score=0.9,
            critique="Looks correct.",
            replanning_instructions="",
        )

    mocker.patch(
        "eomas_assistant.agents.evaluator.llm_helper.call_llm_with_schema",
        side_effect=_fake_call_llm_with_schema,
    )

    response = AgentResponse(
        agent_name="geography",
        items=[
            TextResponseItem(content="Showing Bremen in true color."),
            MapResponseItem(
                title="Bremen",
                center_latitude=53.0793,
                center_longitude=8.8017,
                zoom=9,
                eo_images=[
                    TiledEOImage(
                        bbox_wgs84_lat_lon=BoundingBox(
                            min_latitude=53.0,
                            min_longitude=8.7,
                            max_latitude=53.2,
                            max_longitude=8.9,
                        ),
                        asset_key="TRUE_COLOR",
                        asset_title="TRUE_COLOR",
                        tiles_url_template="https://example.com/wmts/{z}/{x}/{y}.png",
                    )
                ],
            ),
        ],
        metadata={
            "route": "geography",
            "stac_images": [],
        },
    )

    agent = EvaluatorAgent(llm_client=mocker.Mock())
    result = agent.run(
        messages=[
            HumanMessage(content="Show me Bremen in NDVI."),
            AIMessage(content="Here is Bremen in NDVI."),
            HumanMessage(content="Now show the same area in true color"),
        ],
        response=response,
    )

    assert result.approved is True
    captured_messages = captured["messages"]
    assert isinstance(captured_messages, list)
    assert captured_messages[0].content == "Show me Bremen in NDVI."
    assert captured_messages[2].content == "Now show the same area in true color"
