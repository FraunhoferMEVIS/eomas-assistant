# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import folium
import httpx
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

from eomas_assistant.config.settings import get_settings
from eomas_assistant.models.response_models import (
    AgentResponse,
    ErrorResponseItem,
    MapResponseItem,
    TextResponseItem,
)
from eomas_assistant.models.schemas import LocalEOImage, TiledEOImage


def render_agent_response(response: AgentResponse) -> None:
    """Render typed agent outputs in Streamlit chat context."""
    stac_images = _extract_stac_images(response)

    for output_index, output in enumerate(response.items):
        if isinstance(output, TextResponseItem):
            st.markdown(output.content)
            continue

        if isinstance(output, MapResponseItem):
            st.caption(output.title)
            _render_map(
                output,
                output.eo_images[0] if output.eo_images else None,
                stac_images,
                component_key=_build_map_component_key(response, output, output_index),
            )
            continue

        if isinstance(output, ErrorResponseItem):
            st.error(output.message)

    _render_available_stac_images_table(response)
    _render_reasoning_trace(response)
    _render_evaluation_summary(response)


def _render_available_stac_images_table(response: AgentResponse) -> None:
    """Render discovered STAC scene availability as a compact table."""

    value = response.metadata.get("available_stac_images")
    if not isinstance(value, list) or not value:
        return

    table_rows, chart_points = _build_stac_table_rows_and_chart_points(value)

    if not table_rows:
        return

    with st.expander("Available imagery in query range", expanded=False):
        # st.table(table_rows)
        if chart_points:
            st.caption("Cloud cover by acquisition date")
            st.vega_lite_chart(
                _build_cloud_cover_vega_spec(chart_points),
                width="stretch",
            )


def _build_cloud_cover_vega_spec(
    chart_points: list[dict[str, float | datetime]],
) -> dict[str, Any]:
    """Build a Vega-Lite line chart spec for cloud cover over acquisition time."""

    values = [
        {
            "acquisition_date": point["acquisition_date"].isoformat(),  # type: ignore
            "stac_cc": point["stac_cc"],
            "roi_cc": point.get("roi_cc"),
        }
        for point in sorted(
            chart_points,
            key=lambda point: point["acquisition_date"],
        )
        if isinstance(point.get("acquisition_date"), datetime)
        and isinstance(point.get("stac_cc"), int | float)
    ]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "transform": [
            {"fold": ["stac_cc", "roi_cc"], "as": ["series", "cloud_cover"]},
            {"filter": "isValid(datum.cloud_cover)"},
            {
                "calculate": (
                    "datum.series == 'stac_cc' ? "
                    "'Cloud cover (% of STAC frame)' : "
                    "'Cloud cover (% of ROI)'"
                ),
                "as": "series_label",
            },
        ],
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {
                "field": "acquisition_date",
                "type": "temporal",
                "title": "Acquisition date",
            },
            "y": {
                "field": "cloud_cover",
                "type": "quantitative",
                "title": "Cloud cover (%)",
                "scale": {"domain": [0, 100]},
            },
            "color": {
                "field": "series_label",
                "type": "nominal",
                "title": "Series",
            },
            "tooltip": [
                {
                    "field": "acquisition_date",
                    "type": "temporal",
                    "title": "Acquisition date",
                },
                {
                    "field": "series_label",
                    "type": "nominal",
                    "title": "Series",
                },
                {
                    "field": "cloud_cover",
                    "type": "quantitative",
                    "title": "Cloud cover (%)",
                    "format": ".1f",
                },
            ],
        },
    }


def _build_stac_table_rows_and_chart_points(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, float | datetime]]]:
    """Build display rows and chart points from STAC image metadata entries."""

    table_rows: list[dict[str, str]] = []
    chart_points: list[dict[str, float | datetime]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        acquisition_date = entry.get("acquisition_date", "unknown")
        stac_cc_raw = entry.get("stac_cc")
        roi_cc = entry.get("roi_cc")

        if isinstance(stac_cc_raw, int | float):
            stac_cc = float(stac_cc_raw)
            stac_cc_text = f"{stac_cc:.1f}%"
        else:
            stac_cc = None
            stac_cc_text = "unknown"

        table_rows.append(
            {
                "acquisition_date": str(acquisition_date),
                "stac_cc": stac_cc_text,
            }
        )

        parsed_acquisition_date = _parse_stac_acquisition_datetime(acquisition_date)
        if parsed_acquisition_date is None or stac_cc is None:
            continue

        point = {
            "acquisition_date": parsed_acquisition_date,
            "stac_cc": stac_cc,
        }
        if isinstance(roi_cc, int | float):
            point["roi_cc"] = float(roi_cc)
        chart_points.append(point)

    return table_rows, chart_points


def _parse_stac_acquisition_datetime(value: Any) -> datetime | None:
    """Parse ISO-8601 timestamps used by STAC metadata; return None for unknown values."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() == "unknown"
    ):
        return None

    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _render_reasoning_trace(response: AgentResponse) -> None:
    """Render a concise, user-facing reasoning trace when available."""
    trace = response.metadata.get("reasoning_trace")
    if not isinstance(trace, list) or not trace:
        return

    with st.expander("Reasoning process", expanded=False):
        for step in trace:
            if isinstance(step, str) and step.strip():
                st.markdown(f"- {step}")


def _render_evaluation_summary(response: AgentResponse) -> None:
    """Render evaluation and retry details for the current workflow result."""

    lines = _build_evaluation_summary_lines(response)
    if not lines:
        return

    with st.expander("Evaluation loop", expanded=False):
        for line in lines:
            st.markdown(f"- {line}")


def _build_evaluation_summary_lines(response: AgentResponse) -> list[str]:
    """Build compact evaluation and retry summary lines from response metadata."""

    metadata = response.metadata
    lines: list[str] = []

    attempt_count = metadata.get("attempt_count")
    max_attempts = metadata.get("max_attempts")
    if isinstance(attempt_count, int) and isinstance(max_attempts, int):
        lines.append(f"Attempts: {attempt_count}/{max_attempts}")

    evaluation_status = metadata.get("evaluation_status")
    if isinstance(evaluation_status, str) and evaluation_status.strip():
        lines.append(f"Final evaluation status: {evaluation_status}")

    evaluation = metadata.get("evaluation")
    if isinstance(evaluation, dict):
        score = evaluation.get("score")
        if isinstance(score, int | float):
            lines.append(f"Final evaluation score: {float(score):.2f}")

        critique = evaluation.get("critique")
        if isinstance(critique, str) and critique.strip():
            lines.append(f"Critique: {critique.strip()}")

        replanning = evaluation.get("replanning_instructions")
        if isinstance(replanning, str) and replanning.strip():
            lines.append(f"Retry guidance: {replanning.strip()}")

    return lines


def _extract_stac_images(response: AgentResponse) -> list[LocalEOImage]:
    """Read STAC-backed local image overlays from response metadata, if present."""

    value = response.metadata.get("stac_images")
    if not isinstance(value, list) or not value:
        return []

    parsed_images: list[LocalEOImage] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        try:
            parsed_images.append(LocalEOImage.model_validate(entry))
        except Exception:
            continue

    return parsed_images


def _render_map(
    output: MapResponseItem,
    response_image: TiledEOImage | None,
    stac_images: list[LocalEOImage],
    component_key: str,
) -> None:
    """Render a Leaflet map with optional EO tile overlay and geography layers."""
    map_obj = _build_map(output, response_image, stac_images)
    acquired_at_text = (
        _format_acquired_at(response_image.acquired_at)
        if response_image is not None
        else None
    )
    if acquired_at_text is not None:
        st.caption(f"WMTS acquisition date: {acquired_at_text}")

    st.markdown(
        """
        <style>
            .leaflet-control,
            .leaflet-bar a,
            .leaflet-control-layers-toggle {
                font-size: 0.8rem;
            }

            .leaflet-control-layers-expanded,
            .leaflet-tooltip,
            .leaflet-popup-content {
                font-size: 0.8rem;
                line-height: 1.2;
            }

            .leaflet-bar a {
                width: 24px;
                height: 24px;
                line-height: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st_folium(
            map_obj,
            key=component_key,
            use_container_width=True,
            height=640,
            returned_objects=[],
        )


def _build_map_component_key(
    response: AgentResponse,
    output: MapResponseItem,
    output_index: int,
) -> str:
    """Return a stable unique key for a rendered folium component."""

    digest = sha256(
        f"{response.request_id}:{output_index}:{output.title}".encode("utf-8")
    ).hexdigest()
    return f"map-{digest}"


def _build_map(
    output: MapResponseItem,
    response_image: TiledEOImage | None,
    stac_images: list[LocalEOImage],
) -> folium.Map:
    """Build a folium map with optional EO tile overlay and geography layers."""
    map_obj = folium.Map(
        location=[output.center_latitude, output.center_longitude],
        zoom_start=output.zoom,
        tiles="CartoDB positron",
        control_scale=True,
    )
    Fullscreen(
        position="topright", title="Fullscreen", title_cancel="Exit Fullscreen"
    ).add_to(map_obj)

    satellite_layer = _build_satellite_layer(response_image)
    if satellite_layer is not None:
        satellite_layer.add_to(map_obj)

    _add_stac_frame_layers(map_obj, stac_images)

    if output.geojson is not None:
        folium.GeoJson(
            output.geojson,
            name=output.title,
            style_function=lambda _: {
                "color": "#ff6347",
                "weight": 2,
                "fill": False,
                # "fillColor": "#ff6347",
                # "fillOpacity": 0.2,
            },
            highlight_function=lambda _: {"weight": 3, "fillOpacity": 0.3},
        ).add_to(map_obj)
    elif output.points:
        for point in output.points:
            folium.CircleMarker(
                location=[point.latitude, point.longitude],
                radius=5,
                color="#cc1f1a",
                fill=True,
                fill_color="#cc1f1a",
                fill_opacity=0.9,
                tooltip=point.label or None,
            ).add_to(map_obj)

    if response_image is not None:
        _add_image_bbox_outline(map_obj, response_image)

    folium.LayerControl(collapsed=True).add_to(map_obj)
    return map_obj


def _add_image_bbox_outline(map_obj: folium.Map, img: TiledEOImage) -> None:
    """Draw the image coverage rectangle to make tile footprint visible."""
    min_lon, min_lat, max_lon, max_lat = img.bbox_wgs84_lat_lon.as_lon_lat_tuple()
    acquired_at_text = _format_acquired_at(img.acquired_at)
    tooltip_text = f"AOI: {img.asset_title}"
    if acquired_at_text is not None:
        tooltip_text = f"{tooltip_text} ({acquired_at_text})"

    folium.Polygon(
        locations=[
            [min_lat, min_lon],
            [max_lat, min_lon],
            [max_lat, max_lon],
            [min_lat, max_lon],
            [min_lat, min_lon],
        ],
        color="#0b7285",
        weight=2,
        fill=False,
        tooltip=tooltip_text,
    ).add_to(map_obj)


def _add_stac_frame_layers(
    map_obj: folium.Map, stac_images: list[LocalEOImage]
) -> None:
    """Attach local STAC frame overlays as optional (hidden-by-default) layers."""

    for index, image in enumerate(stac_images, start=1):
        layer = _build_stac_frame_layer(image, layer_index=index)
        if layer is not None:
            layer.add_to(map_obj)


def _build_stac_frame_layer(
    image: LocalEOImage,
    layer_index: int,
) -> folium.TileLayer | None:
    source_path = image.source_path
    if source_path is None or not source_path.strip():
        return None

    local_file = Path(source_path).resolve()
    if not local_file.is_file():
        print(f"STAC overlay file not found: {local_file}")
        return None

    settings = get_settings()
    if not settings.titiler_base_url.strip():
        print("Skipping STAC overlay because TITILER_BASE_URL is empty.")
        return None
    cache_root = Path(settings.stac_cache_root).resolve()
    if not _is_path_within_root(local_file, cache_root):
        print(
            f"Skipping STAC overlay outside configured cache root. "
            f"file={local_file} cache_root={cache_root}"
        )
        return None

    relative_path = local_file.relative_to(cache_root).as_posix()
    encoded_source_url = quote(relative_path, safe="")
    tiles_url = (
        f"{settings.titiler_base_url.rstrip('/')}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?url={encoded_source_url}"
    )

    min_lon, min_lat, max_lon, max_lat = image.bbox_wgs84_lat_lon.as_lon_lat_tuple()
    layer_name = f"STAC frame {layer_index}: {image.asset_title}"

    return folium.TileLayer(
        tiles=tiles_url,
        attr="STAC frame (TiTiler)",
        name=layer_name,
        overlay=True,
        control=True,
        opacity=0.72,
        show=False,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    )


def _is_path_within_root(path: Path, root: Path) -> bool:
    """Return True when path resolves inside root (or equals root)."""

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def _build_satellite_layer(img: TiledEOImage | None) -> folium.TileLayer | None:
    if img is None:
        print("No EOImage available for map overlay.")
        return None

    tiles_url_template = _resolve_tiles_url_template(img)
    if tiles_url_template is None:
        print("No valid tile URL template available for EO imagery.")
        return None

    options: dict[str, Any] = {}
    if img.tile_size is not None:
        options["tile_size"] = img.tile_size

    layer_name = img.asset_title
    acquired_at_text = _format_acquired_at(img.acquired_at)
    if acquired_at_text is not None:
        layer_name = f"{layer_name} ({acquired_at_text})"

    return folium.TileLayer(
        tiles=tiles_url_template,
        attr="EO imagery",
        name=layer_name,
        overlay=True,
        control=True,
        opacity=0.9,
        min_zoom=img.min_zoom if img.min_zoom is not None else 0,
        max_zoom=img.max_zoom,
        **options,
    )


def _resolve_tiles_url_template(img: TiledEOImage) -> str | None:
    """Resolve a URL template directly or via TileJSON metadata."""
    if img.tiles_url_template is not None and img.tiles_url_template.strip():
        return img.tiles_url_template

    if img.tilejson_url is None or not img.tilejson_url.strip():
        return None

    tilejson = _fetch_tilejson(img.tilejson_url)
    if tilejson is None:
        return None

    tiles = tilejson.get("tiles")
    if isinstance(tiles, list) and tiles:
        first = tiles[0]
        if isinstance(first, str) and first.strip():
            return first

    return None


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_tilejson(url: str) -> dict[str, Any] | None:
    """Load TileJSON metadata once per URL for a short cache window."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Failed to fetch TileJSON from {url}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def _format_acquired_at(acquired_at: datetime | None) -> str | None:
    """Format acquisition timestamps for map labels/tooltips."""

    if acquired_at is None:
        return None
    return acquired_at.isoformat().replace("+00:00", "Z")
