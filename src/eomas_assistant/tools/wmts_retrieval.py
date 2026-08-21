# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import base64
from datetime import timedelta
import logging
import requests
from urllib.parse import quote
import xml.etree.ElementTree as ET

from eomas_assistant.models import GeoLocation
from eomas_assistant.tools.datetimeconversion import datetime_range_to_str
from eomas_assistant.config import AppSettings, get_settings
from eomas_assistant.models.schemas import TimeRange, DataRequest, TiledEOImage
from eomas_assistant.tools.geography import ensure_minimal_bbox_span

logger = logging.getLogger(__name__)


def _construct_wmts_instance_url(settings: AppSettings) -> str:
    if settings.sentinel_hub_instance_id is None or not settings.sentinel_hub_instance_id.strip():
        raise RuntimeError(
            "SENTINEL_HUB_INSTANCE_ID is not configured. "
            "Set this environment variable to enable tiled EO imagery responses."
        )

    base_url = settings.sentinel_hub_wmts_base_url.rstrip("/")
    instance_id = quote(settings.sentinel_hub_instance_id.strip(), safe="")
    return f"{base_url}/{instance_id}"


def _construct_wmts_gettile_url(
    settings: AppSettings,
    datetime_range: TimeRange,
    layer: str,
    max_cc: float | None = 100.0,  # range 0..100
) -> str:
    instance_url = _construct_wmts_instance_url(settings)

    effective_style = settings.sentinel_hub_style.strip() or "default"
    effective_format = settings.sentinel_hub_format.strip() or "image/png"

    encoded_layer = quote(layer, safe="")
    encoded_style = quote(effective_style, safe="")
    encoded_format = quote(effective_format, safe="/")
    encoded_matrix_set = quote(settings.sentinel_hub_tile_matrix_set, safe="")
    datetime_range_str = datetime_range_to_str(datetime_range)
    encoded_time = quote(datetime_range_str, safe=":/-TZ.")

    # see:
    # https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/OGC/WMTS.html
    # https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/OGC/AdditionalRequestParameters.html
    # but note that the actual defaults are given in the layer configuration, so
    # we need to explicitly set things like MAXCC here to override the defaults:
    final_url = (
        f"{instance_url}"
        "?SERVICE=WMTS"
        "&REQUEST=GetTile"
        "&VERSION=1.0.0"
        f"&LAYER={encoded_layer}"
        f"&STYLE={encoded_style}"
        f"&FORMAT={encoded_format}"
        f"&TILEMATRIXSET={encoded_matrix_set}"
        "&TILEMATRIX={z}"
        "&TILEROW={y}"
        "&TILECOL={x}"
        f"&TIME={encoded_time}"
        "&TEMPORAL=true"
        "&WARNINGS=no"
        "&PRIORITY=mostRecent"
    )
    if max_cc is not None:
        final_url += f"&MAXCC={max_cc}"

    # custom_script = _load_custom_script("custom_script_temporal_agricultural_growth.js")
    # encoded_script = quote(custom_script)
    # final_url += f"&EVALSCRIPT={encoded_script}"

    # print(f"Constructed WMTS tiles URL template: {final_url}")
    return final_url


def request_available_wmts_layers() -> dict[str, str]:
    settings = get_settings()
    requests.get(
        _construct_wmts_instance_url(settings),
        params={"SERVICE": "WMTS", "REQUEST": "GetCapabilities"},
    )

    response = requests.get(
        _construct_wmts_instance_url(settings),
        params={"SERVICE": "WMTS", "REQUEST": "GetCapabilities"},
    )
    root = ET.fromstring(response.content)

    # Namespaces used in the document
    ns = {
        "wmts": "http://www.opengis.net/wmts/1.0",
        "ows": "http://www.opengis.net/ows/1.1",
    }

    layers = {}
    for layer in root.iterfind(".//wmts:Contents/wmts:Layer", ns):
        identifier_el = layer.find("ows:Identifier", ns)
        title_el = layer.find("ows:Title", ns)

        if identifier_el is None or title_el is None:
            continue  # skip malformed entries defensively

        layer_id = identifier_el.text.strip() if identifier_el.text else ""
        title = title_el.text.strip() if title_el.text else ""
        layers[layer_id] = title

    return layers


def _load_custom_script(filename: str):
    with open(filename) as f:
        text = f.read()
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def construct_tiled_eo_image_with_wmts_metadata(
    geo_location: GeoLocation,
    data_request: DataRequest,
) -> TiledEOImage | None:
    """Resolve tiled EO image metadata using GeoLocation.time_range as temporal input."""

    assert geo_location.bbox_wgs84_lat_lon
    bbox_wgs84_lat_lon = ensure_minimal_bbox_span(geo_location.bbox_wgs84_lat_lon)
    settings = get_settings()

    try:
        assert data_request.wmts_layer is not None, "wmts_layer must be set in data_request"
        assert data_request.acquired_at is not None, "acquired_at must be set in data_request"

        logger.info(f"Requested WMTS layer: {data_request.wmts_layer}")
        logger.info(
            f"Requested WMTS acquisition date: {data_request.acquired_at.isoformat() if data_request.acquired_at else 'None'}"
        )

        start_timepoint = data_request.acquired_at.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # FIXME: extending the time range is a hot fix for the fact that the
        # Sentinel 2 strips have gaps, so a single day is not enough for a
        # gap-less display:
        timerange = TimeRange(
            start_timepoint=start_timepoint - timedelta(days=5),
            end_timepoint=start_timepoint + timedelta(days=1),
        )

        wmts_url = _construct_wmts_gettile_url(
            settings=settings,
            datetime_range=timerange,
            layer=data_request.wmts_layer,
            #            max_cc=data_request.max_cloud_cover,
        )
    except Exception as exc:
        # can this happen at all? the URL construction is deterministic and should not fail AFAICS
        logger.error(f"WMTS image resolution failed", exc_info=exc)
        return None

    try:
        image = TiledEOImage(
            bbox_wgs84_lat_lon=bbox_wgs84_lat_lon,
            asset_key=data_request.wmts_layer,
            asset_title=data_request.wmts_layer or "Unknown Asset",
            acquired_at=data_request.acquired_at,
            tiles_url_template=wmts_url,
            min_zoom=settings.sentinel_hub_min_zoom,
            max_zoom=settings.sentinel_hub_max_zoom,
            tile_size=settings.sentinel_hub_tile_size,
        )
    except Exception as exc:
        logger.exception(f"Data response build failed", exc_info=exc)
        return None

    return image
