# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

import pystac
from pystac_client import Client, ItemSearch

from eomas_assistant.models.schemas import BoundingBox, TimeRange
from eomas_assistant.tools.datetimeconversion import datetime_range_to_str

EARTH_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1"


def format_acquisition_date(item: pystac.Item) -> str:
    """Format the acquisition date of a STAC item as an ISO 8601 string."""
    if item.datetime is not None:
        return item.datetime.strftime("%Y-%m-%d")
    return "unknown"


def find_sentinel2_assets_in_time_range(
    bbox_wgs84: BoundingBox,
    datetime_range: TimeRange | None,
    max_cloud_cover: float | None,
    max_items: int = 1,
) -> ItemSearch:
    """Search Sentinel-2 STAC items for a WGS84 bbox and optional TimeRange filter."""

    client = Client.open(EARTH_SEARCH_URL)
    return client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox_wgs84.as_lon_lat_tuple()),
        datetime=datetime_range_to_str(datetime_range),
        query=(
            {"eo:cloud_cover": {"lt": max_cloud_cover}}
            if max_cloud_cover is not None
            else None
        ),
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        limit=min(max_items, 100),
        max_items=max_items,
    )
