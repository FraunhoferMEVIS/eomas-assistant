# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from eomas_assistant.models.schemas import BoundingBox


def ensure_minimal_bbox_span(
    bbox: BoundingBox,
    min_lon_span: float = 0.2,
    min_lat_span: float = 0.1,
) -> BoundingBox:
    """
    Ensure that the given bounding box has at least the specified minimum span
    in both longitude and latitude. If the span is smaller than the minimum,
    expand the box symmetrically around its center.
    """
    min_lon, min_lat, max_lon, max_lat = bbox.as_lon_lat_tuple()
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    if lon_span < min_lon_span:
        mid_lon = (min_lon + max_lon) / 2
        min_lon = mid_lon - min_lon_span / 2
        max_lon = mid_lon + min_lon_span / 2

    if lat_span < min_lat_span:
        mid_lat = (min_lat + max_lat) / 2
        min_lat = mid_lat - min_lat_span / 2
        max_lat = mid_lat + min_lat_span / 2

    return BoundingBox(
        min_latitude=min_lat,
        min_longitude=min_lon,
        max_latitude=max_lat,
        max_longitude=max_lon,
    )
