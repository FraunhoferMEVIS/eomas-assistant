# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from typing import Any
import numpy as np
import pystac

from eomas_assistant.tools.downloader import EOImageDownloader
from eomas_assistant.tools.rasterize_geojson import rasterize_geojson


def compute_roi_statistics(
    stac_items: list[pystac.Item],
    stac_asset_key: str,
    roi_geojson: dict[str, Any],
) -> dict[str, float]:
    """Compute statistics (mean, min, max, stddev) of the pixel values inside
    the region of interest for a given acquisition date (in ISO 8601
    YYYY-MM-DD format).
    """

    downloader = EOImageDownloader()
    assets = downloader.find_assets_by_key(stac_items, stac_asset_key)
    if not assets:
        raise ValueError(f"No assets with key '{stac_asset_key}' found for the given STAC items.")

    image_data, image_crs, image_transform = downloader.download_and_merge_assets(
        assets
    )

    roi_mask = rasterize_geojson(
        roi_geojson, image_crs, image_transform, image_data.shape[-2:], invert=False
    )
    masked_pixels = np.ma.masked_array(image_data, mask=roi_mask)

    return dict(
        mean=float(masked_pixels.mean()),
        min=float(masked_pixels.min()),
        max=float(masked_pixels.max()),
        stddev=float(masked_pixels.std()),
    )
