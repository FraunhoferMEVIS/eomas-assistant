# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from typing import Any, Tuple
import rasterio
import numpy as np
import pystac

from eomas_assistant.tools.downloader import EOImageDownloader
from eomas_assistant.tools.rasterize_geojson import rasterize_geojson


def compute_derived_index(
    stac_items: list[pystac.Item],
    index_name: str,
) -> Tuple[np.ndarray, rasterio.CRS, rasterio.Affine]:
    """Compute a derived index (e.g. NDVI) from the pixel values of the specified
    asset type inside the region of interest for a given acquisition date (in ISO
    8601 YYYY-MM-DD format).
    """

    assert index_name == 'NDVI', f"Derived index '{index_name}' is not supported. Only 'NDVI' is currently implemented."
    # NDVI = (MIR - NIR) / (MIR + NIR), with MIR = B12_20m and NIR = B08_20m for Sentinel-2 L2A
    needed_assets = ['B12_20m', 'B8A_20m']

    result_crs = None
    result_transform = None
    base_data = dict()

    downloader = EOImageDownloader()
    for asset_key in needed_assets:
        assets = downloader.find_assets_by_key(stac_items, asset_key)

        image_data, image_crs, image_transform = downloader.download_and_merge_assets(
            assets
        )

        if result_crs is None:
            result_crs = image_crs
            result_transform = image_transform
        else:
            if result_crs != image_crs:
                raise ValueError(f"CRS of asset '{asset_key}' does not match the CRS of previous assets.")
            if result_transform != image_transform:
                raise ValueError(f"Transform of asset '{asset_key}' does not match the transform of previous assets.")

        base_data[asset_key] = image_data

    # TODO: generalize to other indices (requires formula parsing, e.g., based on indexdatabase API)
    nir_data = base_data['B8A_20m']
    mir_data = base_data['B12_20m']

    # Compute NDVI
    ndvi = (mir_data - nir_data) / (mir_data + nir_data)

    assert result_crs is not None
    assert result_transform is not None

    return ndvi, result_crs, result_transform


def compute_roi_statistics(
    stac_items: list[pystac.Item],
    stac_asset_key_or_index_name: str,
    roi_geojson: dict[str, Any],
) -> dict[str, float]:
    """Compute statistics (mean, min, max, stddev) of the pixel values inside
    the region of interest for a given acquisition date (in ISO 8601
    YYYY-MM-DD format).
    """

    downloader = EOImageDownloader()
    try:
        assets = downloader.find_assets_by_key(stac_items, stac_asset_key_or_index_name)

        image_data, image_crs, image_transform = downloader.download_and_merge_assets(
            assets
        )
    except KeyError:
        image_data, image_crs, image_transform = compute_derived_index(
            stac_items, stac_asset_key_or_index_name
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
