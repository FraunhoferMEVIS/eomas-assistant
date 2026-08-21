# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from typing import Any

import rasterio
import numpy as np
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


def _geojson_crs_name(geojson: dict[str, Any]) -> str:
    """Read CRS from GeoJSON if present, otherwise assume WGS84."""
    crs = geojson.get("crs")
    if isinstance(crs, dict):
        properties = crs.get("properties")
        if isinstance(properties, dict):
            name = properties.get("name")
            if isinstance(name, str) and name:
                return name
    return "EPSG:4326"


def _extract_geometries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    geometry_list: list[dict[str, Any]] = []

    if payload.get("type") == "FeatureCollection":
        for feature in payload.get("features", []):
            if isinstance(feature, dict) and isinstance(feature.get("geometry"), dict):
                geometry_list.append(feature["geometry"])
    elif payload.get("type") == "Feature" and isinstance(payload.get("geometry"), dict):
        geometry_list.append(payload["geometry"])
    elif isinstance(payload.get("type"), str):
        geometry_list.append(payload)

    if not geometry_list:
        raise ValueError("No geometries found in GeoJSON payload")

    return geometry_list


def _rasterize_geometries(
    geometries: list[dict[str, Any]],
    source_crs: str,
    target_crs: rasterio.CRS,
    target_transform: rasterio.Affine,
    target_shape: tuple[int, int],
    invert: bool = True,
) -> np.ndarray:
    """Rasterize a list of GeoJSON geometries into a numpy array of boolean
    dtype (True for pixels inside geometries iff invert is True).
    """
    transformed_geometries = [transform_geom(source_crs, target_crs, geom) for geom in geometries]
    return geometry_mask(
        transformed_geometries,
        out_shape=target_shape,
        transform=target_transform,
        invert=invert,
    )


def rasterize_geojson(
    geojson: dict[str, Any],
    target_crs: rasterio.CRS,
    target_transform: rasterio.Affine,
    target_shape: tuple[int, int],
    invert: bool = True,
) -> np.ndarray:
    """Rasterize a GeoJSON object into a numpy array of boolean dtype (True for
    pixels inside geometries iff invert is True).
    """
    source_crs = _geojson_crs_name(geojson)
    geometries = _extract_geometries(geojson)
    return _rasterize_geometries(
        geometries=geometries,
        source_crs=source_crs,
        target_crs=target_crs,
        target_transform=target_transform,
        target_shape=target_shape,
        invert=invert,
    )


def rasterize_geojson_on_dataset(
    geojson: dict[str, Any],
    dataset: rasterio.DatasetReader,
    invert: bool = True,
) -> np.ndarray:
    """Rasterize a GeoJSON object into a numpy array of boolean dtype (True for
    pixels inside geometries iff invert is True) on the given rasterio dataset.
    """
    return rasterize_geojson(
        geojson=geojson,
        target_crs=dataset.crs,
        target_transform=dataset.transform,
        target_shape=(dataset.height, dataset.width),
        invert=invert,
    )
