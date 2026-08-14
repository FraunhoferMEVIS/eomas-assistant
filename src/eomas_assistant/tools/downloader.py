# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import collections
import contextlib
import logging
import os
from pathlib import Path, PurePosixPath
from tempfile import mkstemp
from urllib.parse import urlparse

import boto3
import numpy as np
from typing import Tuple
import pystac
import rasterio
import rasterio.windows
import rasterio.merge
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ProfileNotFound


CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

logger = logging.getLogger(__name__)


class AuthenticationFailed(RuntimeError):
    """Raised when authentication credentials for EO data access are unavailable."""


class EOImageDownloader:
    """Download and persist EO image assets to local cache storage."""

    def __init__(
        self,
        endpoint_url: str = "https://eodata.dataspace.copernicus.eu",
    ) -> None:
        self._endpoint_url = endpoint_url

    @staticmethod
    def ensure_safe_path(cdse_local_path: str) -> str:
        """Validate a CDSE-provided relative POSIX path for local caching."""

        error_message = (
            "Invalid CDSE local path. Expected an ASCII-only relative POSIX path "
            f"without empty, dot, dot-dot, or drive-prefixed segments: {cdse_local_path}"
        )

        try:
            cdse_local_path.encode("ascii", errors="strict")
        except UnicodeEncodeError as exc:
            raise RuntimeError(error_message) from exc

        parts = cdse_local_path.split("/")
        is_valid = (
            "\\" not in cdse_local_path
            and bool(parts)
            and all(part not in {"", ".", ".."} for part in parts)
            and not (len(parts[0]) >= 2 and parts[0][1] == ":" and parts[0][0].isalpha())
        )

        if not is_valid:
            raise RuntimeError(error_message)

        return "/".join(parts)

    def download_asset_and_cache_it(self, asset: pystac.Asset) -> Path:
        """Download a STAC asset from S3-compatible storage and cache it locally."""

        safe_local_path = self.ensure_safe_path(asset.extra_fields["file:local_path"])
        local_path = CACHE_DIR / Path(*PurePosixPath(safe_local_path).parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        href = asset.href
        if local_path.exists():
            if local_path.stat().st_size > 0:
                local_path.touch()  # mark as recently used
                return local_path
            else:
                local_path.unlink(missing_ok=True)  # remove empty file and redownload

        parsed = urlparse(href)
        if parsed.scheme != "s3":
            raise RuntimeError(f"Asset href is not an S3 URL: {href}")

        logger.info(f"Downloading asset from URL: {href}")

        try:
            s3 = boto3.client("s3", endpoint_url=self._endpoint_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            with open(local_path, "wb") as file_handle:
                s3.download_fileobj(bucket, key, file_handle)
        except (NoCredentialsError, PartialCredentialsError, ProfileNotFound) as exc:
            local_path.unlink(missing_ok=True)
            raise AuthenticationFailed(
                "EO data access failed because default AWS credentials could not be resolved."
            ) from exc

        if local_path.stat().st_size == 0:
            local_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded empty asset file: {href}")

        return local_path

    def download_and_merge_assets(self, assets: list[pystac.Asset]) -> Tuple[np.ndarray, rasterio.CRS, rasterio.Affine]:
        """Download multiple STAC assets and merge them into a single image.
        Return merged_data (ndarray), merged_crs (CRS), and merged_transform (Affine)."""

        if not assets:
            raise ValueError("No assets provided for download and merge.")

        downloaded_paths = [
            self.download_asset_and_cache_it(asset) for asset in assets
        ]
        with contextlib.ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(path)) for path in downloaded_paths]

            crs = collections.Counter([str(ds.crs) for ds in datasets])
            logger.debug(f"Merging {len(datasets)} datasets, CRS: {crs}")

            if len(crs) > 1:
                (single_crs, _count), = crs.most_common(1)
                logger.debug(f"Cannot merge {len(datasets)} datasets, CRS differ: {crs} (selecting {single_crs})")
                datasets = [ds for ds in datasets if str(ds.crs) == single_crs]

            #out_fd, out_path = mkstemp(prefix="merged", suffix=".tif", dir=CACHE_DIR)
            #os.close(out_fd)  # Close the file descriptor; rasterio will handle the file
            merged_data, merged_transform = rasterio.merge.merge(datasets)  #, dst_path=out_path)

        return merged_data, datasets[0].crs, merged_transform

    def find_assets_by_key(self, items: list[pystac.Item], asset_key: str) -> list[pystac.Asset]:
        """Find specific assets for a list of STAC items."""

        result: list[pystac.Asset] = []
        for item in items:
            asset = item.assets.get(asset_key)
            if asset:
                result.append(asset)
            else:
                logger.warning(f"No asset with key '{asset_key}' found for item {item.id}")

        return result

    def download_and_merge_cloud_probability(
        self,
        stac_items: list[pystac.Item],
    ) -> Tuple[np.ndarray, rasterio.CRS, rasterio.Affine]:
        """Compute the mean cloud probability within a region of interest (ROI).
        In recent Copernicus data, there is a cloud probability asset with key
        'CLD' which is then used.  Alternatively, the scene classification layer
        (SCL) is be used to derive cloud probability, but this is less accurate.
        """

        cld_assets = self.find_assets_by_key(stac_items, "CLD_20m")
        if cld_assets:
            cld_prob, cld_crs, cld_transform = self.download_and_merge_assets(cld_assets)
        else:
            scl_assets = self.find_assets_by_key(stac_items, "SCL_20m")
            scl_prob, cld_crs, cld_transform = self.download_and_merge_assets(scl_assets)

            SCL_CLOUD_PROBABILITY_LUT = [
                0.0,  # 0: No data
                0.0,  # 1: Saturated or defective
                0.0,  # 2: Dark area pixels
                10.0, # 3: Cloud shadows
                0.0,  # 4: Vegetation
                0.0,  # 5: Bare soils
                0.0,  # 6: Water
                5.0,  # 7: Clouds low probability / Unclassified
                30.0, # 8: Clouds medium probability
                100.0,# 9: Clouds high probability
                5.0,  #10: Thin cirrus
                0.0,  #11: Snow or ice
            ]
            cld_prob = np.take(SCL_CLOUD_PROBABILITY_LUT, scl_prob.astype(int))

        return cld_prob, cld_crs, cld_transform

    @staticmethod
    def cropped_png_path(local_path: Path) -> Path:
        """Return a stable PNG path next to the source asset."""

        return local_path.with_name(f"{local_path.stem}_cropped.png")

    @staticmethod
    def cropped_geotiff_path(local_path: Path) -> Path:
        """Return a stable GeoTIFF path next to the source asset."""

        return local_path.with_name(f"{local_path.stem}_cropped.tif")

    @staticmethod
    def write_dataset_to_png(
        dataset: rasterio.DatasetReader,
        output_path: Path,
        window: rasterio.windows.Window | None = None,
    ) -> Path:
        """Write a raster window to PNG, normalizing non-uint8 data."""

        data = dataset.read(window=window)

        if data.dtype != np.uint8:
            data = np.round(data / float(data.max()) * 255).astype(np.uint8)

        profile = {
            **dataset.profile,
            **{
                "driver": "PNG",
                "width": data.shape[2],
                "height": data.shape[1],
                "count": data.shape[0],
                "dtype": data.dtype,
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        return output_path

    @staticmethod
    def write_dataset_to_geotiff(
        dataset: rasterio.DatasetReader,
        output_path: Path,
        window: rasterio.windows.Window | None = None,
    ) -> Path:
        """Write a raster window to GeoTIFF while preserving georeferencing metadata."""

        data = dataset.read(window=window)
        if data.ndim != 3:
            raise RuntimeError("Expected raster data with shape (bands, height, width).")
        transform = dataset.window_transform(window) if window is not None else dataset.transform
        profile = {
            **dataset.profile,
            **{
                "driver": "GTiff",
                "width": data.shape[2],
                "height": data.shape[1],
                "count": data.shape[0],
                "dtype": data.dtype,
                "transform": transform,
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        return output_path
