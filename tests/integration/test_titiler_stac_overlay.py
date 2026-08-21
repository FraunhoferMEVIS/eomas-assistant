# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from eomas_assistant.app.titiler_app import app as titiler_app
from eomas_assistant.models.schemas import BoundingBox, LocalEOImage
from eomas_assistant.ui.renderers import _build_stac_frame_layer


@pytest.mark.integration
def test_stac_overlay_layer_uses_restricted_titiler_url_and_serves_tilejson() -> None:
    with TemporaryDirectory() as tmp_dir:
        cache_root = Path(tmp_dir) / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        rel_path = Path("scene") / "crop.tif"
        tif_path = cache_root / rel_path
        tif_path.parent.mkdir(parents=True, exist_ok=True)

        width = 16
        height = 16
        bounds = (8.55, 53.00, 8.95, 53.22)
        transform = from_bounds(*bounds, width=width, height=height)
        data = np.full((1, height, width), 42, dtype=np.uint8)

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        image = LocalEOImage(
            bbox_wgs84_lat_lon=BoundingBox(
                min_latitude=53.00,
                min_longitude=8.55,
                max_latitude=53.22,
                max_longitude=8.95,
            ),
            asset_key="TCI_10m",
            asset_title="True Color",
            source_path=str(tif_path),
        )

        patched_settings = SimpleNamespace(
            titiler_base_url="http://127.0.0.1:8000",
            stac_cache_root=str(cache_root),
        )

        with (
            patch("eomas_assistant.ui.renderers.get_settings", return_value=patched_settings),
            patch("eomas_assistant.app.titiler_app.get_settings", return_value=patched_settings),
        ):
            layer = _build_stac_frame_layer(image=image, layer_index=1)
            assert layer is not None
            assert isinstance(layer.tiles, str)

            parsed = urlparse(layer.tiles)
            query = parse_qs(parsed.query)
            assert parsed.path.endswith("/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png")
            assert query.get("url")
            assert unquote(query["url"][0]) == rel_path.as_posix()

            client = TestClient(titiler_app)
            ok_response = client.get(
                f"/cog/WebMercatorQuad/tilejson.json?url={rel_path.as_posix()}"
            )
            assert ok_response.status_code == 200
            payload = ok_response.json()
            assert isinstance(payload.get("tiles"), list)
            assert payload["tiles"]

            traversal_response = client.get("/cog/WebMercatorQuad/tilejson.json?url=../outside.tif")
            assert traversal_response.status_code == 403

            absolute_response = client.get(
                f"/cog/WebMercatorQuad/tilejson.json?url={tif_path.as_posix()}"
            )
            assert absolute_response.status_code == 400


@pytest.mark.integration
def test_titiler_serves_real_png_tile_for_cached_geotiff() -> None:
    with TemporaryDirectory() as tmp_dir:
        cache_root = Path(tmp_dir) / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        rel_path = Path("scene") / "tile-source.tif"
        tif_path = cache_root / rel_path
        tif_path.parent.mkdir(parents=True, exist_ok=True)

        width = 32
        height = 32
        bounds = (8.55, 53.00, 8.95, 53.22)
        transform = from_bounds(*bounds, width=width, height=height)
        data = np.full((1, height, width), 120, dtype=np.uint8)

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        patched_settings = SimpleNamespace(
            titiler_base_url="http://127.0.0.1:8000",
            stac_cache_root=str(cache_root),
        )

        with patch("eomas_assistant.app.titiler_app.get_settings", return_value=patched_settings):
            client = TestClient(titiler_app)
            tile_response = client.get(
                f"/cog/tiles/WebMercatorQuad/0/0/0.png?url={rel_path.as_posix()}"
            )

            assert tile_response.status_code == 200
            assert tile_response.headers.get("content-type", "").startswith("image/png")
            assert tile_response.content

            blocked_response = client.get("/cog/tiles/WebMercatorQuad/0/0/0.png?url=../outside.tif")
            assert blocked_response.status_code == 403
