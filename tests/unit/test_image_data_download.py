# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

import unittest
from datetime import UTC, datetime
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from botocore.exceptions import ProfileNotFound
from pystac_client import ItemSearch

from eomas_assistant.tools.wmts_retrieval import (
    construct_tiled_eo_image_with_wmts_metadata,
)
from eomas_assistant.tools.wmts_retrieval import request_available_wmts_layers
from eomas_assistant.tools.find_available_satellite_data import (
    find_sentinel2_assets_in_time_range,
)
from eomas_assistant.models.schemas import (
    BoundingBox,
    DataRequest,
    GeoLocation,
    LocalEOImage,
    TimeRange,
)
from eomas_assistant.tools.downloader import (
    AuthenticationFailed,
    EOImageDownloader,
)

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# DATA_DIR = PROJECT_ROOT / "data"


# class TestFallbackHandling(unittest.TestCase):
#   def test_presence_of_default_image_returns_bremen_window(self):
#     self.assertEqual(_load_default_eoimage().asset_key, "bremen_window")

#   @skip # temporally disabled since unsure which format the invoke call needs
#   def test_download_of_image_failing_returns_bremen_window(self):
#     image = _download_from_sentinel.invoke(
#       location="Bremen",
#       bbox_wgs_lat_lon=(8.55, 53.00, 8.95, 53.22),
#       timepoint="2023-01-01T00:00:00Z/2024-12-31T23:59:59Z"
#     )
#     self.assertEqual(
#       image.asset_key,
#       "bremen_window")


class TestGetAvailableEOImages(unittest.TestCase):

    def test_requesting_eodata_bremen_frame_and_date_returns_list(self):
        imageList = find_sentinel2_assets_in_time_range(
            bbox_wgs84=BoundingBox(
                min_latitude=53.00,
                min_longitude=8.55,
                max_latitude=53.22,
                max_longitude=8.95,
            ),
            datetime_range=TimeRange(
                start_timepoint=datetime(2022, 1, 1, 0, 0, 0, tzinfo=UTC),
                end_timepoint=datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
            ),
            max_cloud_cover=80.0,
        )
        self.assertIsInstance(imageList, ItemSearch)
        self.assertGreater(
            len(list(imageList.item_collection())),
            0,
        )


class TestAuthenticationErrors(unittest.TestCase):

    @patch(
        "eomas_assistant.tools.downloader.boto3.client",
        side_effect=ProfileNotFound(profile="default"),
    )
    def test_missing_profile_raises_authentication_failed(self, mock_boto3_client):
        asset = SimpleNamespace(
            href="s3://example-bucket/path/to/asset.jp2",
            extra_fields={"file:local_path": "safe/path/to/asset.jp2"},
        )
        downloader = EOImageDownloader()

        with self.assertRaises(AuthenticationFailed) as raised_exc:
            downloader.download_asset_and_cache_it(asset=asset)  # type: ignore

        mock_boto3_client.assert_called_once_with(
            "s3", endpoint_url="https://eodata.dataspace.copernicus.eu"
        )
        self.assertIn("default AWS credentials", str(raised_exc.exception))
        self.assertIsInstance(raised_exc.exception.__cause__, ProfileNotFound)

    @patch(
        "eomas_assistant.tools.wmts_retrieval._construct_wmts_gettile_url",
        side_effect=AuthenticationFailed("missing auth"),
    )
    def test_download_images_auth_failure_returns_empty_response(
        self,
        mock_construct_wmts,
    ):
        geo_location = GeoLocation(
            query="Bremen",
            name="Bremen",
            latitude=53.0793,
            longitude=8.8017,
            display_name="Bremen, Germany",
            bbox_wgs84_lat_lon=BoundingBox(
                min_latitude=53.00,
                min_longitude=8.55,
                max_latitude=53.22,
                max_longitude=8.95,
            ),
        )
        request = DataRequest(
            wmts_layer="TRUE_COLOR",
            acquired_at=datetime(2024, 5, 16, 10, 40, 21, tzinfo=UTC),
        )

        response = construct_tiled_eo_image_with_wmts_metadata(
            geo_location=geo_location, data_request=request
        )

        self.assertEqual(response, None)
        mock_construct_wmts.assert_called_once()

    @patch(
        "eomas_assistant.tools.wmts_retrieval._construct_wmts_gettile_url",
        return_value="https://example.com/wmts/{z}/{x}/{y}.png",
    )
    def test_wmts_image_includes_acquisition_date_from_latest_stac_item(
        self,
        mock_construct_wmts,
    ):
        geo_location = GeoLocation(
            query="Bremen",
            name="Bremen",
            latitude=53.0793,
            longitude=8.8017,
            display_name="Bremen, Germany",
            bbox_wgs84_lat_lon=BoundingBox(
                min_latitude=53.00,
                min_longitude=8.55,
                max_latitude=53.22,
                max_longitude=8.95,
            ),
        )
        request = DataRequest(
            wmts_layer="TRUE_COLOR",
            max_cloud_cover=20.0,
            acquired_at=datetime(2024, 5, 16, 10, 40, 21, tzinfo=UTC),
        )

        response = construct_tiled_eo_image_with_wmts_metadata(
            geo_location=geo_location, data_request=request
        )

        self.assertIsNotNone(response)
        assert response is not None  # for linter
        self.assertEqual(
            response.acquired_at, datetime(2024, 5, 16, 10, 40, 21, tzinfo=UTC)
        )
        mock_construct_wmts.assert_called_once()


class TestWmtsLayerDiscovery(unittest.TestCase):

    def test_request_available_wmts_layers_returns_multiple_layers(self):
        layers = request_available_wmts_layers()

        self.assertGreater(
            len(layers),
            1,
        )


if __name__ == "__main__":
    unittest.main()
