# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from eomas_assistant.app.titiler_app import _validated_local_dataset_path
from eomas_assistant.app.titiler_app import app as titiler_app


class TestTiTilerPathValidation(unittest.TestCase):
    @patch("eomas_assistant.app.titiler_app.get_settings")
    def test_accepts_file_within_cache_root(self, mock_get_settings) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            dataset = cache_root / "scene" / "crop.tif"
            dataset.parent.mkdir(parents=True, exist_ok=True)
            dataset.write_text("x", encoding="ascii")
            mock_get_settings.return_value = SimpleNamespace(stac_cache_root=str(cache_root))

            resolved = _validated_local_dataset_path("scene/crop.tif")

            self.assertEqual(resolved, str(dataset.resolve()))

    @patch("eomas_assistant.app.titiler_app.get_settings")
    def test_rejects_parent_traversal(self, mock_get_settings) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            mock_get_settings.return_value = SimpleNamespace(stac_cache_root=str(cache_root))

            with self.assertRaises(HTTPException) as raised:
                _validated_local_dataset_path("../outside.tif")

            self.assertEqual(raised.exception.status_code, 403)

    @patch("eomas_assistant.app.titiler_app.get_settings")
    def test_rejects_absolute_paths(self, mock_get_settings) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            mock_get_settings.return_value = SimpleNamespace(stac_cache_root=str(cache_root))

            with self.assertRaises(HTTPException) as raised:
                _validated_local_dataset_path(str((cache_root / "a.tif").resolve()))

            self.assertEqual(raised.exception.status_code, 400)


class TestTiTilerLogging(unittest.TestCase):
    def test_healthz_request_is_logged(self) -> None:
        client = TestClient(titiler_app)

        with self.assertLogs("eomas_assistant.app.titiler_app", level="INFO") as captured:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            any("TiTiler request method=GET path=/healthz" in line for line in captured.output)
        )

    @patch("eomas_assistant.app.titiler_app.get_settings")
    def test_blocked_path_is_logged(self, mock_get_settings) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            mock_get_settings.return_value = SimpleNamespace(stac_cache_root=str(cache_root))

            client = TestClient(titiler_app)
            with self.assertLogs("eomas_assistant.app.titiler_app", level="INFO") as captured:
                response = client.get("/cog/WebMercatorQuad/tilejson.json?url=../outside.tif")

            self.assertEqual(response.status_code, 403)
            self.assertTrue(
                any("outside cache root" in line for line in captured.output),
                msg=f"Expected path-validation warning in logs, got: {captured.output}",
            )
            self.assertTrue(
                any("path=/cog/WebMercatorQuad/tilejson.json" in line for line in captured.output),
                msg=f"Expected request log in logs, got: {captured.output}",
            )
