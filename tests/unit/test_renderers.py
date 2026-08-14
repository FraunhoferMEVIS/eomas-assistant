# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from eomas_assistant.ui.renderers import (
    _build_cloud_cover_vega_spec,
    _build_stac_table_rows_and_chart_points,
    _is_path_within_root,
    _parse_stac_acquisition_datetime,
)


class TestRenderers(unittest.TestCase):

    def test_build_stac_rows_and_chart_points_filters_invalid_chart_entries(self) -> None:
        rows, chart_points = _build_stac_table_rows_and_chart_points(
            [
                {
                    "acquisition_date": "2024-05-01T10:00:00Z",
                    "stac_cc": 12.5,
                },
                {
                    "acquisition_date": "unknown",
                    "stac_cc": 9.0,
                },
                {
                    "acquisition_date": "2024-05-02T10:00:00Z",
                    "stac_cc": None,
                },
                {
                    "acquisition_date": "2024-05-03T10:00:00Z",
                    "stac_cc": 22.1,
                },
            ]
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["stac_cc"], "12.5%")
        self.assertEqual(rows[1]["stac_cc"], "9.0%")
        self.assertEqual(rows[2]["stac_cc"], "unknown")
        self.assertEqual(rows[3]["stac_cc"], "22.1%")

        self.assertEqual(len(chart_points), 2)
        self.assertEqual(chart_points[0]["stac_cc"], 12.5)
        self.assertEqual(
            chart_points[0]["acquisition_date"],
            datetime(2024, 5, 1, 10, 0, 0, tzinfo=UTC),
        )
        self.assertEqual(chart_points[1]["stac_cc"], 22.1)
        self.assertEqual(
            chart_points[1]["acquisition_date"],
            datetime(2024, 5, 3, 10, 0, 0, tzinfo=UTC),
        )

    def test_parse_stac_acquisition_datetime(self) -> None:
        self.assertEqual(
            _parse_stac_acquisition_datetime("2024-05-01T10:00:00Z"),
            datetime(2024, 5, 1, 10, 0, 0, tzinfo=UTC),
        )
        self.assertIsNone(_parse_stac_acquisition_datetime("unknown"))
        self.assertIsNone(_parse_stac_acquisition_datetime("not-a-date"))

    def test_build_cloud_cover_vega_spec_sorts_points(self) -> None:
        spec = _build_cloud_cover_vega_spec(
            [
                {
                    "acquisition_date": datetime(2024, 5, 2, 10, 0, 0, tzinfo=UTC),
                    "stac_cc": 20.0,
                    "roi_cc": 11.0,
                },
                {
                    "acquisition_date": datetime(2024, 5, 1, 10, 0, 0, tzinfo=UTC),
                    "stac_cc": 10.0,
                },
            ]
        )

        self.assertEqual(spec["$schema"], "https://vega.github.io/schema/vega-lite/v5.json")
        values = spec["data"]["values"]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]["acquisition_date"], "2024-05-01T10:00:00+00:00")
        self.assertEqual(values[0]["stac_cc"], 10.0)
        self.assertIsNone(values[0]["roi_cc"])
        self.assertEqual(values[1]["acquisition_date"], "2024-05-02T10:00:00+00:00")
        self.assertEqual(values[1]["stac_cc"], 20.0)
        self.assertEqual(values[1]["roi_cc"], 11.0)
        self.assertEqual(spec["transform"][0]["fold"], ["stac_cc", "roi_cc"])
        self.assertEqual(spec["encoding"]["y"]["field"], "cloud_cover")
        self.assertEqual(spec["encoding"]["color"]["field"], "series_label")

    def test_is_path_within_root_accepts_nested_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "a" / "b" / "frame.tif"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("dummy", encoding="ascii")

            self.assertTrue(_is_path_within_root(nested, root))

    def test_is_path_within_root_rejects_parent_traversal(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            root.mkdir(parents=True, exist_ok=True)
            outside = Path(tmp) / "outside" / "frame.tif"
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("dummy", encoding="ascii")

            self.assertFalse(_is_path_within_root(outside, root))

