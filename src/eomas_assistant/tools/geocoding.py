# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import re
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from eomas_assistant.models.schemas import GeoLocation


class Geocoding:
    """OpenStreetMap Nominatim wrapper for point and area geocoding."""

    def __init__(self, base_url: str, user_agent: str, timeout_seconds: int) -> None:
        self._search_endpoint = f"{base_url.rstrip('/')}/search"
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def geocode(self, query: str) -> GeoLocation | None:
        """Resolve a free-text query into a normalized `GeoLocation`.

        Requests multiple candidates including polygon geometry and applies
        internal ranking to improve results for broad administrative places.
        """
        params = {
            "q": query,
            "format": "json",
            "limit": 5,
            "polygon_geojson": 1,
        }
        headers = {"User-Agent": self._user_agent}

        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.get(self._search_endpoint, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, list) or not data:
            return None

        chosen = self._select_best_result(query=query, results=data)
        lat = float(chosen["lat"])
        lon = float(chosen["lon"])
        name = str(chosen.get("name") or chosen.get("display_name", query).split(",")[0].strip())

        return GeoLocation(
            query=query,
            name=name,
            latitude=lat,
            longitude=lon,
            display_name=str(chosen.get("display_name", query)),
            geojson=self._extract_supported_geojson(chosen),
        )

    def _select_best_result(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Pick the most suitable Nominatim result for the given query.

        For short generic place names, prefer administrative entities such as
        states/regions when available; otherwise keep the top API result.
        """
        normalized_query = query.strip().lower()
        only_letters = re.sub(r"[^a-z\s]", "", normalized_query)
        token_count = len([token for token in only_letters.split(" ") if token])
        looks_like_generic_place = token_count <= 2 and "," not in normalized_query

        if looks_like_generic_place:
            administrative_matches = [
                item
                for item in results
                if str(item.get("addresstype", "")).lower()
                in {"state", "region", "province", "county"}
            ]
            if administrative_matches:
                administrative_matches.sort(
                    key=lambda item: int(item.get("place_rank", 1000))
                )
                return administrative_matches[0]

        return results[0]

    def _extract_supported_geojson(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Convert supported polygon geometries into a GeoJSON FeatureCollection."""
        raw_geojson = result.get("geojson")
        if not isinstance(raw_geojson, dict):
            return None

        geometry_type = str(raw_geojson.get("type", ""))
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            return None

        coordinates = raw_geojson.get("coordinates")
        if not coordinates:
            return None

        feature = {
            "type": "Feature",
            "properties": {
                "name": result.get("display_name", "Area"),
            },
            "geometry": {
                "type": geometry_type,
                "coordinates": coordinates,
            },
        }
        return {
            "type": "FeatureCollection",
            "features": [feature],
        }
