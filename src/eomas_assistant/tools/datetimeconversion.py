# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from eomas_assistant.models.schemas import TimeRange


# FIXME: This function unexpectedly applies hard-coded defaults.  I would rather
# let the LLM fill any gaps in the user request.  Also, the name/signature does
# not reveal this behavior, and it is not clear what the intent of this function
# is / who's calling it.
def datetime_range_to_str(time_range: TimeRange | None) -> str:
    """Convert TimeRange into STAC/WMTS datetime string, using defaults if incomplete."""

    default_range = "2025-01-01T00:00:00Z/2026-12-31T23:59:59Z"
    if time_range is None:
        print("Could not convert/extract timerange, applying default: " + default_range)
        return default_range

    if time_range.start_timepoint is None or time_range.end_timepoint is None:
        print("Could not convert/extract timerange, applying default: " + default_range)
        return default_range

    return cdse_datetime_range_str(time_range.start_timepoint, time_range.end_timepoint)


def _assume_utc_if_naive(dt: datetime) -> datetime:
    """Return a datetime with UTC timezone if naive, otherwise return unchanged."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def cdse_datetime_range_str(start_datetime: datetime, end_datetime: datetime) -> str:
    start_iso = _assume_utc_if_naive(start_datetime).isoformat().replace("+00:00", "Z")
    end_iso = _assume_utc_if_naive(end_datetime).isoformat().replace("+00:00", "Z")
    return f"{start_iso}/{end_iso}"
