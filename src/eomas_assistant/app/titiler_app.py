# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from titiler.core.factory import TilerFactory

from eomas_assistant.config.settings import get_settings

LOGGER = logging.getLogger(__name__)


def _resolve_cache_root() -> Path:
    return Path(get_settings().stac_cache_root).resolve()


def _validated_local_dataset_path(url: Annotated[str, Query(...)]) -> str:
    """Resolve a cache-relative dataset path and reject traversal attempts."""

    requested_path = url.strip()
    if not requested_path:
        LOGGER.warning("TiTiler path validation failed: empty dataset path.")
        raise HTTPException(status_code=400, detail="Missing dataset path.")

    relative_path = Path(requested_path)
    if relative_path.is_absolute():
        LOGGER.warning(
            "TiTiler path validation failed: absolute path rejected path=%s", requested_path
        )
        raise HTTPException(status_code=400, detail="Dataset path must be relative.")

    candidate = (_resolve_cache_root() / relative_path).resolve()
    try:
        candidate.relative_to(_resolve_cache_root())
    except ValueError as exc:
        LOGGER.warning(
            "TiTiler path validation failed: outside cache root path=%s cache_root=%s",
            requested_path,
            _resolve_cache_root(),
        )
        raise HTTPException(
            status_code=403,
            detail="Requested dataset path is outside configured cache root.",
        ) from exc

    if not candidate.is_file():
        LOGGER.warning("TiTiler path validation failed: dataset not found path=%s", candidate)
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    LOGGER.info("TiTiler path validation succeeded path=%s", candidate)

    return str(candidate)


app = FastAPI(title="EOMAS TiTiler", docs_url=None, redoc_url=None)


@app.middleware("http")
async def log_incoming_requests(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    duration_ms = (perf_counter() - started) * 1000.0
    client_host = request.client.host if request.client is not None else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    LOGGER.info(
        "TiTiler request method=%s path=%s query=%s status=%s"
        " duration_ms=%.2f client=%s user_agent=%s",
        request.method,
        request.url.path,
        request.url.query,
        response.status_code,
        duration_ms,
        client_host,
        user_agent,
    )
    return response


cog = TilerFactory(path_dependency=_validated_local_dataset_path)
app.include_router(cog.router, prefix="/cog", tags=["Cloud Optimized GeoTIFF"])


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
