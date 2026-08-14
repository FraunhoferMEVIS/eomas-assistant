# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        alias_generator=str.upper,
    )

    app_name: str = "EOMAS Assistant"
    debug: bool = False

    llm_model: str = "eve-esa/EVE-Instruct"
    llm_provider: Literal["ollama", "vllm_openai"] = "vllm_openai"

    vllm_base_url: str = "https://vllm.cloud.intern.mevis.fraunhofer.de/gen/v1"
    vllm_api_key: SecretStr | None = None

    ollama_base_url: str = "https://ollama.cloud.intern.mevis.fraunhofer.de"
    llm_ca_bundle_path: str | None = None

    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "eomas-assistant/0.1.0"

    request_timeout_seconds: int = 30
    workflow_max_attempts: int = 3

    imagery_provider: Literal["sentinel_hub_wmts"] = "sentinel_hub_wmts"
    sentinel_hub_wmts_base_url: str = "https://sh.dataspace.copernicus.eu/ogc/wmts"
    sentinel_hub_instance_id: str | None = None
    sentinel_hub_tile_matrix_set: str = "PopularWebMercator256"
    sentinel_hub_style: str = "default"
    sentinel_hub_format: str = "image/png"
    sentinel_hub_min_zoom: int = 0
    sentinel_hub_max_zoom: int = 18
    sentinel_hub_tile_size: int = 256

    stac_render_format: Literal["geotiff", "png"] = "geotiff"
    titiler_base_url: str = "http://127.0.0.1:8000"
    stac_cache_root: str = "cache"

    @property
    def llm_base_url(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_base_url
        return self.vllm_base_url

    @model_validator(mode="after")
    def validate_ca_bundle_paths(self) -> AppSettings:
        """Fail fast on startup if configured CA bundle files are missing."""
        value = self.llm_ca_bundle_path
        if value is not None and value.strip() and not Path(value).is_file():
            raise ValueError(f"LLM_CA_BUNDLE_PATH points to a missing file: {value}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
