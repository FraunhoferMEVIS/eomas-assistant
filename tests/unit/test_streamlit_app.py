# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_with_apptest(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9")

    app_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "eomas_assistant"
        / "app"
        / "streamlit_app.py"
    )

    at = AppTest.from_file(str(app_path))
    at.run(timeout=5)

    assert len(at.exception) == 0
    assert len(at.title) == 1
    assert at.title[0].value == "🛰️ EOMAS Assistant"
    assert len(at.caption) == 1
    assert at.caption[0].value == (
        "Extensible agentic earth observation chat assistant"
    )
