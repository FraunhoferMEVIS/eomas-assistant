# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.integration
def test_live_streamlit_app_accepts_chat_input_and_completes_request() -> None:
    app_path = (
        Path(__file__).resolve().parents[2] / "src" / "eomas_assistant" / "app" / "streamlit_app.py"
    )
    user_prompt = "show me lyon in april 2020"

    at = AppTest.from_file(str(app_path))
    at.run(timeout=10)
    at.chat_input[0].set_value(user_prompt)
    at.run(timeout=120)

    assert len(at.exception) == 0
    assert at.session_state.chat_history[0]["role"] == "user"
    assert at.session_state.chat_history[0]["content"] == user_prompt
    assert at.session_state.chat_history[1]["role"] == "assistant"

    response = at.session_state.chat_history[1]["content"]
    print(f"Assistant response metadata: {response.metadata}")
    assert response.metadata["query"] == user_prompt
    assert response.metadata.get("route") != "error"
    assert response.items
