# EOMAS Assistant

![Screencast of EOMAS Assistant with a simple example query just mentioning a location, implying a request for a recent, cloud-free, true-color image](images/simple_screencast.mp4)

Extensible agentic chat assistant for Earth Observability requests, built with:

- LangGraph for orchestration
- Python 3.11+
- Streamlit for the prototype UI
- LLM backend via vLLM / Ollama server
- Pydantic validation for LLM outputs
- Rasterio / GDAL for geospatial data processing
- Copernicus Data Space Ecosystem (CDSE) services for satellite imagery download & visualization

This repository is licensed under the [BSD 3-Clause Clear License](LICENSE).

### LLM

We currently use the [EVE-Instruct](https://huggingface.co/eve-esa/EVE-Instruct) version of [ESA's EVE LLM](https://eve.philab.esa.int/about) (earth virtual expert, fine-tuned from Mistral-Small-3.2-24B-Instruct-2506), served via a vLLM server hosted in-house (at MEVIS).

We initially used a [quantized version of EVE](https://huggingface.co/jejwalsh/EVE-Instruct-GGUF) (Q8) in GGUF format, served via [Ollama](https://ollama.com/), but that combination did not support tool calls.

Finally, it is also possible to use other LLMs, but note that the behavior varies between models.  For instance, we found that while gpt-5.6-luna often makes better (more consistent) use of the provided tools, the geography extraction is less reliable than with EVE-Instruct with the same prompt (which was developed for EVE).

-----------------

## Local execution

### Prerequisites

- Python 3.11+
- `uv` installed
- LLM service reachable from your machine

### Setup

#### 1. Install dependencies

```powershell
uv sync --all-extras
```

#### 2. Obtain credentials and keys for public STAC API

Visit https://documentation.dataspace.copernicus.eu/APIs/S3.html for instructions on how to obtain access keys for S3 downloads of satellite imagery from the Copernicus Data Space Ecosystem (CDSE).  Set up the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your `.env` file.  (You can use the `.env.example` file as a template.)

#### 3. Obtain WMTS instance ID

In order to use the [Copernicus WMTS service](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/OGC/WMTS.html), one has to register at [Copernicus Dashboard](https://shapps.dataspace.copernicus.eu/dashboard) and create a configuration (e.g. using the "Full WMS template") in the "configuration utility". Once created, the instance ID can be found under "service endpoints" (see also [the respective CDSE documentation](https://dataspace.copernicus.eu/news/2024-2-19-integrating-satellite-imagery-web-applications)). That ID has to go into our .env file:

```env
...
SENTINEL_HUB_INSTANCE_ID=<your_instance_id>
...
```

#### 4. Start TiTiler (required for STAC frame layers on the map)

```bash
uv run python -m uvicorn eomas_assistant.app.titiler_app:app --host 0.0.0.0 --port 8000
```

### Run the App

```powershell
uv run streamlit run src/eomas_assistant/app/streamlit_app.py
```

If `uv` is not installed, run directly from the local virtual environment:

Windows PowerShell:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m streamlit run src/eomas_assistant/app/streamlit_app.py
```

Linux/Unix (bash/zsh):

```bash
PYTHONPATH=src ./.venv/bin/python -m streamlit run src/eomas_assistant/app/streamlit_app.py
```

Open the shown local URL in your browser.

### Run Tests

```powershell
uv run pytest
```

Optional quality checks:

```powershell
uv run ruff check .
uv run mypy src
```

## Architecture Notes

We have initially sketched a multi-agent system architecture for EOMAS, which is
shown in the following diagram:
![Architecture](images/architecture.png)

Our current implementation is a simplified version of this architecture, and we
plan to further simplify it towards a single-agent system.  We found that a MAS
is not a good fit for this use case, where we would have to introduce a lot more
communication between the agents in order to organize the responsibilities and
not lose information.

### Workflow Graph

The LangGraph pipeline compiled in `AgentWorkflow._build_graph()` follows this control flow:

```mermaid
flowchart TD;
	__start__([<p>__start__</p>]):::first
	orchestrator(orchestrator)
	conversation(conversation)
	geography(geography)
	eo_imagery(eo_imagery)
	eo_imagery_tools(eo_imagery_tools)
	evaluator(evaluator)
	unsupported(unsupported)
	error(error)
	__end__([<p>__end__</p>]):::last
	__start__ --> orchestrator;
	conversation --> evaluator;
	eo_imagery -. &nbsp;tools&nbsp; .-> eo_imagery_tools;
	eo_imagery -. &nbsp;done&nbsp; .-> evaluator;
	eo_imagery_tools --> eo_imagery;
	error --> evaluator;
	evaluator -. &nbsp;approved&nbsp; .-> __end__;
	evaluator -. &nbsp;retry&nbsp; .-> orchestrator;
	geography -.-> eo_imagery;
	geography -. &nbsp;done&nbsp; .-> evaluator;
	orchestrator -.-> conversation;
	orchestrator -.-> error;
	orchestrator -.-> geography;
	orchestrator -.-> unsupported;
	unsupported --> evaluator;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

### State

The state passed through the graph comprises the user query, message history, and incremental agent outputs:

- The `OrchestratorAgent` creates the `plan` to answer the user request and routes the request to a fitting agent.
- The `GeographyAgent` writes the user-facing response and, when successful, the resolved `geo_location`.
- The `EOImageryAgent` creates a `AssetCatalog` with the STAC information based on the above request analysis.
- Then, it creates a `DataRequest` with the necessary information for the WMTS imagery overlay.
- Several agents may augment the `response` with things (text, map, tabular information for plotting etc.) to be rendered in the UI.
- The `EvaluatorAgent` is responsible for the `evaluation` part of the state, which is used to determine whether the request has been successfully answered or whether it should be re-routed.

## Limitations

### Missing context window management

Ideally, the app would configure/know how big the LLM context window is,
compact/trim message history, and detect when e.g. a tool call result or message
history is too big. As it stands now, depending on the LLM provider, it may cut
off exceeding tokens silently and output quality deteriorates.

### MAS Limitations

As already stated above, we found that a multi-agent system (MAS) is not only
not necessary for this prototype, but in fact counterproductive. The initial MAS
approach causes more LLM calls (higher computational cost, increased latency)
than necessary, without providing benefits in terms of modularity or
maintainability.  As it stands, we would have to introduce a lot more
communication between the agents in order to organize the responsibilities and
not lose information.
