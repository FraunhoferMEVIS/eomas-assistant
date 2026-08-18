# EOMAS Assistant at MEVIS

The following information is intended for internal use at Fraunhofer MEVIS. For
external users, please refer to the [README.md](README.md) file. In particula,
this file is not part of the public repository at
https://github.com/FraunhoferMEVIS/eomas-assistant.

## Running on the cluster

> [!NOTE]
> The streamlit app is constantly running on nomad:
> **https://eomas-assistant.cloud.intern.mevis.fraunhofer.de/**

If the service/job should be stopped, you can re-start it via the nomad web UI: https://cassiopeia.cloud.intern.mevis.fraunhofer.de/ui/jobs/eomas-assistant@project-eomas

or using the nomad CLI tool:

```powershell
$env:NOMAD_ADDR="https://cassiopeia.cloud.intern.mevis.fraunhofer.de"
$env:NOMAD_TOKEN="..." # has to be refreshed once a week, see https://cassiopeia.cloud.intern.mevis.fraunhofer.de/ui/settings/tokens
nomad run nomad-job.hcl
```

## Running TiTiler and Streamlit together in tmux (Linux/macOS)

```bash
cd /home/hlaue/Developer/git/EOMAS/eomas-assistant
uv sync --all-extras
tmux new-session -d -s eomas-local
tmux send-keys -t eomas-local:0.0 'cd /home/hlaue/Developer/git/EOMAS/eomas-assistant && export STAC_CACHE_ROOT=cache && export TITILER_BASE_URL=http://127.0.0.1:8000 && uv run python -m uvicorn eomas_assistant.app.titiler_app:app --host 0.0.0.0 --port 8000' C-m
tmux split-window -h -t eomas-local:0
tmux send-keys -t eomas-local:0.1 'cd /home/hlaue/Developer/git/EOMAS/eomas-assistant && export TITILER_BASE_URL=http://127.0.0.1:8000 && export STAC_CACHE_ROOT=cache && uv run streamlit run src/eomas_assistant/app/streamlit_app.py' C-m
tmux select-layout -t eomas-local:0 even-horizontal
tmux attach -t eomas-local
```

Stop both services:

```bash
tmux kill-session -t eomas-local
```
