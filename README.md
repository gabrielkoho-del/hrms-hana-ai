# HRMS HANA AI

AI Agent with DAB (Data API Builder) + SAP HANA data layer. Chat UI is LibreChat.

## Local Development

### Prerequisites

```powershell
pip install -r requirements.txt
```

### FastAPI Backend
```powershell
uvicorn agent.main:app --reload --port 8001
```

### LibreChat (Chat UI)
LibreChat is vendored under `docker/LibreChat/` (gitignored). It is pre-configured to call the backend at `http://host.docker.internal:8000/v1`.

```powershell
cd docker/LibreChat
docker compose up
```

Open http://localhost:3080. If the JWT in `librechat.yaml` is expired, regenerate it:
```powershell
python tools/generate_test_jwt.py
```

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.
