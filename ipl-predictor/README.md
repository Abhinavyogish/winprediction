# IPL Predictor

Clean, production-style Python project for parsing Cricsheet IPL CSVs (2008–2024), producing parquet datasets, and serving a stub FastAPI API.

## Project layout

```
ipl-predictor/
  data/
    raw/         # put Cricsheet IPL match CSVs here
    processed/   # parquet outputs
  src/
    ingestion/
      parser.py
      pipeline.py
    api/
      main.py
  tests/
```

## Setup (venv)

From the `ipl-predictor/` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the pipeline

1) Copy your Cricsheet IPL CSVs into `data/raw/` (filenames like `1082591.csv`).

2) Run:

```powershell
python -m src.ingestion.pipeline
```

Outputs:
- `data/processed/matches.parquet`
- `data/processed/deliveries.parquet`

## Run the API (local)

```powershell
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:
- `GET /health` → `{"status":"ok"}`
- `POST /predict` → `{"message":"prediction placeholder"}`

## Run tests

```powershell
pytest -q
```

## Run with Docker

From `ipl-predictor/`:

```powershell
docker compose up --build
```

Then open:
- `http://localhost:8000/health`

