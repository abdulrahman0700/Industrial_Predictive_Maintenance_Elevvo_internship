# Industrial Predictive Maintenance

An end-to-end machine learning system that predicts whether an industrial machine is about to fail, identifies **which failure mode** is likely, and explains the cause in plain language using a local LLM. Built during the **Elevvo AI/ML Engineering internship** on the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), following the Google **PACE** framework (Plan, Analyze, Construct, Execute).

---

## Highlights

- **Two-stage prediction**
  1. A LightGBM classifier predicts *failure / no failure*.
  2. If a failure is predicted, an XGBoost multi-output classifier flags the failure type: **TWF, HDF, PWF, OSF, RNF**.
- **LLM explanation**: a small Ollama model (`qwen3:0.6b`, via LangChain) writes a two-sentence engineering explanation of the likely cause.
- **FastAPI backend** with strict Pydantic validation of sensor ranges.
- **PostgreSQL persistence** of every reading and every prediction (SQLAlchemy).
- **Streamlit UI** with a prediction form and a dashboard tab.
- **MLflow** experiment tracking (parameters, metrics, confusion matrix, model artifacts).
- **Fully containerised** with Docker Compose (Postgres, Ollama, API, UI).
- **Unit tests** with pytest.

---

## Architecture

```
┌─────────────┐   POST /predict/   ┌──────────────┐        ┌──────────────┐
│ Streamlit UI│ ─────────────────► │   FastAPI    │ ─────► │  PostgreSQL  │
│   :8501     │ ◄───────────────── │    :8000     │        │ (readings +  │
└─────────────┘   JSON response    │              │        │ predictions) │
                                   │  LightGBM    │        └──────────────┘
                                   │  XGBoost     │
                                   │      │       │        ┌──────────────┐
                                   │      └───────┼──────► │    Ollama    │
                                   └──────────────┘        │ (LLM explain)│
                                                           └──────────────┘
```

**Prediction flow** (`POST /predict/`):

1. Validate the payload against the sensor schema.
2. Store the raw reading in the `AI4I` table.
3. Engineer features and run the failure model.
4. **No failure** → store the prediction and return `MF_Predction: 0`.
5. **Failure** → run the failure-type model, store the prediction, ask the LLM for an explanation, and return the per-type flags plus the explanation.

---

## Dataset

The AI4I 2020 dataset has **10,000 rows** of simulated milling-machine data.

| Feature | Description |
|---|---|
| `Type` | Product quality variant: `L`, `M`, or `H` |
| `Air temperature [K]` | Ambient temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Spindle speed |
| `Torque [Nm]` | Torque |
| `Tool wear [min]` | Cumulative tool usage |
| `Machine failure` | **Target**: 1 if any failure mode occurred |
| `TWF / HDF / PWF / OSF / RNF` | Tool wear, heat dissipation, power, overstrain, and random failures |

### Engineered features

| Feature | Formula |
|---|---|
| `Power_W` | `Torque × (rpm × 2π / 60)` |
| `strain` | `Tool_wear × Torque` |
| `Temperature_difference` | `Process_temperature − Air_temperature` |

---

## Results

Figures are from the project notebooks (80/20 train/test split, `random_state=42`).

**Failure vs. no failure (LightGBM, `class_weight='balanced'`)**

- Recall on failures: **~94%**; healthy machines correctly identified: **~99%**
- 25 false negatives on the test split
- Top predictors: `Tool_wear` and `strain` (~44% of model importance together)

**Failure type (XGBoost, multi-output)**

| Failure type | Precision | Recall | F1 |
|---|---|---|---|
| TWF (tool wear) | 1.00 | 0.70 | 0.83 |
| HDF (heat dissipation) | 1.00 | 1.00 | 1.00 |
| PWF (power) | 1.00 | 1.00 | 1.00 |
| OSF (overstrain) | 0.97 | 1.00 | 0.98 |
| RNF (random) | 0.00 | 0.00 | 0.00 (no test support) |

Weighted F1 across failure types: **0.94**. `RNF` is essentially random in the source data, so it is not meaningfully predictable.

---

## Project structure

```
.
├── api/                    # FastAPI app
│   ├── main.py             # app + data endpoints
│   └── routes/             # health, predict, explain (LLM)
├── app/streamlit_app.py    # Streamlit UI
├── src/
│   ├── config.py           # pydantic-settings configuration
│   ├── data_preparation/   # preprocessing + feature engineering
│   ├── database/           # SQLAlchemy engine and models
│   ├── models/             # training (MLflow), prediction, metrics
│   └── schemas/            # Pydantic validation schemas
├── LLM/                    # Ollama client + prompt template
├── Notebooks/              # PACE notebooks (01_Plan → 04_Execute)
├── data/                   # raw, processed, and sample data
├── models_artifacts/       # trained models + MLflow artifacts
├── docker/                 # Dockerfiles + requirements
├── tests/                  # pytest suite
├── docker-compose.yml
└── docker.env              # environment for Docker Compose
```

---

## Getting started

### Option 1: Docker Compose (recommended)

Requirements: Docker with Compose v2.

```bash
docker compose --env-file docker.env up --build
```

This starts PostgreSQL, Ollama, a one-shot job that pulls the LLM, the API, and the UI. The first run downloads the Ollama model, so the explanation may be slow or unavailable for the first minute or two.

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

Stop everything with `docker compose down` (add `-v` to also delete the database and model volumes).

### Option 2: Run locally

Requirements: Python 3.12, a running PostgreSQL instance, and [Ollama](https://ollama.com) with the model pulled.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r docker/requirements.api.txt -r docker/requirements.ui.txt

ollama pull qwen3:0.6b

# Point the app at your own database (defaults assume the Docker hostname "postgres")
export DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost:5432/DBNAME

uvicorn api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

### Configuration

Settings live in `src/config.py` and can be overridden through environment variables or a `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | Postgres on host `postgres` | SQLAlchemy connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL_NAME` | `qwen3:0.6b` | Model used for explanations |
| `API_URL` (UI only) | `http://localhost:8000` | Where the Streamlit app finds the API |

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health/` | Health check |
| `POST` | `/predict/` | Predict failure, failure type, and explanation |
| `GET` | `/data/{id}` | One stored sensor reading |
| `GET` | `/all/data` | All stored sensor readings |
| `GET` | `/all/Predictiondata` | All stored predictions |

### Example request

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "Product_ID": "M14860",
    "Type": "M",
    "Air_temperature__K": 298.1,
    "Process_temperature__K": 308.6,
    "Rotational_speed__rpm": 1551,
    "Torque__Nm": 42.8,
    "Tool_wear__min": 0
  }'
```

### Responses

No failure predicted:

```json
{ "Status": 200, "MF_Predction": 0 }
```

Failure predicted:

```json
{
  "Status": 200,
  "Tool Wear Failures": 1,
  "Heat Disspation Failures": 0,
  "Power Failures": 0,
  "Overstrain Failures": 0,
  "Random Failures": 0,
  "LLM_Explaination_Failure": "..."
}
```

### Input validation

| Field | Constraint |
|---|---|
| `Product_ID` | string, 6–8 characters |
| `Type` | single character (`L`, `M`, `H`) |
| `Air_temperature__K` | 288.0 – 312.0 |
| `Process_temperature__K` | 299.0 – 322.0 |
| `Rotational_speed__rpm` | 885 – 2175 |
| `Torque__Nm` | 2.5 – 77.5 |
| `Tool_wear__min` | 0 – 250 |

Out-of-range input returns `422 Unprocessable Entity`.

---

## Training and experiment tracking

The notebooks in `Notebooks/` document the full PACE workflow:

| Notebook | Stage |
|---|---|
| `01_Plan.ipynb` | Business problem, stakeholders, scope |
| `02_Analyze.ipynb` | Exploratory data analysis |
| `03_Construct_v1/v2.ipynb` | Modelling (failure and failure type) |
| `04_Execute.ipynb` | Results, business recommendations, limitations |

Production-style training code is in `src/models/train.py`. It builds a scikit-learn `Pipeline` (one-hot encoding + standard scaling + LightGBM) and logs parameters, recall/precision/F1/FDR, a confusion matrix, and the model to MLflow.

To browse runs locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Testing

```bash
pytest tests/ -v
```

The suite covers the health endpoint, input-schema validation, feature engineering, preprocessing, and the guarantee that the target and failure-type columns never leak into the training features.

---

## Tech stack

Python · FastAPI · Streamlit · scikit-learn · LightGBM · XGBoost · SHAP · MLflow · SQLAlchemy · PostgreSQL · LangChain · Ollama · Docker Compose · pytest

---

## Limitations and future work

- Evaluation uses a single train/test split; cross-validation would give a more reliable estimate.
- The data is simulated, so the model should be piloted on a real production line before deployment.
- The model should support maintenance prioritisation, not replace inspection schedules.
- A small LLM (0.6B parameters) gives short, sometimes generic explanations; a larger model or a retrieval step would improve quality.
- Possible extensions: time-to-failure (survival) modelling, drift monitoring, and authentication on the API.

---

## Security note

`docker.env` ships with a default database password for local development only. Change it, and keep real credentials out of version control, before deploying anywhere shared.

---

## Author

**Abdul Rahman Magdy Albasousy**: AI Engineer / Data Science, built as part of the Elevvo internship.

## License

Distributed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.
