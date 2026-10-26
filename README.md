# 🛠️ Industrial Predictive Maintenance

> An end-to-end machine-learning system that predicts whether an industrial machine is about to fail, identifies **which type of failure** is likely, and explains the result in plain English with a local LLM.
> Built as part of the **Elevvo Internship**.

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Dataset](#-dataset)
7. [Modeling Approach & Results](#-modeling-approach--results)
8. [Getting Started (Local)](#-getting-started-local)
9. [API Reference](#-api-reference)
10. [Testing](#-testing)
11. [Deploying on AWS](#-deploying-on-aws)
12. [Limitations & Future Work](#-limitations--future-work)
13. [License](#-license)

---

## 🔍 Overview

A single unplanned machine failure can cost a factory heavily in downtime and emergency repairs. This project turns raw sensor readings into an actionable early warning:

1. **Stage 1 – Will the machine fail?** A binary classifier (LightGBM) predicts failure vs. no failure.
2. **Stage 2 – Why will it fail?** If a failure is predicted, a multi-output classifier (XGBoost) flags the likely failure modes: **TWF** (Tool Wear), **HDF** (Heat Dissipation), **PWF** (Power), **OSF** (Overstrain), **RNF** (Random).
3. **Stage 3 – Explain it.** A small local LLM (Ollama, `qwen3:0.6b`) writes a short, plain-language explanation of the likely cause.

Every request and prediction is stored in PostgreSQL, and a Streamlit UI provides a prediction form and a dashboard.

The project followed the **PACE framework** (Plan → Analyze → Construct → Execute); each stage has its own notebook.

> **Project goal:** not just raw accuracy, but keeping the **False Discovery Rate (FDR)** low so that false alarms don't halt production unnecessarily.

---

## ✨ Key Features

- **Two-stage ML pipeline** — failure detection followed by failure-type diagnosis.
- **Engineered features** — `Power_W`, `strain` (torque × tool wear) and `Temperature_difference`.
- **Imbalance-aware training** — cost-sensitive learning (`class_weight='balanced'`) for a 93.6 % / 6.4 % class split.
- **Experiment tracking** with MLflow.
- **Explainability** — SHAP analysis in the notebooks, plus LLM-generated explanations at inference time.
- **FastAPI backend** with strict Pydantic input validation (sensor ranges are enforced).
- **Persistence** — sensor readings and predictions saved to PostgreSQL via SQLAlchemy.
- **Streamlit UI** — prediction form and a monitoring dashboard.
- **Fully containerized** — one `docker compose up` starts Postgres, Ollama, the API and the UI.

---

## 🏗️ Architecture

```
                ┌─────────────────────┐
   User ──────▶ │  Streamlit UI :8501 │
                └──────────┬──────────┘
                           │ HTTP (API_URL)
                           ▼
                ┌─────────────────────┐        ┌──────────────────────┐
                │   FastAPI API :8000 │ ─────▶ │  Ollama (qwen3:0.6b) │
                │  • validation       │  LLM   └──────────────────────┘
                │  • feature eng.     │
                │  • LightGBM  (fail?)│        ┌──────────────────────┐
                │  • XGBoost (type?)  │ ─────▶ │   PostgreSQL :5432   │
                └─────────────────────┘  SQL   └──────────────────────┘
```

All four services run on a single Docker Compose network. Only ports **8000** (API) and **8501** (UI) are published to the host.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn, LightGBM, XGBoost, imbalanced-learn, SHAP |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL 16, SQLAlchemy |
| LLM | Ollama (`qwen3:0.6b`), LangChain |
| UI | Streamlit |
| DevOps | Docker, Docker Compose, pytest |

---

## 📁 Project Structure

```
.
├── api/                      # FastAPI application
│   ├── main.py               # App entry point + data endpoints
│   └── routes/
│       ├── health.py         # GET  /health/
│       ├── predict.py        # POST /predict/
│       └── explain.py        # LLM explanation helper
├── app/
│   └── streamlit_app.py      # Streamlit UI (Predict + Dashboard tabs)
├── LLM/
│   ├── llmModel.py           # Ollama chat-model wrapper
│   └── prompt.py             # Prompt template
├── src/
│   ├── config.py             # Central settings (pydantic-settings)
│   ├── data_preparation/     # preprocessing + feature engineering
│   ├── models/               # train.py, train1.py, predict.py, metrics.py
│   ├── database/             # SQLAlchemy models + connection
│   └── schemas/              # Pydantic request/response schemas
├── Notebooks/                # PACE notebooks (Plan, Analyze, Construct v1/v2, Execute)
├── data/
│   ├── Row_data/             # Original AI4I dataset
│   ├── processed/            # Cleaned datasets
│   └── Sample/               # 2,000-row high-failure sample for testing
├── models_artifacts/         # Trained models (model.pkl, model1.pkl, ...) + MLflow info
├── docker/                   # Dockerfile.api, Dockerfile.ui, requirements
├── tests/                    # pytest suite
├── docker-compose.yml
├── docker.env                # Compose environment variables
└── LICENSE
```

---

## 📊 Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 rows, 14 columns.

| Feature | Description |
|---|---|
| `Type` | Product quality variant: `L`, `M`, `H` |
| `Air temperature [K]` | 288 – 312 |
| `Process temperature [K]` | 299 – 322 |
| `Rotational speed [rpm]` | 885 – 2175 |
| `Torque [Nm]` | 2.5 – 77.5 |
| `Tool wear [min]` | 0 – 250 |
| `Machine failure` | Target: 1 = failure |
| `TWF / HDF / PWF / OSF / RNF` | Individual failure modes |

**Key EDA findings**

- Strong linear relationship between air and process temperature.
- Heavily imbalanced target: **93.6 %** no-failure vs. **6.4 %** failure.
- Most failures are tool-wear, overstrain or heat-dissipation related, and multiple failure modes can occur at once.
- Failures cluster at high torque (~43–77 Nm) and high tool wear (~160–249 min).
- By machine type, `L` accounts for ~71.8 % of failures, `M` ~20.8 %, `H` ~7.4 %.

---

## 🤖 Modeling Approach & Results

**Feature engineering**

```
Power_W                = Torque × Rotational speed × (2π / 60)
strain                 = Tool wear × Torque
Temperature_difference = Process temperature − Air temperature
```

**Preprocessing:** `OneHotEncoder` for `Type`, `StandardScaler` for numeric features, wrapped in a scikit-learn `ColumnTransformer` + `Pipeline`. 80/20 train/test split (`random_state=42`).

### Stage 1 — Machine failure (LightGBM, test set)

| Metric | Value |
|---|---|
| Accuracy | 98.4 % |
| Precision | 94.1 % |
| Recall | 81.7 % |
| F1-score | 87.5 % |
| False Discovery Rate | ~5.9 % |

LightGBM was compared against Logistic Regression, Random Forest and XGBoost in `03_Construct_v1.ipynb`.

### Stage 2 — Failure type (XGBoost, multi-output, test set, weighted averages)

| Metric | Value |
|---|---|
| Precision | 98.6 % |
| Recall | 90.8 % |
| F1-score | 93.9 % |

The model struggles with **Random Failures (RNF)** (F1 = 0) — these are, by definition, not predictable from sensor patterns.

### What drives the predictions (SHAP)

`Tool wear` and the engineered `strain` feature are the top two predictors, followed by `Temperature_difference` and `Rotational speed`.

> ⚠️ Training accuracy is ~100 % vs. 98.4 % test accuracy, which hints at some overfitting. Cross-validation is listed under [Future Work](#-limitations--future-work).

---

## 🚀 Getting Started (Local)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2
- ~4 GB of free RAM (Ollama + Postgres + API)

### Option A — Docker Compose (recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Review / edit the environment file
cat docker.env

# 3. Build and start everything
docker compose --env-file docker.env up -d --build
```

On first start, the `ollama-pull` service downloads the LLM (`qwen3:0.6b`) and exits — this can take a minute.

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API (Swagger docs) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health/ |

Stop everything with `docker compose --env-file docker.env down` (add `-v` to also delete the database and model volumes).

### Option B — Run without Docker

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r docker/requirements.api.txt -r docker/requirements.ui.txt
```

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg2://postgres:<password>@localhost:5432/postgres
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=qwen3:0.6b
```

You need a running PostgreSQL instance and a local [Ollama](https://ollama.com) with the model pulled (`ollama pull qwen3:0.6b`). Then:

```bash
uvicorn api.main:app --reload --port 8000          # terminal 1
API_URL=http://localhost:8000 streamlit run app/streamlit_app.py   # terminal 2
```

### Retraining the models

The training classes live in `src/models/train.py` (failure) and `src/models/train1.py` (failure type). Runs are tracked with MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health/` | Liveness check |
| `POST` | `/predict/` | Predict failure, failure types and LLM explanation |
| `GET` | `/data/{id}` | Fetch one stored sensor record |
| `GET` | `/all/data` | All stored sensor records |
| `GET` | `/all/Predictiondata` | All stored predictions |

**Example request**

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

**Response — no failure predicted**

```json
{ "Status": 200, "MF_Predction": 0 }
```

**Response — failure predicted**

```json
{
  "Status": 200,
  "Tool Wear Failures": 1,
  "Heat Disspation Failures": 0,
  "Power Failures": 0,
  "Overstrain Failures": 1,
  "Random Failures": 0,
  "LLM_Explaination_Failure": "..."
}
```

**Input validation ranges** (invalid values return HTTP `422`)

| Field | Range |
|---|---|
| `Product_ID` | 6–8 characters |
| `Type` | exactly 1 character (`L`/`M`/`H`) |
| `Air_temperature__K` | 288.0 – 312.0 |
| `Process_temperature__K` | 299.0 – 322.0 |
| `Rotational_speed__rpm` | 885 – 2175 |
| `Torque__Nm` | 2.5 – 77.5 |
| `Tool_wear__min` | 0 – 250 |

---

## 🧪 Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

The suite covers the health endpoint, Pydantic validation rules, feature-engineering formulas, column renaming, null counting, and a check that the target and ID columns never leak into the training features.

---

## ☁️ Deploying on AWS

This section walks through deploying the full stack (Postgres + Ollama + API + UI) on a single **Amazon EC2** instance using Docker Compose. It is the simplest path because the project already ships a working `docker-compose.yml`. A managed alternative (ECR + ECS + RDS) is outlined afterwards.

### Deployment architecture

```
Internet
   │
   ▼
┌──────────────────── AWS VPC ─────────────────────────┐
│  Security Group: 22 (your IP), 8501, 8000            │
│                                                      │
│   ┌────────────── EC2 (Ubuntu 22.04/24.04) ───────┐  │
│   │  Docker Compose                               │  │
│   │   ├─ ui      :8501  ──▶ api                   │  │
│   │   ├─ api     :8000  ──▶ postgres, ollama      │  │
│   │   ├─ postgres (internal only)  ── EBS volume  │  │
│   │   └─ ollama   (internal only)  ── EBS volume  │  │
│   └───────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Step 0 — Prerequisites

- An AWS account and an IAM user (or role) with permission to use EC2.
- An EC2 **key pair** (`.pem` file) for SSH access.
- Your project pushed to a Git repository (GitHub/GitLab), or the folder ready to copy with `scp`.

### Step 1 — Launch an EC2 instance

In the **EC2 console → Launch instance**:

| Setting | Recommended value |
|---|---|
| Name | `predictive-maintenance` |
| AMI | Ubuntu Server 22.04 or 24.04 LTS |
| Instance type | `t3.large` (2 vCPU, 8 GB RAM) — minimum `t3.medium` (4 GB) |
| Key pair | Your existing key pair |
| Storage | 30 GB gp3 (images, Postgres data and the Ollama model) |

> 💡 The `qwen3:0.6b` model is tiny, so CPU-only instances are enough. A GPU instance is **not** required.

### Step 2 — Configure the Security Group

Add these **inbound rules**:

| Type | Port | Source | Purpose |
|---|---|---|---|
| SSH | 22 | **My IP** | Admin access |
| Custom TCP | 8501 | `0.0.0.0/0` (or your IP / office range) | Streamlit UI |
| Custom TCP | 8000 | **My IP** only (optional) | API / Swagger docs |

> 🔒 **Never** open ports `5432` (Postgres) or `11434` (Ollama). The compose file deliberately does not publish them, so they stay reachable only inside the Docker network.

### Step 3 — Connect and install Docker

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

```bash
# Install Docker Engine + Compose plugin
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh

# Run docker without sudo (log out and back in afterwards)
sudo usermod -aG docker $USER
exit
```

Reconnect over SSH and verify:

```bash
docker --version
docker compose version
```

### Step 4 — Get the code onto the server

```bash
git clone <your-repo-url> predictive-maintenance
cd predictive-maintenance
```

(Or copy it from your machine: `scp -i your-key.pem -r ./project ubuntu@<EC2_PUBLIC_IP>:~/predictive-maintenance`.)

### Step 5 — Set production secrets

Edit `docker.env` **before** the first start and replace the demo password:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<a-long-random-password>
POSTGRES_DB=postgres
OLLAMA_MODEL="qwen3:0.6b"
```

Generate a strong password with `openssl rand -base64 24`, and **don't commit** this file to a public repository. The `postgres` volume only reads `POSTGRES_PASSWORD` the first time it initializes, so set it now.

### Step 6 — Build and start the stack

```bash
docker compose --env-file docker.env up -d --build
```

Check that everything is running:

```bash
docker compose ps
docker compose logs -f api          # Ctrl+C to exit
docker compose logs ollama-pull     # should end after the model downloads
```

The first start takes a few minutes (image builds + model download).

### Step 7 — Verify

```bash
curl http://localhost:8000/health/
# {"status":"Healthy","code":200}
```

From your browser:

- **UI:** `http://<EC2_PUBLIC_IP>:8501`
- **API docs:** `http://<EC2_PUBLIC_IP>:8000/docs`

### Step 8 — Make it production-ready

**Keep the IP stable.** Allocate an **Elastic IP** (EC2 → Elastic IPs → Allocate → Associate) so the address survives reboots.

**Auto-start on reboot.** The compose services use `restart: unless-stopped`, and Docker starts on boot, so the stack comes back automatically:

```bash
sudo systemctl enable docker
```

**Domain + HTTPS (optional but recommended).** Point a DNS record (e.g., Route 53) at the Elastic IP and put Nginx in front of the UI with a free Let's Encrypt certificate:

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

```nginx
# /etc/nginx/sites-available/pm-app
server {
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;     # Streamlit uses WebSockets
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/pm-app /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d your-domain.com
```

Once HTTPS is working, open **443** (and 80) in the Security Group and **close 8501**.

**Backups.** Take periodic EBS snapshots (or AWS Backup) of the instance volume, or dump the database:

```bash
docker compose exec postgres pg_dump -U postgres postgres > backup_$(date +%F).sql
```

### Updating the deployment

```bash
cd predictive-maintenance
git pull
docker compose --env-file docker.env up -d --build
```

### Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `.dockerIgnore` seems ignored, build is slow | Docker looks for a file named **`.dockerignore`** (lowercase). On Linux (case-sensitive) the current `.dockerIgnore` is not picked up — rename it: `git mv .dockerIgnore .dockerignore`. |
| Variables show up blank / "variable is not set" warning | Compose only reads `.env` automatically. Always pass `--env-file docker.env`, or rename the file to `.env`. |
| `api` container exits or restarts | Check `docker compose logs api`. Usually Postgres isn't healthy yet or `DATABASE_URL` credentials don't match `docker.env`. |
| Prediction takes very long or times out | First LLM call loads the model into memory. Use at least 4 GB RAM and confirm `ollama-pull` finished (`docker compose logs ollama-pull`). |
| UI says "API unreachable" | Check `docker compose ps` and `curl localhost:8000/health/` on the server. |
| Out of disk space | `docker system prune -af` (careful: removes unused images) or enlarge the EBS volume. |
| Can't open the UI from the browser | Security Group is missing the inbound rule for port 8501. |

### Estimated cost (rough, `us-east-1`, on-demand)

| Resource | Approx. monthly |
|---|---|
| `t3.large` running 24/7 | ~$60 |
| 30 GB gp3 EBS | ~$2.50 |
| Elastic IP (attached to a running instance) | Free (charged only when idle/extra) |

Stop the instance when you're not using it to avoid compute charges. Check the [AWS Pricing Calculator](https://calculator.aws/) for current prices in your region.

### Alternative: managed deployment (ECR + ECS Fargate + RDS)

For a more scalable, production-grade setup:

| Component | AWS service |
|---|---|
| Container images (`api`, `ui`) | **Amazon ECR** |
| Running containers | **Amazon ECS on Fargate** (one service each for `api` and `ui`) |
| PostgreSQL | **Amazon RDS for PostgreSQL** (set `DATABASE_URL` accordingly) |
| Ollama | An ECS service/EC2 instance with enough memory, reachable privately by the API via `OLLAMA_BASE_URL` |
| Secrets | **AWS Secrets Manager** / SSM Parameter Store (instead of `docker.env`) |
| Traffic | **Application Load Balancer** + ACM certificate for HTTPS |

Outline:

```bash
# Authenticate Docker to ECR and push the images
aws ecr create-repository --repository-name pm-api
aws ecr create-repository --repository-name pm-ui
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

docker build -f docker/Dockerfile.api -t <account-id>.dkr.ecr.<region>.amazonaws.com/pm-api:latest .
docker build -f docker/Dockerfile.ui  -t <account-id>.dkr.ecr.<region>.amazonaws.com/pm-ui:latest .
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/pm-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/pm-ui:latest
```

Then create ECS task definitions that pass `DATABASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL_NAME` (API) and `API_URL` (UI), place them in private subnets behind an ALB, and restrict RDS to the API's security group only.

### Security checklist

- [ ] Replaced the default `POSTGRES_PASSWORD`
- [ ] `docker.env` / `.env` are **not** committed to Git
- [ ] Ports 5432 and 11434 are **not** exposed
- [ ] SSH restricted to your IP (or use AWS Systems Manager Session Manager)
- [ ] HTTPS enabled and port 8501 closed to the public
- [ ] Regular EBS snapshots / database backups

---

## 🔭 Limitations & Future Work

- Only ~638 failure examples exist, so failure-type models are data-starved (especially RNF).
- Evaluation uses a single train/test split — add **cross-validation** and early stopping for a more reliable estimate and to reduce overfitting.
- Try **SMOTE** or threshold tuning to improve recall (25 false negatives on the test set).
- Add authentication and rate limiting to the API.
- Add database migrations (Alembic) instead of `create_all`.
- Move from tabular snapshots to **time-series** data for time-to-failure (survival) analysis.
- Add CI/CD (GitHub Actions → ECR → ECS) for automated deployments.
- Keep a human in the loop for maintenance decisions; treat predictions as prioritization, not replacement, of inspections.

---

## 📄 License

Distributed under the license in the [`LICENSE`](LICENSE) file.

---

## 🙏 Acknowledgements

- Dataset: *AI4I 2020 Predictive Maintenance Dataset* (UCI Machine Learning Repository).
- Built during the **Elevvo** internship program.
