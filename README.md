# Credit Scoring MLOps Project

Projet de deploiement et de monitoring d'un modele de scoring credit base sur Home Credit Default Risk.

Le depot couvre la chaine MLOps complete autour d'un modele de scoring :

- preparation des donnees ;
- entrainement et tracking MLflow ;
- export d'artefacts de serving ;
- API FastAPI ;
- interface Streamlit ;
- stockage des predictions ;
- monitoring technique ;
- analyse de drift ;
- benchmarks de performance ;
- tests automatises ;
- CI/CD ;
- deploiement Hugging Face Spaces.

## Objectif

Servir un score de defaut en quasi temps reel, journaliser les predictions et monitorer la solution en local et a distance.

## Stack

### Local

```text
Docker Compose
  |- FastAPI API
  |- PostgreSQL
  |- Streamlit
  |- Prometheus
  `- Grafana
```

### Distant

```text
Hugging Face Docker Space
  |- Nginx on port 7860
  |- Streamlit on /
  |- FastAPI on /api
  `- Supabase PostgreSQL logging
```

## Structure du depot

```text
src/credexp/
  config.py
  data/
  db/
  modeling/
  monitoring/
  serving/
  utils/

scripts/
  build_features.py
  train_mlflow.py
  train_final.py
  tune_optuna.py
  explainability.py
  init_db.py
  run_api.py
  monitoring_drift.py
  profile_inference.py
  benchmark_api.py
  benchmark_batching.py
  benchmark_onnx.py

tests/
  test_api.py
  test_data_io.py
  test_one_hot_encoder.py
  test_threshold.py

streamlit_app/
  app.py
  pages/

docker/
  api.Dockerfile
  prometheus.Dockerfile
  prometheus/prometheus.yml

deploy/huggingface/
  Dockerfile
  README.md
  nginx.conf
  start.sh

.github/workflows/
  ci.yml
  deploy_huggingface.yml

notebooks/
  01_build_features.ipynb
  02_eda.ipynb
  03_training_mlflow.ipynb
  04_tuning_registry_final.ipynb
  05_explainability.ipynb
  06_drift_monitoring.ipynb
  07_performance_optimization.ipynb

reports/
  coverage/
  monitoring/
  performance/
  screenshots/
  demo.md
  soutenance_marp.md

artifacts/models/
  pipeline.joblib
  threshold.json
  feature_columns.json
```

## Modele et artefacts

Le serving s'appuie sur trois artefacts minimaux :

```text
artifacts/models/pipeline.joblib
artifacts/models/threshold.json
artifacts/models/feature_columns.json
```

MLflow sert au tracking et au registry pendant l'entrainement. Le deploiement embarque ensuite les artefacts exportes pour garder une image Docker autonome.

## API FastAPI

Implementation :

```text
src/credexp/serving/api.py
```

Endpoints locaux :

| Endpoint | Methode | Role |
|---|---|---|
| `/health` | GET | Healthcheck |
| `/model-info` | GET | Metadonnees du modele charge |
| `/predict` | POST | Prediction unitaire |
| `/predict_batch` | POST | Prediction batch |
| `/metrics` | GET | Metriques Prometheus |
| `/docs` | GET | Swagger UI |

URL locale :

```text
http://127.0.0.1:8000/docs
```

URL distante :

```text
https://<owner>-credexp.hf.space/api/docs
```

Le modele est charge une seule fois au demarrage de l'API puis reutilise pour toutes les requetes.

## Interface Streamlit

Fichiers :

```text
streamlit_app/app.py
streamlit_app/pages/1_Scoring_Client.py
streamlit_app/pages/2_Monitoring_Dev.py
```

Pages disponibles :

1. `Scoring Client`
2. `Monitoring Dev`

URL locale :

```text
http://127.0.0.1:8501
```

URL distante :

```text
https://<owner>-credexp.hf.space
```

## Stockage des predictions

Table cible : `predictions`

Champs suivis :

- `request_id`
- `sk_id_curr`
- `model_name`
- `model_version`
- `threshold`
- `proba_default`
- `decision`
- `latency_ms`
- `status_code`
- `error_message`
- `input_payload`
- `output_payload`

Base locale :

```text
postgresql+psycopg://postgres:postgres@localhost:5432/credexp
```

Base distante :

```text
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/postgres?sslmode=require
```

## Monitoring

### Prometheus

```text
http://127.0.0.1:9090
http://127.0.0.1:9090/targets
```

La cible attendue est `credexp_api` en statut `UP`.

### Grafana

```text
http://127.0.0.1:3000
```

Identifiants par defaut :

```text
admin / admin
```

### Drift Evidently

Commande :

```powershell
uv run python scripts/monitoring_drift.py --limit 500
```

Sorties :

```text
reports/monitoring/evidently_drift.html
reports/monitoring/evidently_drift_meta.json
```

Notebook associe :

```text
notebooks/06_drift_monitoring.ipynb
```

## Performance

Commandes principales :

```powershell
uv run python scripts/profile_inference.py
uv run python scripts/benchmark_api.py
uv run python scripts/benchmark_batching.py
uv run python scripts/benchmark_onnx.py
```

Sorties :

```text
reports/performance/cprofile_inference_top20.txt
reports/performance/inference_benchmark.json
reports/performance/api_benchmark.json
reports/performance/batching_benchmark.json
reports/performance/onnx_benchmark.json
```

Notebook associe :

```text
notebooks/07_performance_optimization.ipynb
```

## Lancement local

Prerequis : Python 3.12, `uv`, Docker Desktop.

Installation :

```powershell
uv sync --all-groups
```

Demarrage de la stack :

```powershell
docker compose down -v
docker compose up --build -d
Start-Sleep -Seconds 30
docker compose ps
```

Generation de predictions de demo :

```powershell
1..50 | ForEach-Object { .\api_examples\test_api.ps1 }
```

Verification PostgreSQL :

```powershell
docker exec -it credexp_db psql -U postgres -d credexp -c "\dt"
docker exec -it credexp_db psql -U postgres -d credexp -c "SELECT created_at, sk_id_curr, model_version, proba_default, decision, latency_ms FROM predictions ORDER BY created_at DESC LIMIT 10;"
```

Services locaux :

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8000/docs |
| Streamlit | http://127.0.0.1:8501 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 |

## Tests et qualite

Lint :

```powershell
uv run ruff check .
uv run ruff format --check .
```

Tests :

```powershell
uv run pytest -q
```

Rapports :

```text
reports/coverage/coverage.xml
reports/coverage/html/
```

Le seuil de couverture configure dans `pyproject.toml` est de `20`.

## CI/CD

Workflow CI :

```text
.github/workflows/ci.yml
```

Declencheurs :

- `push` sur `main` et `develop`
- `pull_request` sur `main` et `develop`

Contenu :

- sync des dependances avec `uv`
- ruff check et format
- `pytest`
- validation de `docker compose`
- build de l'image `docker/api.Dockerfile`

Workflow de deploiement Hugging Face :

```text
.github/workflows/deploy_huggingface.yml
```

Declencheurs :

- `push` sur `develop`
- `workflow_dispatch`

Secrets attendus :

```text
HF_TOKEN
HF_SPACE_ID
DATABASE_URL
```


## Limites et suites

- Les donnees brutes Kaggle ne sont pas versionnees dans Git.
- Le drift reste qualitatif quand le volume de predictions est faible.
- Le deploiement Hugging Face est une preuve de concept realiste, pas une infra cloud complete.
- Les prochaines evolutions naturelles sont l'alerting, le retraining et des tests end-to-end de staging.
