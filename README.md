# Credit Scoring MLOps Project

Projet de déploiement et de monitoring d'un modèle de scoring crédit dans un contexte de production simulé, basé sur le dataset **Home Credit Default Risk**.

Le projet couvre le cycle de vie complet d'un modèle de machine learning : préparation des données, entraînement, tracking, versioning, déploiement local et distant, monitoring, analyse de drift, optimisation de performance, tests automatisés et CI/CD.

## 1. Objectif

L'objectif est de déployer un modèle de scoring crédit capable d'attribuer en quasi temps réel une probabilité de défaut à un client, puis de surveiller son comportement en production.

Le projet intègre :

- industrialisation du modèle ;
- API d'inférence ;
- conteneurisation ;
- stockage des prédictions ;
- monitoring technique ;
- monitoring de drift ;
- tests automatisés ;
- CI/CD ;
- déploiement distant ;
- documentation et reproductibilité.

## 2. Fonctionnalités principales

- Feature engineering Home Credit.
- Entraînement, comparaison et sélection de modèles.
- Tracking MLflow.
- Tuning Optuna.
- Entraînement final et export d'un pipeline d'inférence.
- Versioning MLflow Model Registry.
- Explainability globale et locale avec feature importance et SHAP.
- API FastAPI pour le scoring.
- Interface utilisateur Streamlit.
- Stockage PostgreSQL local.
- Stockage distant Supabase.
- Monitoring Prometheus / Grafana.
- Analyse de drift avec Evidently.
- Profiling et benchmarks de performance.
- PoC ONNX Runtime.
- Tests automatisés avec pytest et pytest-cov.
- CI/CD GitHub Actions.
- Déploiement distant Hugging Face Spaces.

## 3. Architecture

### 3.1 Stack locale

```text
Docker Compose
  ├── FastAPI API
  ├── PostgreSQL
  ├── Streamlit
  ├── Prometheus
  └── Grafana
```

Flux local :

```text
Model artifacts / MLflow registry
        ↓
FastAPI inference API
        ↓
PostgreSQL prediction logs
        ↓
Monitoring:
  - Streamlit developer dashboard
  - Prometheus / Grafana
  - Evidently data drift report
```

### 3.2 Déploiement distant

```text
Hugging Face Docker Space
  ├── Nginx public port 7860
  ├── Streamlit at /
  ├── FastAPI at /api
  ├── embedded joblib model artifacts
  └── Supabase PostgreSQL logging
```

Le déploiement distant expose une interface Streamlit, une API FastAPI documentée via Swagger, le modèle embarqué dans l'image Docker et une base Supabase pour stocker les prédictions.

## 4. Composants techniques

| Composant | Rôle |
|---|---|
| MLflow | Tracking des expériences et registry pendant l'entraînement |
| LightGBM | Modèle final retenu |
| FastAPI | API de scoring et documentation Swagger |
| Streamlit | Interface utilisateur et dashboard simplifié |
| PostgreSQL | Stockage local des prédictions |
| Supabase | Stockage PostgreSQL distant |
| Prometheus | Collecte des métriques techniques |
| Grafana | Visualisation des métriques techniques |
| Evidently | Analyse de drift |
| Docker Compose | Orchestration locale |
| Hugging Face Spaces | Déploiement distant public |
| GitHub Actions | CI/CD |
| pytest / pytest-cov | Tests automatisés et couverture |
| cProfile | Profiling |
| ONNX Runtime | PoC d'optimisation |

## 5. Structure du dépôt

```text
src/credexp/
  config.py
  data/
    io.py
    build_features.py
  modeling/
    dataset.py
    metrics.py
    pipelines.py
    preprocess.py
    train.py
    tuning.py
    threshold.py
    explainability.py
  serving/
    api.py
    model_loader.py
    schemas.py
  db/
    models.py
    session.py
    init_db.py
    crud.py
  monitoring/
    drift.py
  utils/
    logging.py

scripts/
  build_features.py
  train_mlflow.py
  train_final.py
  tune_optuna.py
  explainability.py
  init_db.py
  print_paths.py
  run_api.py
  monitoring_drift.py
  profile_inference.py
  benchmark_api.py
  benchmark_batching.py
  benchmark_onnx.py

streamlit_app/
  app.py
  pages/
    1_Scoring_Client.py
    2_Monitoring_Dev.py

docker/
  api.Dockerfile
  prometheus.Dockerfile
  prometheus/
    prometheus.yml

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
  monitoring/
  performance/
  coverage/
  screenshots/

artifacts/models/
  pipeline.joblib
  threshold.json
  feature_columns.json

api_examples/
  curl_examples.md
  test_api.ps1

docker-compose.yml
README.md
demo.md
```

## 6. Modèle retenu

Le modèle retenu est un **LightGBM** optimisé selon un coût métier personnalisé.

Étapes réalisées :

1. construction des features à partir des fichiers Home Credit ;
2. validation croisée stratifiée ;
3. prise en compte du déséquilibre de classe ;
4. optimisation du seuil de décision selon un coût métier ;
5. tuning avec Optuna ;
6. entraînement final sur l'ensemble de développement ;
7. évaluation sur un holdout jamais vu ;
8. versioning dans MLflow Model Registry ;
9. analyse d'explicabilité globale et locale.

Nom du modèle dans MLflow :

```text
credit_scoring_model
```

Version retenue :

```text
Version 2 / Production
```

## 7. Artefacts de serving

Pour le serving local et distant, le modèle est embarqué sous forme d'artefacts minimaux :

```text
artifacts/models/
  pipeline.joblib
  threshold.json
  feature_columns.json
```

- `pipeline.joblib` : pipeline complet d'inférence ;
- `threshold.json` : seuil métier optimisé ;
- `feature_columns.json` : ordre exact des features attendu par le modèle.

MLflow reste utilisé pour le tracking, la comparaison des modèles et le registry pendant la phase d'entraînement. Le déploiement utilise ensuite les artefacts exportés pour garantir un conteneur autonome et reproductible.

## 8. API FastAPI

### 8.1 Endpoints locaux

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API fonctionne |
| `/model-info` | GET | Retourne le modèle, la version, le seuil et le nombre de features |
| `/predict` | POST | Retourne une prédiction pour un client |
| `/predict_batch` | POST | Retourne des prédictions pour plusieurs clients |
| `/metrics` | GET | Expose les métriques Prometheus |
| `/docs` | GET | Documentation Swagger interactive |

Documentation locale :

```text
http://127.0.0.1:8000/docs
```

### 8.2 Endpoints distants Hugging Face

URL distante :

```text
https://<owner>-credexp.hf.space
```

Routes :

| Route | Description |
|---|---|
| `/` | Interface Streamlit |
| `/api/health` | Healthcheck FastAPI |
| `/api/docs` | Swagger FastAPI |
| `/api/predict` | Prédiction unitaire |
| `/api/predict_batch` | Prédictions batch |
| `/api/metrics` | Métriques Prometheus exposées par l'API |

### 8.3 Exemple de payload

```json
{
  "sk_id_curr": 123456,
  "features": {
    "EXT_SOURCE_1": 0.52,
    "EXT_SOURCE_2": 0.71,
    "EXT_SOURCE_3": 0.41
  }
}
```

### 8.4 Exemple de réponse

```json
{
  "proba_default": 0.3197,
  "decision": 0,
  "threshold": 0.49,
  "model_name": "credit_scoring_model",
  "model_version": "hf-space",
  "latency_ms": 46.02
}
```

## 9. PostgreSQL et Supabase

Chaque appel API peut être stocké dans la table `predictions`.

Colonnes principales :

- `id` ;
- `created_at` ;
- `request_id` ;
- `sk_id_curr` ;
- `model_name` ;
- `model_version` ;
- `threshold` ;
- `proba_default` ;
- `decision` ;
- `latency_ms` ;
- `status_code` ;
- `error_message` ;
- `input_payload` ;
- `output_payload`.

Supabase est utilisé comme base PostgreSQL distante. La connexion est configurée par variable secrète :

```text
DATABASE_URL
```

Format attendu :

```text
postgresql+psycopg://USER:PASSWORD@HOST:PORT/postgres?sslmode=require
```

Sur Hugging Face, l'utilisation du **Supabase Session Pooler** est recommandée afin d'éviter les problèmes de connexion directe IPv6.

## 10. Streamlit

Pages :

1. **Scoring Client** : interface utilisateur pour envoyer un payload à l'API et afficher le score ;
2. **Monitoring Dev** : vue synthétique des dernières prédictions, scores et latences stockés dans PostgreSQL ou Supabase.

URL locale :

```text
http://127.0.0.1:8501
```

URL remote :

```text
https://<owner>-credexp.hf.space
```

## 11. Monitoring technique

### Prometheus

```text
http://127.0.0.1:9090
http://127.0.0.1:9090/targets
```

La cible `credexp_api` doit être `UP`.

### Grafana

```text
http://127.0.0.1:3000
```

Identifiants par défaut :

```text
admin / admin
```

Datasource Prometheus :

```text
http://prometheus:9090
```

Panels recommandés :

- débit de requêtes ;
- nombre total de requêtes ;
- statuts HTTP ;
- latence moyenne de l'API.

## 12. Monitoring de drift

Le drift est analysé avec Evidently.

Sources utilisées :

- référence : `data/processed/reference.parquet` ;
- production : inputs stockés dans PostgreSQL ou Supabase via les appels API.

Commande :

```powershell
uv run python scripts/monitoring_drift.py --limit 500
```

Rapport généré :

```text
reports/monitoring/evidently_drift.html
```

Notebook associé :

```text
notebooks/06_drift_monitoring.ipynb
```

## 13. Performance et optimisation

Commandes :

```powershell
uv run python scripts/profile_inference.py
uv run python scripts/benchmark_api.py
uv run python scripts/benchmark_batching.py
uv run python scripts/benchmark_onnx.py
```

Rapports :

```text
reports/performance/
```

Conclusions principales :

- la latence API est faible ;
- les principaux coûts viennent de la validation sklearn, du preprocessing et des conversions pandas/numpy ;
- le batching fournit le plus fort gain pratique ;
- ONNX Runtime a été validé comme optimisation PoC sur l'estimateur LightGBM ;
- la configuration finale conserve le pipeline sklearn pour la robustesse, tout en exposant un endpoint batch.

Notebook associé :

```text
notebooks/07_performance_optimization.ipynb
```

## 14. Lancement local complet

Pré-requis : Python 3.12, `uv`, Docker Desktop, Git.

Installation :

```powershell
uv sync --all-groups
```

Lancement :

```powershell
docker compose down -v
docker compose up --build -d
Start-Sleep -Seconds 30
docker compose ps
```

Génération de prédictions de démonstration :

```powershell
1..50 | ForEach-Object { .\api_examples\test_api.ps1 }
```

Vérification PostgreSQL :

```powershell
docker exec -it credexp_db psql -U postgres -d credexp -c "\dt"
```

```powershell
docker exec -it credexp_db psql -U postgres -d credexp -c "SELECT created_at, sk_id_curr, model_version, proba_default, decision, latency_ms FROM predictions ORDER BY created_at DESC LIMIT 10;"
```

Interfaces locales :

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8000/docs |
| Streamlit | http://127.0.0.1:8501 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 |
| Prometheus targets | http://127.0.0.1:9090/targets |

## 15. Déploiement distant Hugging Face

Fichiers principaux :

```text
deploy/huggingface/Dockerfile
deploy/huggingface/README.md
deploy/huggingface/nginx.conf
deploy/huggingface/start.sh
```

Routage :

```text
/              -> Streamlit
/api/health    -> FastAPI healthcheck
/api/docs      -> Swagger
/api/predict   -> prédiction
/api/metrics   -> métriques
```

Le déploiement est automatisé par :

```text
.github/workflows/deploy_huggingface.yml
```

Secrets nécessaires :

```text
HF_TOKEN
HF_SPACE_ID
DATABASE_URL
```

## 16. Tests, qualité et coverage

Lint :

```powershell
uv run ruff check .
uv run ruff format --check .
```

Tests :

```powershell
uv run pytest -q
```

La configuration génère :

```text
reports/coverage/coverage.xml
reports/coverage/html/
```

Un seuil minimal de couverture est défini à 20 %. Le coverage observé est d'environ 27 %. Ce seuil est volontairement modéré car le dépôt contient du code expérimental, des notebooks et des scripts de training. Les tests ciblent prioritairement les composants critiques de production : API, validation d'entrée, I/O, schémas et logique métier.

## 17. CI/CD

Le projet contient deux workflows GitHub Actions.

### CI générale

```text
.github/workflows/ci.yml
```

Ce workflow :

- se déclenche sur `push` et `pull_request` ;
- installe Python et `uv` ;
- synchronise les dépendances ;
- exécute Ruff ;
- vérifie le formatage ;
- lance Pytest avec coverage ;
- valide Docker Compose ;
- construit l'image Docker locale de l'API ;
- upload le rapport de couverture comme artifact.

### Déploiement Hugging Face

```text
.github/workflows/deploy_huggingface.yml
```

Ce workflow :

- se déclenche sur `push` vers `develop` et manuellement via `workflow_dispatch` ;
- relance les contrôles qualité ;
- relance les tests avec coverage ;
- vérifie les artefacts modèle ;
- construit l'image Docker Hugging Face ;
- prépare le dossier du Space ;
- déploie automatiquement vers Hugging Face Spaces avec `HF_TOKEN`.

## 18. Captures d'écran

Captures recommandées :

```text
01_github_history.png
02_github_actions_success.png
03_fastapi_docs.png
04_fastapi_predict_response.png
05_streamlit_scoring.png
06_streamlit_monitoring.png
07_postgres_predictions.png
08_prometheus_target_up.png
09_grafana_dashboard.png
10_evidently_drift_report.png
11_mlflow_registry_model_v2.png
12_performance_notebook.png
13_onnx_benchmark_json.png
14_supabase_predictions.png
15_huggingface_space_streamlit.png
16_huggingface_space_api_docs.png
17_github_action_deploy_hf_success.png
18_pytest_coverage_report.png
19_supabase_prediction_from_hf.png
```

## 19. Contenu principal

- historique Git ;
- scripts API ;
- Dockerfile ;
- Docker Compose ;
- tests automatisés ;
- pipeline CI/CD ;
- déploiement Hugging Face ;
- notebooks de monitoring et performance ;
- rapport Evidently ;
- screenshots de PostgreSQL, Supabase, Prometheus, Grafana, FastAPI, Streamlit, GitHub Actions et Hugging Face ;
- README et documentation de démonstration.

## 20. Points de vigilance et limites

- Les données Kaggle brutes ne sont pas versionnées dans Git.
- Les secrets ne sont jamais commités.
- Le drift est interprété qualitativement lorsque peu de prédictions sont disponibles.
- Le seuil de coverage est volontairement modéré.
- Le déploiement Hugging Face est une démonstration réaliste, mais ne remplace pas une infrastructure cloud production complète avec autoscaling et alerting avancé.

## 21. Perspectives

- Alertes Prometheus/Grafana.
- Monitoring de performance modèle sur données labellisées récentes.
- Retraining automatisé.
- Séparation staging/production.
- Versioning avancé des jeux de référence.
- Tests end-to-end sur environnement de staging.

## Auteur

Benoît Girard
