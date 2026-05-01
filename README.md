# Credit Scoring MLOps Project

Projet de déploiement et de monitoring d'un modèle de scoring crédit dans un contexte de production simulé.

Ce projet a été réalisé dans le cadre d'un exercice MLOps autour du dataset **Home Credit Default Risk**. Il couvre l'ensemble du cycle de vie d'un modèle de machine learning, depuis la préparation des données jusqu'au déploiement local, au monitoring et à l'analyse de performance.

## 1. Objectif du projet

L'objectif est de déployer un modèle de scoring crédit capable d'attribuer en quasi temps réel une probabilité de défaut à un client, puis de surveiller son comportement en production.

Le projet ne se limite pas à la modélisation. Il intègre aussi les aspects d'industrialisation, de monitoring, de qualité logicielle, de conteneurisation et de reproductibilité.

## 2. Fonctionnalités principales

Le projet inclut :

- feature engineering sur les données Home Credit ;
- entraînement et sélection de modèles ;
- tracking des expérimentations avec MLflow ;
- tuning avec Optuna ;
- entraînement final et export d'un pipeline d'inférence ;
- versioning du modèle dans MLflow Model Registry ;
- explainability globale et locale avec feature importance et SHAP ;
- API FastAPI pour le scoring ;
- interface utilisateur Streamlit ;
- stockage des prédictions dans PostgreSQL ;
- monitoring technique avec Prometheus et Grafana ;
- analyse de data drift avec Evidently ;
- profiling et benchmarks de performance ;
- benchmark ONNX Runtime ;
- tests automatisés et pipeline CI/CD GitHub Actions.

## 3. Architecture générale

Flux principal :

```text
MLflow Model Registry
        ↓
FastAPI inference API
        ↓
PostgreSQL prediction logs
        ↓
Monitoring:
  - Prometheus / Grafana for technical metrics
  - Evidently for data drift
  - Streamlit for user and developer dashboards
```

Composants :

| Composant | Rôle |
|---|---|
| MLflow | Tracking des expériences et registre du modèle |
| FastAPI | API de scoring et documentation Swagger |
| PostgreSQL | Stockage des inputs, outputs, latence et métadonnées |
| Streamlit | Interface utilisateur et monitoring simplifié |
| Prometheus | Collecte des métriques techniques exposées par l'API |
| Grafana | Visualisation des métriques techniques |
| Evidently | Analyse de dérive des données |
| Docker Compose | Orchestration locale de la stack complète |
| GitHub Actions | Tests, lint, format et build Docker en CI |

## 4. Structure du dépôt

```text
src/credexp/
  config.py
  data/
    io.py
    build_features.py
  modeling/
    dataset.py
    export.py
    metrics.py
    pipelines.py
    preprocess.py
    train.py
    tuning.py
    threshold.py
    explainability.py
  serving/
    api.py
    inference.py
    model_loader.py
    schemas.py
  db/
    models.py
    session.py
    init_db.py
    crud.py
  monitoring/
    drift.py
    metrics.py
  utils/
    logging.py
    profiling.py

scripts/
  build_features.py
  export_inference_artifacts.py
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
  screenshots/

data/
  raw/
  processed/

artifacts/
  models/

mlflow/
  mlflow.db

demo.md
```

## 5. Modèle retenu

Le modèle retenu est un **LightGBM** optimisé selon un coût métier personnalisé.

Étapes réalisées :

1. validation croisée stratifiée ;
2. prise en compte du déséquilibre de classe ;
3. optimisation du seuil de décision selon un coût métier ;
4. tuning avec Optuna ;
5. entraînement final sur l'ensemble de développement ;
6. évaluation sur un holdout jamais vu ;
7. versioning dans MLflow Model Registry ;
8. analyse d'explicabilité globale et locale.

Nom du modèle dans MLflow :

```text
credit_scoring_model
```

Version retenue :

```text
Version 2 / Production
```

## 6. API FastAPI

### Endpoints disponibles

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API fonctionne |
| `/model-info` | GET | Retourne le modèle, la version, le seuil et le nombre de features |
| `/predict` | POST | Retourne une prédiction pour un client |
| `/predict_batch` | POST | Retourne des prédictions pour plusieurs clients |
| `/metrics` | GET | Expose les métriques Prometheus |
| `/docs` | GET | Documentation Swagger interactive |

### Documentation interactive

```text
http://127.0.0.1:8000/docs
```

### Exemple de payload

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

### Exemple de réponse

```json
{
  "proba_default": 0.3197,
  "decision": 0,
  "threshold": 0.42,
  "model_name": "credit_scoring_model",
  "model_version": "2",
  "latency_ms": 46.02
}
```

## 7. Base de données PostgreSQL

Chaque appel API est stocké dans la table `predictions`.

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

Cette table permet :

- l'audit des prédictions ;
- le monitoring de la latence ;
- le suivi des scores ;
- la détection de drift ;
- l'analyse des erreurs.

## 8. Interface Streamlit

L'application Streamlit propose deux vues :

1. **Scoring Client** : interface utilisateur pour envoyer un payload à l'API et afficher le score ;
2. **Monitoring Dev** : vue synthétique des dernières prédictions, scores et latences stockés dans PostgreSQL.

URL locale :

```text
http://127.0.0.1:8501
```

## 9. Monitoring technique

### Prometheus

Prometheus collecte les métriques exposées par FastAPI sur `/metrics`.

URL locale :

```text
http://127.0.0.1:9090
```

Page importante :

```text
http://127.0.0.1:9090/targets
```

La cible `credexp_api` doit être `UP`.

### Grafana

Grafana visualise les métriques collectées par Prometheus.

URL locale :

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

## 10. Monitoring de drift

Le drift est analysé avec Evidently.

Sources utilisées :

- référence : `data/processed/reference.parquet` ;
- production : inputs stockés dans PostgreSQL via les appels API.

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

## 11. Performance et optimisation

Le système a été évalué à plusieurs niveaux :

1. inférence locale du pipeline sklearn ;
2. latence API FastAPI de bout en bout ;
3. inférence batch vs inférence unitaire ;
4. profiling avec `cProfile` ;
5. benchmark ONNX Runtime.

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

## 12. Lancement local complet

### Pré-requis

- Python 3.12 ;
- `uv` ;
- Docker Desktop ;
- Git.

### Installation Python

```powershell
uv sync --all-groups
```

### Lancement de la stack Docker

```powershell
docker compose down -v
docker compose up --build -d
Start-Sleep -Seconds 30
docker compose ps
```

### Initialisation de la base

```powershell
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/credexp"
uv run python scripts/init_db.py
```

### Génération de prédictions de démo

```powershell
1..50 | ForEach-Object { .\api_examples\test_api.ps1 }
```

### Génération du drift et des benchmarks

```powershell
uv run python scripts/monitoring_drift.py --limit 500
uv run python scripts/profile_inference.py
uv run python scripts/benchmark_api.py
uv run python scripts/benchmark_batching.py
uv run python scripts/benchmark_onnx.py
```

### Interfaces locales

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8000/docs |
| Streamlit | http://127.0.0.1:8501 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 |

## 13. Tests et qualité

### Lint et format

```powershell
uv run ruff check .
uv run ruff format --check .
```

### Tests automatisés

```powershell
uv run pytest -q
```

### Docker Compose

```powershell
docker compose config
```

## 14. CI/CD

Le dépôt contient un workflow GitHub Actions qui :

- se déclenche sur `push` et `pull_request` ;
- installe Python et `uv` ;
- synchronise les dépendances ;
- exécute Ruff ;
- exécute Pytest ;
- valide la configuration Docker Compose ;
- construit l'image Docker de l'API.

Fichier :

```text
.github/workflows/ci.yml
```

## 15. Démonstration

Guide de démonstration :

```text
demo.md
```

## 16. Contenu principal

- historique Git ;
- scripts API ;
- Dockerfile ;
- Docker Compose ;
- tests automatisés ;
- pipeline CI/CD ;
- notebooks de monitoring et performance ;
- rapport Evidently ;
- screenshots de PostgreSQL, Prometheus, Grafana, FastAPI et Streamlit ;
- README et documentation de démonstration.

## 17. Perspectives

Améliorations possibles :

- alertes automatiques Prometheus/Grafana ;
- monitoring de performance modèle sur données labellisées récentes ;
- retraining automatisé.

## Auteur

Benoît Girard
