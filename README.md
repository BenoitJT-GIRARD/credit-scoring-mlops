# Credit Scoring MLOps Project

Projet de déploiement et de monitoring d'un modèle de scoring crédit dans un contexte de production simulé.

Ce projet a été réalisé dans le cadre d'un exercice MLOps autour du dataset **Home Credit Default Risk**. Il couvre l'ensemble du cycle de vie d'un modèle :

- feature engineering
- entraînement et sélection de modèle
- suivi des expérimentations via MLflow
- déploiement d'une API FastAPI
- stockage des prédictions en base PostgreSQL
- monitoring via Prometheus et Grafana
- interface utilisateur via Streamlit
- analyse de dérive des données via Evidently

## Objectif du projet

L'objectif est de déployer un modèle de scoring crédit capable d'attribuer en quasi temps réel une probabilité de défaut à un client, puis de surveiller son comportement en production.

Le projet ne se limite pas à la modélisation. Il intègre aussi les aspects d'industrialisation, de monitoring, de qualité logicielle et de reproductibilité.

## Architecture du projet

### Composants principaux

- **MLflow** : suivi des expérimentations et registre de modèles
- **FastAPI** : service d'inférence
- **PostgreSQL** : stockage des appels de production
- **Streamlit** : interface utilisateur et vue de monitoring simplifiée
- **Prometheus** : collecte des métriques d'API
- **Grafana** : visualisation des métriques techniques
- **Evidently** : détection de dérive des données

### Flux général

`Model Registry -> API FastAPI -> PostgreSQL -> Monitoring (Prometheus / Grafana / Evidently) -> UI Streamlit`

## Structure du dépôt

```text
src/credexp/
  config.py
  data/
  modeling/
  serving/
  db/
  monitoring/
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

streamlit_app/
  app.py
  pages/

docker/
  api.Dockerfile
  prometheus/

notebooks/
  01_build_features.ipynb
  02_eda.ipynb
  03_training_mlflow.ipynb
  04_tuning_registry_final.ipynb
  05_explainability.ipynb
  06_drift_monitoring.ipynb

data/
  raw/
  processed/

artifacts/
  models/

mlflow/
  mlflow.db
```

## Modèle retenu

Le modèle retenu est un **LightGBM** optimisé selon un coût métier personnalisé, puis enregistré dans **MLflow Model Registry**.

Étapes principales :

- validation croisée stratifiée
- optimisation du seuil de décision
- tuning avec Optuna
- entraînement final sur l'ensemble de développement
- évaluation sur holdout
- enregistrement dans le registre MLflow

## API FastAPI

### Endpoints disponibles

- `GET /health` : vérifie que l'API fonctionne
- `GET /model-info` : retourne le nom du modèle, sa version, le seuil et le nombre de features
- `POST /predict` : retourne une prédiction pour un client
- `POST /predict_batch` : retourne les prédictions pour plusieurs clients
- `GET /metrics` : expose les métriques Prometheus

### Documentation interactive

Swagger est disponible ici :

- `http://127.0.0.1:8000/docs`

## Interface Streamlit

L'interface Streamlit propose :

- une page `Scoring Client`
- une page `Monitoring Dev`

Elle permet :

- de tester facilement l'API
- de visualiser les dernières prédictions stockées
- de présenter le projet de manière plus accessible qu'avec Swagger seul

## Base de données PostgreSQL

Les appels API sont stockés dans une table `predictions` avec notamment :

- identifiant de requête
- identifiant client
- version du modèle
- score prédit
- décision
- latence
- payload d'entrée
- payload de sortie

Cela permet :

- un audit des appels
- le monitoring
- la détection de dérive

## Monitoring

### Monitoring technique

- Prometheus collecte les métriques exposées par l'API
- Grafana visualise :
  - le nombre de requêtes
  - le débit
  - la latence
  - la répartition des codes HTTP

### Monitoring des données

- Evidently compare les données de production récentes aux données de référence
- un rapport HTML de drift est généré automatiquement

## Lancement local

### Pré-requis

- Python 3.12
- `uv`
- Docker Desktop
- Git

### Installation Python

```bash
uv sync
```

### Lancement de la stack Docker

```bash
docker compose down -v
docker compose up --build -d
```

### Vérification

- API : `http://127.0.0.1:8000/docs`
- Streamlit : `http://127.0.0.1:8501`
- Prometheus : `http://127.0.0.1:9090`
- Grafana : `http://127.0.0.1:3000`

## Utilisation de l'API

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

## Test de l’API

L’API peut être testée de trois façons :

- via **Swagger UI** : `http://127.0.0.1:8000/docs`
- via **curl**
- via le script PowerShell fourni dans `api_examples/test_api.ps1`

Des exemples de requêtes sont fournis dans :

```text
api_examples/curl_examples.md
```

## Génération du rapport de drift

```bash
uv run python scripts/monitoring_drift.py --limit 500
```

Rapport généré :

- `reports/monitoring/evidently_drift.html`

## Tests

### Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

### Tests

```bash
uv run pytest -q
```

## CI/CD

Le dépôt inclut un workflow GitHub Actions qui :

- exécute les tests automatisés
- vérifie le lint et le formatage
- construit l'image Docker

## Contenu principal

- notebooks d'analyse et de monitoring
- scripts d'entraînement et d'API
- Dockerfile et `docker-compose.yml`
- pipeline CI/CD
- rapport de drift
- captures de monitoring et de stockage
- historique Git

## Perspectives d'amélioration

- intégration d'alertes automatiques
- déploiement cloud complet
- retraining automatisé
- optimisation de performance avec profiling ou ONNX Runtime
- déploiement public via Hugging Face Spaces

## Auteur

Benoît Girard
