# API Test Examples with curl

Exemples de requêtes `curl` pour tester les principaux endpoints de l'API locale.

Base URL utilisée : `http://127.0.0.1:8000`

## Health

```bash
curl http://127.0.0.1:8000/health
```

## Model Info

```bash
curl http://127.0.0.1:8000/model-info
```

## Predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"sk_id_curr\":123456,\"features\":{\"EXT_SOURCE_1\":0.52,\"EXT_SOURCE_2\":0.71,\"EXT_SOURCE_3\":0.41}}"
```

## Predict Batch

```bash
curl -X POST "http://127.0.0.1:8000/predict_batch" ^
  -H "Content-Type: application/json" ^
  -d "{\"items\":[{\"sk_id_curr\":123456,\"features\":{\"EXT_SOURCE_1\":0.52,\"EXT_SOURCE_2\":0.71,\"EXT_SOURCE_3\":0.41}},{\"sk_id_curr\":123457,\"features\":{\"EXT_SOURCE_1\":0.15,\"EXT_SOURCE_2\":0.22,\"EXT_SOURCE_3\":0.18}}]}"
```
