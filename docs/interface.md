# The two pages, and the API underneath them

A form that scores one applicant, and a log that says what the service has been deciding.
Both are Streamlit, both hold no logic of their own, and both read the palette the figures of
the README use. `docs/architecture.md` says why the boundary sits where it does; this page
says what the two show and how to talk to the API without them.

```bash
docker compose up -d                                   # both pages, on :8501
uv run streamlit run src/credexp/app/Overview.py       # or locally, without Docker
```

## Score an applicant

<!-- source: docs/images/MANIFEST.json -->
![The form, filled with the example payload, and the three metrics the answer is read by](images/streamlit-scoring.png)

| On screen | What it is for |
|---|---|
| The request body | a payload to edit; three features is enough, the rest are imputed |
| Probability of default | the raw score, which is a ranking score and not a calibrated one |
| Decision | what the threshold makes of it: accept, or refuse |
| Latency | what the whole round trip cost, in milliseconds |
| The full response | everything the API returned, including the model version |

Three failures are told apart, because they need three different answers from the reader: a
body that is not valid JSON, a service that refused the request and explained why, and a
service that is not running at that address. `credexp.app.service` is where that happens, and
`tests/unit/app/test_service.py` exercises all three without a browser.

## Recent decisions

<!-- source: docs/images/MANIFEST.json -->
![The log page, reading two hundred rows out of PostgreSQL: the cards, the table, and the three charts under it](images/prediction-log.png)

| Section | The question it answers |
|---|---|
| The four cards | how many decisions, how fast, how risky on average, how many refused |
| The log | the rows themselves, most recent first |
| Distribution of the score | where the scores fall, on the full 0-to-1 scale |
| Latency, request by request | whether the service is slowing down |
| How the threshold split the traffic | the acceptance rate, which is what the business side watches |

Three rules the charts follow, each of them stated in `credexp.app.charts` and pinned in
`tests/unit/app/test_charts.py`. The probability axis is fixed from 0 to 1. The threshold is
drawn on it, dashed, in the palette's reserved colour. And the two outcomes are named in words,
so that no colour scale has to carry the difference between accepted and refused.

## Talking to the API directly

The OpenAPI page at `/docs` is generated from the Pydantic models the service validates
against, so it is never out of date. What follows is the same thing from a terminal.

One applicant:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sk_id_curr": 123456, "features": {"EXT_SOURCE_1": 0.52, "EXT_SOURCE_2": 0.71}}'
```

```json
{"proba_default": 0.476, "decision": 0, "threshold": 0.49,
 "model_name": "credit_scoring_model", "model_version": "joblib", "latency_ms": 23.8}
```

Several at once, in one call rather than one call each:

```bash
curl -s -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"features": {"EXT_SOURCE_1": 0.1}}, {"features": {"EXT_SOURCE_1": 0.9}}]}'
```

What is serving, and whether it is ready to serve:

```bash
curl -s http://127.0.0.1:8000/model-info
curl -s http://127.0.0.1:8000/health    # the process is up
curl -s http://127.0.0.1:8000/ready     # a model is loaded and it can predict
```

A feature left out of the payload is imputed by the pipeline, so a short request is a valid
one. A field name that is not a feature is refused with a 422, and `credexp.serving.schemas`
says what accepting it would mean for the applicant being scored.

## The theme belongs to the repository

`.streamlit/config.toml` carries the page's four colours, and the theme test is what keeps
that file and the figures' module on the same four. Its own docstring says what is
missing from a page that ships without it.
