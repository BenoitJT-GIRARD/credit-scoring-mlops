# How the code is arranged, and where the boundaries are

The project runs as five containers. The README describes what the project decides, and this
page explains how the code is arranged and what each arrangement costs.

## The layout

```
scripts/           the entry points: read arguments, call the package, print what happened
  └── src/credexp/ the package, which holds the logic
        ├── data/        the seven Home Credit tables, joined into one row per applicant
        ├── db/          the prediction-log table and the database session
        ├── modeling/    the pipeline, the threshold, the cost, the explanations
        ├── serving/     the FastAPI application and what it loads
        ├── monitoring/  the Prometheus counters and the Evidently comparison
        ├── app/         the Streamlit pages, and the three modules they call
        └── utils/       the project paths and the logger
```

The compose file at the root of the repository starts five containers: the API, the interface,
PostgreSQL, Prometheus and Grafana. `infra/` holds their images and their provisioning.

## Four boundaries

**The model is a set of files that the service loads.** `models/pipeline.joblib`, its column
order and its threshold are the output of the training. In the containers, the API reads them at
startup and has no access to the training runs that produced them.
`credexp.serving.model_loader` explains what this gains.

A registry records which run produced which artefact, and a file on its own does not. The final
training therefore writes `model_manifest.json` beside the artefact, so that its provenance can
be read without any service, and the README lists what that file holds. The three files
currently under `models/` were trained before the manifest existed and have none.

**Writing the prediction to the log is best effort.** Every scored request is inserted into
PostgreSQL, and a database that does not answer leaves the response unchanged. The system tier
of the tests checks this by scoring an applicant with the log unreachable.

The same tier showed that the property used to hold only once the service was running. Without
a connection deadline, `psycopg` kept retrying an unreachable database until the operating
system gave up, and the service took two minutes to start. `credexp.db.session` now gives the
connection a three-second deadline.

**The interface holds no logic.** Each of the two Streamlit pages only lays out a screen.
`credexp.app.service` calls the API and distinguishes the three ways scoring can fail,
`credexp.app.prediction_log` reads the decision log, and `credexp.app.charts` draws the three
charts. `tests/unit/app/pages/test_pages_are_thin.py` explains the reason for this split and
enforces it.

**A batch of applicants costs one call to the pipeline.** `POST /predict_batch` builds one frame
and calls the pipeline once, and `credexp.serving.api` lists the costs that do not grow with the
number of rows. Measured on the pipeline alone, one applicant inside a batch of fifty costs
0.11 ms, against 4.08 ms when scored by itself. The figures come from
`reports/performance/batching_benchmark.json`, which notebook 07 reads.

## The two probes

`/health` answers without reading the model, and `/model-info` reports the model that was
loaded. `docs/operations.md` gives the table and says which one the container probes. The
service starts listening only after it has loaded the model, so neither route answers during
the load.

## What was left out

The project has no feature store, no scheduler, no retraining loop, and no message queue between
the API and the log. They were rejected because the model is frozen and its training data cannot
be redistributed, so none of them would be exercised, while each would still need maintenance.
`docs/protocol.md` describes what a retraining loop would need.

Two smaller parts are left out as well. The prediction log is written inline, with no
asynchronous queue, because a queue would need a consumer process to keep running for a
demonstration service. The API has no authentication, since it runs locally and nobody would
hold an account for it.
