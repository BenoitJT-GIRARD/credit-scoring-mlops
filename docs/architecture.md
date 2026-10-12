# Why a service, and where the boundaries are

The project runs as four processes, each kept separate from the others for a reason given
below. The README covers what the project decides; this page explains why the code is arranged
as it is, and what each arrangement costs.

## The shape

```
scripts/           the entry points: read arguments, call the package, print what happened
  └── src/credexp/ the package, which is where every decision lives
        ├── data/        the seven Home Credit tables, joined into one row per applicant
        ├── modeling/    the pipeline, the threshold, the cost, the explanations
        ├── serving/     the FastAPI application and what it loads
        ├── monitoring/  the Prometheus counters and the Evidently comparison
        └── app/         the Streamlit pages, and the two modules they are thin over
```

The compose file starts five containers: the API, the interface, PostgreSQL, Prometheus and
Grafana. `infra/` holds all of it.

## Four boundaries worth defending

**The model is a file, and the service loads a file.** `models/pipeline.joblib`, its column
order and its threshold are the deliverable of the training. The API reads them at startup and
knows nothing about how they were produced; `credexp.serving.model_loader` explains what that
decision gains.

That choice has a cost: a registry knows which run produced which artefact, and a file does
not. So the artefact carries `model_manifest.json` beside it, and with its provenance stored
there the model can be understood without any service. `docs/protocol.md` lists what that file
holds.

**Writing the prediction down is best effort.** Every scored request is inserted into
PostgreSQL, and a database that does not answer stops nothing. The system tier tests that
claim by serving an applicant with the log unreachable.

The same tier showed that the claim had been only half true: `psycopg` retried an unreachable
database until the operating system gave up, and the service took two minutes to start.
`credexp.db.session` now gives the connection a three-second deadline.

**The interface holds no logic.** Each of the two Streamlit pages only draws a screen. Calling
the API and naming the three ways that can fail is `credexp.app.service`; reading the decision
log is `credexp.app.prediction_log`; the three charts are `credexp.app.charts`. The reason for
the split is in `tests/unit/app/pages/test_pages_are_thin.py`, which also enforces it.

**Scoring a batch costs one call, not a hundred.** `POST /predict_batch` builds one frame and
calls the pipeline once; `credexp.serving.api` names the costs that do not grow with the number
of rows. One applicant inside a batch of fifty costs 0.11 ms against 4.08 ms alone, measured in
`reports/performance/batching_benchmark.json` and read by notebook 07.

## What the two probes answer

Two probe routes answer two different questions, and an orchestrator acts differently on
each. `docs/operations.md` has the table; in short, the container probes the one that answers
before the model is loaded.

## What was deliberately left out

The project has no feature store, no scheduler, no retraining loop, and no message queue
between the API and the log. Each was considered and rejected for the same reason: the model
is frozen, the data it was fitted on cannot be redistributed, and a component nobody exercises
decays, which leaves the project worse off than never building it. What a retraining loop would
need is written down in `docs/protocol.md` instead of half-built here.

Two smaller things are left out as well. The prediction log is inserted inline, with no
asynchronous write, because a queue would need a consumer to keep alive for a demonstration
service. There is no authentication either: the API is local, and an auth layer nobody can log
in to would be scaffolding rather than security.
