# Why a service, and where the boundaries are

Four processes, and the reason each one is separate from the others. The README says what
the project decides; this page says why the code is arranged the way it is, and what each
arrangement costs.

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
knows nothing about how they were produced; `credexp.serving.model_loader` says what that
decision buys.

The cost of that choice is real. A registry knows which run produced which artefact and a
file does not, so the artefact carries `model_manifest.json` beside it. A model whose
provenance travels with it needs no service to be understood, and `docs/protocol.md` lists what
that file holds.

**Writing the prediction down is best effort.** Every scored request is inserted into
PostgreSQL, and a database that does not answer stops nothing. A claim like that is worth
exactly as much as the test behind it, which is why the system tier serves an applicant with
the log unreachable.

The same tier is what found the cost of the claim being half-true: `psycopg` retried an
unreachable database until the operating system gave up, and the service took two minutes to
start. `credexp.db.session` now gives the connection a three-second deadline.

**The interface holds no logic.** The two Streamlit pages are a screen each. Calling the API
and naming the three ways that can fail is `credexp.app.service`; reading the decision log is
`credexp.app.prediction_log`; the three charts are `credexp.app.charts`. The reason for the
split is in `tests/unit/app/pages/test_pages_are_thin.py`, which is also what keeps it.

**Scoring a batch costs one call, not a hundred.** `POST /predict_batch` builds one frame and
calls the pipeline once; `credexp.serving.api` names the costs that do not grow with the number
of rows. One applicant inside a batch of fifty costs 0.11 ms against 4.08 ms alone, measured in
`reports/performance/batching_benchmark.json` and read by notebook 07.

## What the two probes answer

Two routes, two questions, and an orchestrator that acts differently on each.
`docs/operations.md` has the table, and the short of it is that the container probes the one
that answers before the model is loaded.

## What was deliberately left out

No feature store, no scheduler, no retraining loop, and no message queue between the API and
the log. Each was considered and rejected for the same reason: the model is frozen, the data
it was fitted on cannot be redistributed, and a component nobody exercises rots faster than
one nobody built. What a retraining loop would need is written down in `docs/protocol.md`
instead of half-built here.

Two smaller omissions, for the record. **No asynchronous write**: the prediction log is
inserted inline, because a queue would need a consumer to keep alive for a demonstration
service. **No authentication**: the API is local, and an auth layer nobody can log in to
would be scaffolding rather than security.
