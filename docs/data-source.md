# Where the data comes from, and what may be redistributed

The data is not in this repository, because its licence does not allow redistribution.

## The source

| | |
|---|---|
| Dataset | [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) |
| Provider | Home Credit Group, through a Kaggle competition |
| Tables | seven, in eight of the CSV files, related by `SK_ID_CURR` and `SK_ID_BUREAU` |
| Applications | 307 511 labelled in the download, of which the build keeps 307 507: 276 756 fitted and 30 751 held out; 48 744 unlabelled |
| Default rate | 8.07 % on the labelled rows |
| Terms | the rules of the Kaggle competition, which restrict redistribution |
| Read on | 2026-02-18, the date in the logs of notebook 01, which reads the raw files |

The competition rules restrict redistribution of the data, so this repository ships none of the
raw files. The scripts that need the data stop, or log a warning,
with a message that says which file or frame is missing.

The build drops the four labelled applications whose `CODE_GENDER` is `XNA`. The holdout is 10 %
of the remaining labelled rows, and the development set the shipped model was fitted on is the
other 90 %.

## What is shipped in its place

| What | Where | Why it may be published |
|---|---|---|
| The fitted pipeline | `models/pipeline.joblib` | 4 MB of weights trained by this repository |
| The column order | `models/feature_columns.json` | the names of the engineered columns |
| The threshold | `models/threshold.json` | one number and the rule that chose it |
| The published tables | the tables of `reports/decision/`, `reports/performance/calibration.json` and `reports/model_comparison.csv` | aggregates over groups of applicants |
| The published figures | `reports/figures/` | mostly aggregates, with the exceptions below |
| The tuning study | `reports/tuning/optuna_trials.csv` | hyper-parameters and their scores |

An aggregate over 30 751 applicants does not identify anyone, and the fitted model is this
repository's own work, so both are published. A few figures show individual applicants. Each of
the two SHAP waterfalls prints 24 feature values of one applicant, the SHAP beeswarm draws one
point per applicant, and the screenshot of the decision log, `docs/images/prediction-log.png`,
shows applicant identifiers beside their scores. No complete row of the data is published.

## The seven tables, and what each contributes

| Table | One row is | What is taken from it |
|---|---|---|
| `application_train` / `application_test` | one loan application | the applicant's own fields, and the label |
| `bureau` | one credit held elsewhere, reported to the bureau | counts, sums and extremes per applicant |
| `bureau_balance` | one month of one of those credits | the length and the state of each history |
| `previous_application` | one earlier application to Home Credit | how many, how large, how many refused |
| `POS_CASH_balance` | one month of one earlier point-of-sale loan | the months carried, and the days past due |
| `installments_payments` | one instalment, due and paid | late payments, and what was short |
| `credit_card_balance` | one month of one earlier card | drawings, and how the balance moved |

`src/credexp/data/build_features.py` joins them into one row per applicant, with 796 feature
columns besides the label. The first of these columns is the applicant identifier `SK_ID_CURR`,
which the model therefore receives as a feature. Each aggregate is grouped on `SK_ID_CURR`, so
it summarises one applicant's own records, and no aggregate pools several applicants. The build
applies no date filter. That the secondary tables describe only the period before the
application is a property of how Home Credit built them, which this repository cannot check.
The section of the README on leaks relies on these properties, and
`tests/unit/data/test_build_features.py` pins the choices the join makes.

## Where the files go

```
var/data/raw/         the eight CSV files the build reads, as downloaded from Kaggle
var/data/interim/     declared in credexp.utils.paths; no script writes to it
var/data/processed/   features.parquet, api_holdout.parquet, reference.parquet
```

The files sit under `var/`, and the comment above `DATA_ROOT` in `credexp.utils.paths` gives the
reason. That module is the only one in the project that computes the project root.

To rebuild everything from the download:

```bash
uv run python scripts/build_features.py     # var/data/processed/features.parquet
uv run python scripts/train_final.py        # the holdout, and models/ from the tracked trials
git checkout -- models/                     # back to the shipped artefacts
rm models/model_manifest.json               # written by the training, absent from the shipped set
uv run python scripts/decision_analysis.py  # reports/decision/
```

`train_final.py` replaces the shipped model, and nothing checks that the retraining reproduces it.
The two restore lines bring back the shipped artefacts, so that the analysis describes the
published model.

`credexp.data.build_features` checks the eight files before reading any of them, and names every
missing file in a single error. Without that check, the build would stop at the first `read_csv`
on a missing file and report that file alone.

## The sentinel in `DAYS_EMPLOYED`

`DAYS_EMPLOYED` holds 365 243 for applicants who have never been employed, which is about a
thousand years in a column whose real values are negative. Read as a number, it would distort
every statistic computed from the column. The build replaces it with a missing value, and a test
checks that it does.
