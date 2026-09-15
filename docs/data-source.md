# Where the data comes from, and what may be redistributed

Nothing of it is in this repository, and the reason is a licence rather than a size.

## The source

| | |
|---|---|
| Dataset | [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) |
| Provider | Home Credit Group, through a Kaggle competition |
| Tables | seven, related by `SK_ID_CURR` and `SK_ID_BUREAU` |
| Applications | 307 511 labelled, of which 276 760 fitted and 30 751 held out, plus 48 744 unlabelled |
| Default rate | 8.07 % on the labelled rows |
| Terms | competition rules: use for the competition, no redistribution |
| Read on | 2026-02-21, the day the feature matrix was built |

The competition's terms allow a participant to download and use the data. They do not allow
publishing it again, in whole or in part, and that includes a sample small enough to be
convenient. So this repository ships no row of it, and every script that needs it says which
file it could not find.

## What is shipped instead

| What | Where | Why it may be published |
|---|---|---|
| The fitted pipeline | `models/pipeline.joblib` | 4 MB of weights this repository trained |
| The column order | `models/feature_columns.json` | the names of engineered columns, not data |
| The threshold | `models/threshold.json` | one number and how it was chosen |
| Every published measure | `reports/**` | aggregates over tens of thousands of applicants |
| The tuning study | `reports/tuning/optuna_trials.csv` | hyper-parameters and their scores |

An aggregate over 30 751 applicants identifies nobody, and a fitted model is this
repository's own work. The line is drawn at rows.

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

`src/credexp/data/build_features.py` joins them into 796 columns, one row per applicant.
**Each aggregate looks only at the applicant's own past**, keyed on `SK_ID_CURR`, over tables
that record what happened before the application was made. Nothing is pooled between applicants
and nothing is taken from after the decision. That property is what the README's leak section
rests on, and `tests/unit/data/test_build_features.py` pins the decisions the join makes.

## Where the files go

```
var/data/raw/         the seven CSVs, as downloaded from Kaggle
var/data/interim/     the merged tables, between the CSVs and the matrix
var/data/processed/   features.parquet, api_holdout.parquet, reference.parquet
```

Under `var/` and not under `data/`: `credexp.utils.paths` gives the reason, and it is the
only module of the project that resolves a path at all.

To rebuild everything from the download:

```bash
uv run python scripts/build_features.py     # var/data/processed/features.parquet
uv run python scripts/train_final.py        # models/, from the tracked tuning trials
uv run python scripts/decision_analysis.py  # reports/decision/
```

`build_features.py` fails by name when a table is missing: a feature matrix built from six of
the seven tables is silently a hundred columns short, and a model trains on it without
complaint.

## One thing the sentinel hides

`DAYS_EMPLOYED` carries 365 243 for applicants who have never been employed, which is a
thousand years in a column where every real value is negative. Read as a number it moves
every statistic that touches it. It becomes missing during the build, and a test asserts it.
