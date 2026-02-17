from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from credexp.data.io import read_csv, save_parquet
from credexp.utils.logging import get_logger

log = get_logger(__name__)


REQUIRED_RAW_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]


@contextmanager
def timer(title: str):
    t0 = time.time()
    yield
    log.info(f"{title} - done in {time.time() - t0:.0f}s")


def _check_raw_files(raw_path: Path) -> None:
    missing = [f for f in REQUIRED_RAW_FILES if not (raw_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files in data/raw: " + ", ".join(missing) + f" (checked in: {raw_path})"
        )


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(raw_path: Path, num_rows: int | None = None, nan_as_category: bool = False):
    df = read_csv(raw_path / "application_train.csv", nrows=num_rows)
    test_df = read_csv(raw_path / "application_test.csv", nrows=num_rows)
    log.info(f"Train samples: {len(df)}, test samples: {len(test_df)}")

    # Merge train + test (same as Kaggle logic, modern concat)
    df = pd.concat([df, test_df], axis=0, ignore_index=True)

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df["CODE_GENDER"] != "XNA"]

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], _uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, _cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365243 -> nan
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Some simple new features (percentages)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    del test_df
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(raw_path: Path, num_rows: int | None = None, nan_as_category: bool = True):
    bureau = read_csv(raw_path / "bureau.csv", nrows=num_rows)
    bb = read_csv(raw_path / "bureau_balance.csv", nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for col in bureau_cat:
        cat_aggregations[col] = ["mean"]
    for col in bb_cat:
        cat_aggregations[col + "_MEAN"] = ["mean"]

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index(["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index(["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
    del bureau, closed, closed_agg
    gc.collect()

    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(raw_path: Path, num_rows: int | None = None, nan_as_category: bool = True):
    prev = read_csv(raw_path / "previous_application.csv", nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)

    # Days 365243 -> nan
    cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    prev[cols] = prev[cols].replace(365243, np.nan)

    # Add feature: value ask / value received percentage
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    # Previous applications numeric features
    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }

    # Previous applications categorical features
    cat_aggregations = {col: ["mean"] for col in cat_cols}

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
    approved_agg.columns = pd.Index(["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
    del approved, approved_agg
    gc.collect()

    # Previous Applications: Refused
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
    refused_agg.columns = pd.Index(["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
    del refused, refused_agg, prev
    gc.collect()

    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(raw_path: Path, num_rows: int | None = None, nan_as_category: bool = True):
    pos = read_csv(raw_path / "POS_CASH_balance.csv", nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)

    # Features
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for col in cat_cols:
        aggregations[col] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index(["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()

    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(raw_path: Path, num_rows: int | None = None, nan_as_category: bool = True):
    ins = read_csv(raw_path / "installments_payments.csv", nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    # Days past due and days before due (no negative values)
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
    ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for col in cat_cols:
        aggregations[col] = ["mean"]

    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = pd.Index(["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()

    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(raw_path: Path, num_rows: int | None = None, nan_as_category: bool = True):
    cc = read_csv(raw_path / "credit_card_balance.csv", nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category)

    # General aggregations
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
    cc_agg = cc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index(["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()

    del cc
    gc.collect()
    return cc_agg


def build_feature_matrix(raw_path: Path, debug: bool = False) -> pd.DataFrame:
    """
    Build the final feature matrix exactly like the Kaggle kernel logic:
    application_train_test -> join bureau -> join previous -> join pos -> join installments -> join credit card.
    """
    _check_raw_files(raw_path)

    num_rows = 10000 if debug else None

    df = application_train_test(raw_path, num_rows=num_rows)

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(raw_path, num_rows=num_rows)
        log.info(f"Bureau df shape: {bureau.shape}")
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()

    with timer("Process previous_applications"):
        prev = previous_applications(raw_path, num_rows=num_rows)
        log.info(f"Previous applications df shape: {prev.shape}")
        df = df.join(prev, how="left", on="SK_ID_CURR")
        del prev
        gc.collect()

    with timer("Process POS-CASH balance"):
        pos = pos_cash(raw_path, num_rows=num_rows)
        log.info(f"POS-CASH balance df shape: {pos.shape}")
        df = df.join(pos, how="left", on="SK_ID_CURR")
        del pos
        gc.collect()

    with timer("Process installments payments"):
        ins = installments_payments(raw_path, num_rows=num_rows)
        log.info(f"Installments payments df shape: {ins.shape}")
        df = df.join(ins, how="left", on="SK_ID_CURR")
        del ins
        gc.collect()

    with timer("Process credit card balance"):
        cc = credit_card_balance(raw_path, num_rows=num_rows)
        log.info(f"Credit card balance df shape: {cc.shape}")
        df = df.join(cc, how="left", on="SK_ID_CURR")
        del cc
        gc.collect()

    # IMPORTANT: match kernel training behavior (do not keep synthetic index col)
    df.drop(columns=["index"], errors="ignore", inplace=True)

    log.info(f"Final feature matrix shape: {df.shape}")

    # Sanity checks (non bloquants)
    if "TARGET" in df.columns:
        n_train = int(df["TARGET"].notna().sum())
        n_test = int(df["TARGET"].isna().sum())
        log.info(f"TARGET present: train_rows={n_train} test_rows={n_test}")
    log.info(f"Missing values total={int(df.isna().sum().sum())}")

    return df


def build_and_save(raw_path: Path, out_path: Path, debug: bool = False) -> None:
    df = build_feature_matrix(raw_path=raw_path, debug=debug)
    save_parquet(df, out_path)
