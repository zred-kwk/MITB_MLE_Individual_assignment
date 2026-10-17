import argparse
import os
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import col
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# to call this script: python model_train.py --snapshotdate "2024-09-01"
#
# Windows, relative to the training date T (labels are filtered on the LABEL snapshot_date):
#   OOT        : [T - oot_period_months, T - 1 day]
#   train/test : [T - oot_period_months - train_test_period_months, T - oot_period_months - 1 day]
# Features are joined to labels on loan_id (labels are dated origination + mob, features at origination).

LABEL_STORE_DIR = "datamart/gold/label_store/"
FEATURE_STORE_DIR = "datamart/gold/feature_store/"
MODEL_BANK_DIR = "model_bank/"

# identifiers / label columns / raw free-text columns that are never model inputs
EXCLUDE_COLS = [
    "loan_id", "snapshot_date", "label", "feature_snapshot_date", "Customer_ID",
    "label_def", "loan_start_date", "Occupation", "Type_of_Loan","Clickstream_Months_Observed"
]

PARAM_DIST = {
    "n_estimators": [25, 50],
    "max_depth": [2, 3],  # lower max_depth to simplify the model
    "learning_rate": [0.01, 0.1],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8],
    "gamma": [0, 0.1],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 1.5, 2],
}


# monitoring reference (read by model_monitor.py): bins per feature / score, and the
# number of most frequent categories kept per categorical feature
REFERENCE_BINS = 10
REFERENCE_MAX_CATEGORIES = 20
MOB_MONTHS = 6


def build_config(snapshotdate, train_test_period_months=12, oot_period_months=2, train_test_ratio=0.8):
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] = oot_period_months
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["train_test_ratio"] = train_test_ratio

    # artefact name; model_inference.py is called with --modelname <model_name>
    config["model_version"] = "credit_model_" + snapshotdate.replace("-", "_")
    config["model_name"] = config["model_version"] + ".pkl"
    config["model_bank_directory"] = MODEL_BANK_DIR
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    return config


def read_store(spark, folder, prefix, first_month, last_month):
    """
    Read only the monthly partitions in [first_month, last_month] (first-of-month dates).
    Globbing the whole folder could pick up a partition that gold_feature is still
    writing for the same run, or stale partitions from old runs.
    """
    paths, missing, d = [], [], first_month
    while d <= last_month:
        p = f"{folder}{prefix}_{d:%Y_%m_%d}.parquet"
        (paths if os.path.exists(p) else missing).append(p)
        d += relativedelta(months=1)
    if missing:
        raise FileNotFoundError(f"Missing partitions: {missing}")
    return spark.read.option("header", "true").parquet(*paths)


def load_data(spark, config):
    # Labels needed: label dates from the start of the train/test window up to T - 1 month.
    label_first = config["train_test_start_date"]
    label_last = config["model_train_date"] - relativedelta(months=1)

    # --- label store: filtered to the train/test + OOT window on the label date ---
    label_store_sdf = read_store(spark, LABEL_STORE_DIR, "gold_label_store", label_first, label_last)
    labels_sdf = label_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"])
    )
    n_labels = labels_sdf.count()
    print("extracted labels_sdf", n_labels, config["train_test_start_date"], config["oot_end_date"])
    if n_labels == 0:
        raise ValueError("No labels in the window for snapshot " + config["model_train_date_str"])

    # --- feature store: only the partitions the label window requires. A label dated s needs
    #     the features dated at its origination (s - mob months), so the labels define the
    #     window and loan_id does the matching. Never reads features from T itself. ---
    features_store_sdf = read_store(
        spark, FEATURE_STORE_DIR, "gold_feature_store",
        label_first - relativedelta(months=MOB_MONTHS),
        label_last - relativedelta(months=MOB_MONTHS),
    )
    feat_sdf = (
        features_store_sdf
        .withColumnRenamed("snapshot_date", "feature_snapshot_date")
        .drop("Customer_ID")  # labels already carry Customer_ID
    )

    data_pdf = labels_sdf.join(feat_sdf, on="loan_id", how="left").toPandas()

    # one feature row per loan, and every label must have features
    if len(data_pdf) != n_labels:
        raise ValueError(f"Join changed the row count ({n_labels} labels -> {len(data_pdf)}): duplicate loan_id in the feature store")
    n_unmatched = int(data_pdf["feature_snapshot_date"].isna().sum())
    if n_unmatched > 0:
        unmatched = data_pdf[data_pdf["feature_snapshot_date"].isna()]
        last_feat = features_store_sdf.agg(F.max("snapshot_date")).first()[0]
        raise ValueError(
            f"{n_unmatched} of {n_labels} labels have no feature row. "
            f"Unmatched label dates: {unmatched['snapshot_date'].min()} to {unmatched['snapshot_date'].max()}. "
            f"Last feature partition: {last_feat}. "
            f"Labels need features dated label_date - {MOB_MONTHS} months."
        )

    return data_pdf


def split_data(data_pdf, config):
    # split on the label date
    oot_pdf = data_pdf[
        (data_pdf["snapshot_date"] >= config["oot_start_date"].date())
        & (data_pdf["snapshot_date"] <= config["oot_end_date"].date())
    ]
    train_test_pdf = data_pdf[
        (data_pdf["snapshot_date"] >= config["train_test_start_date"].date())
        & (data_pdf["snapshot_date"] <= config["train_test_end_date"].date())
    ].reset_index(drop=True)

    # fail loudly rather than train on an empty / single-class window
    for name, df in [("oot", oot_pdf), ("train_test", train_test_pdf)]:
        if df.empty or df["label"].nunique() < 2:
            raise ValueError(f"{name} window is empty or single-class for snapshot {config['model_train_date_str']}")

    feature_cols = [c for c in data_pdf.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(data_pdf[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # stratified + grouped 80/20: first fold of a StratifiedGroupKFold (a customer is only ever on one side)
    n_splits = max(2, int(round(1 / (1 - config["train_test_ratio"]))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=88)
    train_idx, test_idx = next(
        sgkf.split(train_test_pdf, train_test_pdf["label"], groups=train_test_pdf["Customer_ID"])
    )

    X_train, y_train = train_test_pdf.loc[train_idx, feature_cols], train_test_pdf.loc[train_idx, "label"]
    X_test, y_test = train_test_pdf.loc[test_idx, feature_cols], train_test_pdf.loc[test_idx, "label"]
    groups_train = train_test_pdf.loc[train_idx, "Customer_ID"]
    X_oot, y_oot = oot_pdf[feature_cols], oot_pdf["label"]

    print("X_train", X_train.shape[0])
    print("X_test", X_test.shape[0])
    print("X_oot", X_oot.shape[0])
    print("y_train", y_train.shape[0], round(y_train.mean(), 2))
    print("y_test", y_test.shape[0], round(y_test.mean(), 2))
    print("y_oot", y_oot.shape[0], round(y_oot.mean(), 2))

    return X_train, X_test, X_oot, y_train, y_test, y_oot, groups_train, feature_cols, num_cols, cat_cols


def train_model(X_train, X_test, X_oot, y_train, y_test, y_oot, groups_train, num_cols, cat_cols):
    # --- preprocess: fit on train only, transform train / test / oot with train statistics ---
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_oot_processed = preprocessor.transform(X_oot)
    print("X_train_processed", X_train_processed.shape)

    # --- hyperparameter search: stratified, customer-grouped CV; scores predict_proba ---
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=88)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=88)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=PARAM_DIST,
        scoring="roc_auc",
        n_iter=10,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train_processed, y_train, groups=groups_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best CV AUC score: ", random_search.best_score_)

    best_model = random_search.best_estimator_
    auc = {
        "train": roc_auc_score(y_train, best_model.predict_proba(X_train_processed)[:, 1]),
        "test": roc_auc_score(y_test, best_model.predict_proba(X_test_processed)[:, 1]),
        "oot": roc_auc_score(y_oot, best_model.predict_proba(X_oot_processed)[:, 1]),
    }
    return best_model, preprocessor, random_search.best_params_, auc


def quantile_bins(values, n_bins=REFERENCE_BINS):
    """Quantile bin edges (min..max, duplicates removed) and the share of values in each bin."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [], []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:  # constant column: a single bin
        edges = np.array([edges[0], edges[0]])
    counts = np.histogram(values, bins=edges)[0]
    return edges.tolist(), (counts / counts.sum()).tolist()


def numeric_reference(series):
    values = pd.to_numeric(series, errors="coerce")
    x = values.dropna().astype(float)
    ref = {"kind": "numeric", "missing_rate": float(values.isna().mean())}
    if x.empty:
        return ref
    edges, pct = quantile_bins(x.to_numpy())
    ref.update({
        "mean": float(x.mean()),
        "median": float(x.median()),
        "var": float(x.var(ddof=1)) if len(x) > 1 else np.nan,
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "edges": edges,
        "pct": pct,
    })
    return ref


def categorical_reference(series):
    # missing values are their own category, the same way model_monitor.py maps them
    shares = series.astype("string").fillna("__MISSING__").value_counts(normalize=True)
    top = shares.iloc[:REFERENCE_MAX_CATEGORIES]
    return {
        "kind": "categorical",
        "missing_rate": float(series.isna().mean()),
        "categories": [str(c) for c in top.index],
        "pct": top.to_numpy(dtype=float).tolist(),
        "other_pct": float(max(0.0, 1.0 - top.sum())),
    }


def build_reference(X_train, y_oot, oot_scores, num_cols, cat_cols):
    """
    Distributions model_monitor.py compares live data against.
      features : raw feature distributions of the training split
      score_*  : model score distribution on the OOT window
      roc_oot  : ROC curve on the OOT window, on a fixed 101-point FPR grid
    """
    score_edges, score_pct = quantile_bins(oot_scores)
    features = {c: numeric_reference(X_train[c]) for c in num_cols}
    features.update({c: categorical_reference(X_train[c]) for c in cat_cols})

    fpr, tpr, _ = roc_curve(y_oot, oot_scores)
    grid = np.linspace(0, 1, 101)
    return {
        "score_edges": score_edges,
        "score_pct": score_pct,
        "features": features,
        "roc_oot": {"fpr": grid.tolist(), "tpr": np.interp(grid, fpr, tpr).tolist()},
    }


def build_artefact(config, model, preprocessor, best_params, auc, feature_cols, num_cols, cat_cols,
                   X_train, X_test, X_oot, y_train, y_test, y_oot, reference):
    results = {}
    for split_name, auc_score in auc.items():
        results["auc_" + split_name] = auc_score
        results["gini_" + split_name] = round(2 * auc_score - 1, 3)

    return {
        "model": model,
        "model_version": config["model_version"],
        # key kept as "stdscaler" because model_inference.py reads this name; the object is a ColumnTransformer
        "preprocessing_transformers": {"stdscaler": preprocessor},
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "data_dates": config,
        "data_stats": {
            "X_train": X_train.shape[0], "X_test": X_test.shape[0], "X_oot": X_oot.shape[0],
            "y_train": round(y_train.mean(), 2), "y_test": round(y_test.mean(), 2), "y_oot": round(y_oot.mean(), 2),
        },
        "results": results,
        "hp_params": best_params,
        # distributions captured at training time; model_monitor.py compares live data to these
        "reference": reference,
    }


def save_artefact(model_artefact, config):
    os.makedirs(config["model_bank_directory"], exist_ok=True)
    filepath = config["model_artefact_filepath"]
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "wb") as file:
        pickle.dump(model_artefact, file)
    os.replace(tmp_path, filepath)
    return filepath


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    # --- set up config ---
    config = build_config(snapshotdate)
    pprint.pprint(config)

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # --- load labels + features (Spark is only needed for this step) ---
    try:
        data_pdf = load_data(spark, config)
    finally:
        spark.stop()

    # --- split, preprocess, train ---
    (X_train, X_test, X_oot, y_train, y_test, y_oot,
     groups_train, feature_cols, num_cols, cat_cols) = split_data(data_pdf, config)

    model, preprocessor, best_params, auc = train_model(
        X_train, X_test, X_oot, y_train, y_test, y_oot, groups_train, num_cols, cat_cols
    )

    # --- monitoring reference: features from the training split, score and ROC from OOT ---
    oot_scores = model.predict_proba(preprocessor.transform(X_oot))[:, 1]
    reference = build_reference(X_train, y_oot, oot_scores, num_cols, cat_cols)

    # --- build + save model artefact ---
    model_artefact = build_artefact(
        config, model, preprocessor, best_params, auc, feature_cols, num_cols, cat_cols,
        X_train, X_test, X_oot, y_train, y_test, y_oot, reference,
    )
    pprint.pprint(model_artefact["results"])

    filepath = save_artefact(model_artefact, config)
    print("Model saved to:", filepath)

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)