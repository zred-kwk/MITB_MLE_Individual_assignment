import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")  # Save plots in Docker/Airflow without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/app")

# A feature whose missing rate rises by more than this versus training is
# flagged as an availability failure (e.g. 0.0 at training -> 1.0 now).
MISSING_RATE_ALERT = 0.5

# PSI / CSI conventional cut-offs (moderate / significant shift).
PSI_WARN, PSI_ALERT = 0.10, 0.25


# ---------------------------------------------------------------------
# PSI and data-loading helpers
# ---------------------------------------------------------------------

def psi(expected_pct, actual_pct, eps=1e-6):
    """Calculate Population Stability Index for two distributions."""
    expected_pct = np.asarray(expected_pct, dtype=float)
    actual_pct = np.asarray(actual_pct, dtype=float)

    if expected_pct.shape != actual_pct.shape:
        raise ValueError(
            f"PSI distributions have different shapes: "
            f"{expected_pct.shape} vs {actual_pct.shape}"
        )

    if expected_pct.size == 0:
        return np.nan

    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )
    )


def binned_pct(values, edges):
    """
    Histogram proportions over `edges`. Values outside the reference range
    are clipped into the outer bins, so tail drift is counted rather than
    silently dropped by np.histogram.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    clipped = np.clip(values, edges[0], edges[-1])
    counts = np.histogram(clipped, bins=edges)[0]
    total = counts.sum()
    return counts / total if total > 0 else None


def read_parquet_to_pandas(spark, path, description):
    """Read a parquet file and raise a useful error if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {description} parquet: {path}"
        )

    return spark.read.parquet(path).toPandas()


def validate_unique_ids(df, description):
    """Prevent duplicate loan IDs from inflating monitoring results."""
    if df["loan_id"].isna().any():
        raise ValueError(f"{description} contains null loan_id values.")

    if df["loan_id"].duplicated().any():
        examples = (
            df.loc[df["loan_id"].duplicated(), "loan_id"]
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"{description} contains duplicate loan_id values. "
            f"Examples: {examples}"
        )


def load_roc_reference(reference):
    """
    Support the common reference formats:
      1. {'fpr': [...], 'tpr': [...]}
      2. {'fpr': [...], 'tpr': [...], ...}
      3. A DataFrame with fpr and tpr columns.
    """
    if reference is None:
        return None

    if isinstance(reference, pd.DataFrame):
        if {"fpr", "tpr"}.issubset(reference.columns):
            return (
                pd.to_numeric(reference["fpr"], errors="coerce").to_numpy(),
                pd.to_numeric(reference["tpr"], errors="coerce").to_numpy(),
            )
        return None

    if isinstance(reference, dict):
        if "fpr" in reference and "tpr" in reference:
            return (
                np.asarray(reference["fpr"], dtype=float),
                np.asarray(reference["tpr"], dtype=float),
            )

    return None


# ---------------------------------------------------------------------
# Model timeline helpers (everything is relative to the model artefact)
# ---------------------------------------------------------------------

def infer_train_date(model_stem, art):
    """
    Training date T of the model: the artefact's own config
    (data_dates["model_train_date"]), else parsed from the filename
    (credit_model_YYYY_MM_DD), else other artefact keys.
    """
    train_date = (art.get("data_dates") or {}).get("model_train_date")
    if train_date is not None:
        return pd.to_datetime(train_date)

    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", model_stem)
    if m:
        return pd.Timestamp(
            year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3))
        )

    for key in ("model_date", "train_snapshot_date", "snapshot_date"):
        if key in art:
            return pd.to_datetime(art[key])

    raise ValueError(
        f"Cannot infer training date from '{model_stem}' or artefact keys."
    )


def first_unseen_cohort(train_date, horizon_months):
    """
    Labels are dated origination + horizon and training keeps labels dated
    before T, so the first origination cohort the model never saw is
    T - horizon (T=2024-09-01, horizon=6 -> 2024-03-01).
    """
    return train_date - pd.DateOffset(months=horizon_months)


def eligible_models(snapshotdate, horizon_months, deployed_only=False):
    """
    Artefacts in model_bank whose first unseen cohort is on or before the
    snapshot (T - horizon <= S). With deployed_only, also require T <= S,
    i.e. skip models that were trained after the snapshot (relevant when
    backfilling old snapshots).
    """
    snapshot = pd.to_datetime(snapshotdate, format="%Y-%m-%d")
    models = []
    for path in sorted(glob.glob(os.path.join(PROJECT_DIR, "model_bank", "*.pkl"))):
        name = os.path.basename(path)
        m = re.search(r"(\d{4})_(\d{2})_(\d{2})", name)
        if not m:
            continue
        train_date = pd.Timestamp(
            year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3))
        )
        if first_unseen_cohort(train_date, horizon_months) > snapshot:
            continue
        if deployed_only and train_date > snapshot:
            continue
        models.append(name)
    return models


def resolve_performance_cohort(snapshot, first_cohort, horizon_months,
                               override=None):
    """
    Labels observed at `snapshot` belong to the cohort originated
    `horizon_months` earlier. Return that cohort date, or None if it is
    before the first unseen cohort (it was part of training).
    """
    if override:
        cohort = pd.to_datetime(override, format="%Y-%m-%d")
    else:
        cohort = snapshot - pd.DateOffset(months=horizon_months)

    return cohort if cohort >= first_cohort else None


def load_previous_scores(pred_dir, model_stem, snapshot, first_cohort):
    """
    Valid scores from the previous month for the SAME model, or None if
    the previous month is before the first unseen cohort or absent.
    """
    prev = snapshot - pd.DateOffset(months=1)
    if prev < first_cohort:
        return None

    path = os.path.join(
        pred_dir,
        f"{model_stem}_predictions_{prev.strftime('%Y_%m_%d')}.parquet",
    )
    if not os.path.exists(path):
        return None

    s = pd.read_parquet(path)["model_predictions"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[np.isfinite(s) & s.between(0, 1)]
    return s if len(s) else None


# ---------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------

def compute_feature_drift(ref_features, drift_features, snapshotdate):
    """Feature drift of the scored loans against the TRAINING reference."""
    covariate_rows = []

    for feature, reference in ref_features.items():
        common = {
            "snapshot_date": snapshotdate,
            "feature": feature,
            "mean_ref": reference.get("mean", np.nan),
            "mean_act": np.nan,
            "median_ref": reference.get("median", np.nan),
            "median_act": np.nan,
            "var_ref": reference.get("var", np.nan),
            "var_act": np.nan,
            "smd_mean": np.nan,
            "smd_median": np.nan,
            "var_ratio": np.nan,
            "missing_rate_ref": reference.get("missing_rate", np.nan),
            "missing_rate_act": np.nan,
            "status": "ok",
        }

        if feature not in drift_features.columns:
            covariate_rows.append({
                **common, "n": 0, "psi": np.nan,
                "status": "feature_not_in_feature_store",
            })
            continue

        raw = drift_features[feature]
        common["missing_rate_act"] = float(raw.isna().mean())

        if reference.get("kind") == "categorical":
            categories = list(reference.get("categories", []))
            expected_pct = np.asarray(reference.get("pct", []), dtype=float)

            if len(categories) != len(expected_pct):
                covariate_rows.append({
                    **common, "n": int(raw.notna().sum()), "psi": np.nan,
                    "status": "invalid_categorical_reference",
                })
                continue

            expected_categories = categories + ["__OTHER__"]
            expected = np.append(expected_pct, reference.get("other_pct", 0.0))
            actual = raw.astype("string").fillna("__MISSING__")
            mapped = actual.where(actual.isin(categories), "__OTHER__")
            actual_pct = (
                mapped.value_counts(normalize=True)
                .reindex(expected_categories, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            expected = expected / expected.sum() if expected.sum() else expected

            covariate_rows.append({
                **common, "n": int(raw.notna().sum()),
                "psi": psi(expected, actual_pct) if actual_pct.sum() else np.nan,
                "status": "ok",
            })
            continue

        feature_all = pd.to_numeric(raw, errors="coerce")
        x = feature_all.dropna().astype(float)
        common["missing_rate_act"] = float(feature_all.isna().mean())

        if x.empty:
            covariate_rows.append({
                **common, "n": 0, "psi": np.nan,
                "status": "all_values_missing",
            })
            continue

        edges = np.asarray(reference.get("edges", []), dtype=float)
        expected_pct = np.asarray(reference.get("pct", []), dtype=float)

        if len(edges) != len(expected_pct) + 1 or len(expected_pct) == 0:
            covariate_rows.append({
                **common, "n": int(len(x)), "psi": np.nan,
                "status": "invalid_numeric_reference",
            })
            continue

        actual_pct = binned_pct(x.to_numpy(), edges)
        var_act = x.var(ddof=1) if len(x) > 1 else np.nan
        mean_ref = reference.get("mean", np.nan)
        median_ref = reference.get("median", np.nan)
        var_ref = reference.get("var", np.nan)
        std_ref = reference.get("std", np.nan)

        common.update({
            "mean_act": float(x.mean()),
            "median_act": float(x.median()),
            "var_act": float(var_act) if pd.notna(var_act) else np.nan,
            "smd_mean": float((x.mean() - mean_ref) / std_ref)
            if pd.notna(std_ref) and std_ref > 0 and pd.notna(mean_ref)
            else np.nan,
            "smd_median": float((x.median() - median_ref) / std_ref)
            if pd.notna(std_ref) and std_ref > 0 and pd.notna(median_ref)
            else np.nan,
            "var_ratio": float(var_act / var_ref)
            if pd.notna(var_act) and pd.notna(var_ref) and var_ref > 0
            else np.nan,
        })

        covariate_rows.append({
            **common, "n": int(len(x)),
            "psi": psi(expected_pct, actual_pct)
            if actual_pct is not None else np.nan,
            "status": "ok",
        })

    return pd.DataFrame(covariate_rows)


def summarise_covariates(covariates):
    """Roll the per-feature table up into run-level metrics and data-quality flags."""
    out = {
        "n_features_monitored": 0,
        "max_feature_psi": np.nan,
        "n_features_psi_gt_0.25": 0,
        "n_features_all_missing": 0,
        "max_missing_rate_delta": np.nan,
        "n_features_missing_shift": 0,
        "data_quality_status": "ok",
    }
    if covariates.empty:
        return out

    delta = (
        pd.to_numeric(covariates["missing_rate_act"], errors="coerce")
        - pd.to_numeric(covariates["missing_rate_ref"], errors="coerce")
    )
    n_shift = int((delta > MISSING_RATE_ALERT).sum())
    # A feature absent from the feature store has no missing rate (NaN), so it
    # never appears in delta. Count it as an availability failure explicitly.
    n_absent = int(
        (covariates["status"] == "feature_not_in_feature_store").sum()
    )

    out.update({
        "n_features_monitored": int(len(covariates)),
        # NOTE: covers only features with data; all-missing ones have no PSI.
        "max_feature_psi": covariates["psi"].max(),
        "n_features_psi_gt_0.25": int((covariates["psi"] > 0.25).sum()),
        "n_features_all_missing": int(
            (covariates["status"] == "all_values_missing").sum()
        ),
        "max_missing_rate_delta": (
            float(delta.max()) if delta.notna().any() else np.nan
        ),
        "n_features_missing_shift": n_shift,
        "data_quality_status": (
            "feature_availability_failure"
            if (n_shift + n_absent) > 0 else "ok"
        ),
    })
    return out


# ---------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------

def compute_performance(spark, pred_dir, model_stem, perf_ts, labels,
                        labels_path, snapshotdate):
    """AUC/ROC of the cohort scored at perf_ts against labels observed at the snapshot."""
    out = {
        "performance_status": "ok",
        "n_performance_predictions": 0,
        "n_labels_matched": 0,
        "n_labels_unmatched": 0,
        "n_labelled_valid": 0,
        "bad_rate": np.nan,
        "auc": np.nan,
        "gini": np.nan,
    }

    if perf_ts is None:
        out["performance_status"] = "cohort_in_training"
        print(
            "Performance skipped: the cohort for this snapshot is before "
            "the first unseen cohort (it was part of training)."
        )
        return out, None

    if labels is None:
        out["performance_status"] = "labels_missing"
        print(f"Performance skipped: {labels_path} not found.")
        return out, None

    performance_date = perf_ts.strftime("%Y-%m-%d")
    performance_ym = performance_date.replace("-", "_")
    performance_preds_path = os.path.join(
        pred_dir, f"{model_stem}_predictions_{performance_ym}.parquet"
    )

    if not os.path.exists(performance_preds_path):
        out["performance_status"] = "cohort_predictions_missing"
        print(f"Performance skipped: {performance_preds_path} not found.")
        return out, None

    performance_preds = read_parquet_to_pandas(
        spark, performance_preds_path, "performance-cohort predictions"
    )

    if not {"loan_id", "model_predictions"}.issubset(performance_preds.columns):
        raise ValueError(
            "Performance predictions must contain loan_id and "
            "model_predictions."
        )
    validate_unique_ids(performance_preds, "Performance predictions")

    performance_df = performance_preds[["loan_id", "model_predictions"]].merge(
        labels[["loan_id", "label"]],
        on="loan_id",
        how="left",
        validate="one_to_one",
        indicator="_label_match",
    )

    n_matched = int((performance_df["_label_match"] == "both").sum())
    n_unmatched = int((performance_df["_label_match"] == "left_only").sum())
    performance_df = performance_df.drop(columns=["_label_match"])

    for c in ("label", "model_predictions"):
        performance_df[c] = pd.to_numeric(performance_df[c], errors="coerce")

    performance_df = performance_df.dropna(subset=["label", "model_predictions"])
    performance_df = performance_df[
        performance_df["label"].isin([0, 1])
        & performance_df["model_predictions"].between(0, 1)
    ]

    out.update({
        "n_performance_predictions": int(len(performance_preds)),
        "n_labels_matched": n_matched,
        "n_labels_unmatched": n_unmatched,
        "n_labelled_valid": int(len(performance_df)),
    })

    if len(performance_df) > 0 and performance_df["label"].nunique() == 2:
        y_true = performance_df["label"].astype(int)
        y_score = performance_df["model_predictions"].astype(float)

        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        out.update({
            "bad_rate": float(y_true.mean()),
            "auc": float(auc),
            "gini": float(2 * auc - 1),
        })

        grid = np.linspace(0, 1, 101)
        current_roc = pd.DataFrame({
            "label_observation_date": snapshotdate,
            "performance_prediction_date": performance_date,
            "fpr": grid,
            "tpr": np.interp(grid, fpr, tpr),
        })
        return out, current_roc

    out["performance_status"] = (
        "no_valid_labelled_predictions"
        if len(performance_df) == 0
        else "single_class"
    )
    print(f"AUC/ROC skipped: {out['performance_status']}.")
    return out, None


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_monitoring_plots(
    output_dir,
    model_stem,
    reference_roc=None,
    current_roc=None,
    reference_auc=None,
):
    """
    Save the reference vs deployed ROC overlay and the AUC history.

    The deployed curves are rebuilt from every saved roc_*.parquet file (one
    per matured cohort), so the overlay does not depend on whether the
    CURRENT run happened to have a matured cohort.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    has_curve = False

    parsed_reference = load_roc_reference(reference_roc)

    if parsed_reference is not None:
        ref_fpr, ref_tpr = parsed_reference
        valid = np.isfinite(ref_fpr) & np.isfinite(ref_tpr)

        if valid.any():
            ax.plot(
                ref_fpr[valid],
                ref_tpr[valid],
                color="black",
                lw=2,
                label=(
                    f"Reference OOT ROC (AUC={reference_auc:.3f})"
                    if reference_auc is not None and np.isfinite(reference_auc)
                    else "Reference OOT ROC"
                ),
            )
            has_curve = True

    # Overlay every matured cohort's deployed ROC from the saved history.
    roc_frames = []
    for path in sorted(glob.glob(os.path.join(output_dir, "roc_*.parquet"))):
        try:
            roc_frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"Could not read ROC history {path}: {exc}")

    # The current run's curve is normally already saved (and so already in
    # the history); adding it again is harmless because of the de-dup below.
    if current_roc is not None and not current_roc.empty:
        roc_frames.append(current_roc)

    if roc_frames:
        rocs = (
            pd.concat(roc_frames, ignore_index=True)
            # Reruns of a snapshot write extra files; keep the latest.
            .drop_duplicates(["label_observation_date", "fpr"], keep="last")
        )
        for _, g in rocs.groupby("label_observation_date"):
            g = g.sort_values("fpr")
            ax.plot(
                g["fpr"],
                g["tpr"],
                alpha=0.75,
                lw=1.2,
                label=(
                    f"Deployed ROC (cohort "
                    f"{g['performance_prediction_date'].iloc[0]})"
                ),
            )
        has_curve = True

    if has_curve:
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
                label="Random classifier")
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC Curves — {model_stem}",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "roc_comparison.png"), dpi=150)
    else:
        print("No valid reference ROC curve or current ROC curve to plot.")

    plt.close(fig)

    # Observed AUC across monitoring runs vs the OOT reference AUC.
    history = []
    for path in glob.glob(os.path.join(output_dir, "metrics_*.parquet")):
        try:
            history.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"Could not read metrics history {path}: {exc}")

    observed = pd.DataFrame(columns=["snapshot_date", "auc"])
    if history:
        metrics = pd.concat(history, ignore_index=True)
        if {"snapshot_date", "auc"}.issubset(metrics.columns):
            metrics["auc"] = pd.to_numeric(metrics["auc"], errors="coerce")
            metrics["snapshot_date"] = pd.to_datetime(
                metrics["snapshot_date"], errors="coerce"
            )
            # Reruns of a snapshot write extra metrics files. Sorting on
            # inference_id as well makes "keep last" deterministic.
            sort_cols = [
                c for c in ("snapshot_date", "inference_id")
                if c in metrics.columns
            ]
            observed = (
                metrics.dropna(subset=["snapshot_date", "auc"])
                .sort_values(sort_cols)
                .drop_duplicates(subset=["snapshot_date"], keep="last")
            )

    has_ref = reference_auc is not None and np.isfinite(reference_auc)

    if not observed.empty or has_ref:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if not observed.empty:
            ax.plot(
                observed["snapshot_date"],
                observed["auc"],
                marker="o",
                label="Observed cohort AUC",
            )
        else:
            ax.text(
                0.5, 0.5,
                "No matured labels for unseen cohorts yet",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_xticks([])

        if has_ref:
            ax.axhline(
                reference_auc,
                linestyle="--",
                label=f"Reference OOT AUC ({reference_auc:.3f})",
            )

        ax.set(
            xlabel="Monitoring snapshot date (label observation)",
            ylabel="ROC AUC",
            title=f"AUC Monitoring — {model_stem}",
            ylim=(0, 1),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "auc_by_snapshot.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------
# Drift plotting (PSI / CSI history, rebuilt from the saved parquet files)
# ---------------------------------------------------------------------

def _load_history(output_dir, prefix):
    """Concatenate <prefix>_*.parquet; inference_id is parsed from the filename."""
    frames = []
    for path in sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.parquet"))):
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            continue
        # <prefix>_YYYY_MM_DD_<inference_id>.parquet
        df["inference_id"] = os.path.splitext(os.path.basename(path))[0].split("_", 4)[-1]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    return out.dropna(subset=["snapshot_date"])


def _psi_thresholds(ax):
    ax.axhline(PSI_WARN, ls="--", lw=1, color="orange", label=f"{PSI_WARN} (moderate)")
    ax.axhline(PSI_ALERT, ls="--", lw=1, color="red", label=f"{PSI_ALERT} (significant)")


def save_drift_plots(output_dir, model_stem, top_n=20):
    """Score PSI, max CSI and per-feature CSI heatmap across snapshots."""
    # --- run-level: score PSI and max feature CSI ---
    m = _load_history(output_dir, "metrics")
    if not m.empty:
        # Reruns write extra files; keep the latest inference_id per snapshot.
        m = (m.sort_values(["snapshot_date", "inference_id"])
               .drop_duplicates("snapshot_date", keep="last"))
        for c in ("score_psi", "score_psi_vs_prev_month", "max_feature_psi"):
            if c in m:
                m[c] = pd.to_numeric(m[c], errors="coerce")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(m["snapshot_date"], m["score_psi"], marker="o",
                label="Score PSI (vs training)")
        ax.plot(m["snapshot_date"], m["score_psi_vs_prev_month"], marker="s",
                label="Score PSI (vs previous month)")
        ax.plot(m["snapshot_date"], m["max_feature_psi"], marker="^",
                label="Max feature CSI (vs training)")
        _psi_thresholds(ax)
        ax.set(xlabel="Monitoring snapshot date", ylabel="PSI",
               title=f"Score PSI and max CSI — {model_stem}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "psi_by_snapshot.png"), dpi=150)
        plt.close(fig)

    # --- feature-level: CSI heatmap, features x snapshots ---
    c = _load_history(output_dir, "covariate")
    if not c.empty:
        c = (c.sort_values(["snapshot_date", "inference_id"])
               .drop_duplicates(["snapshot_date", "feature"], keep="last"))
        c["psi"] = pd.to_numeric(c["psi"], errors="coerce")

        pivot = c.pivot(index="feature", columns="snapshot_date", values="psi")
        order = pivot.max(axis=1).sort_values(ascending=False).head(top_n).index
        pivot = pivot.loc[order]

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1] + 4),
                                        max(3, 0.35 * len(pivot) + 1.5)))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto",
                       cmap="YlOrRd", vmin=0, vmax=PSI_ALERT)  # saturates at 0.25
        ax.set_yticks(range(len(pivot)), pivot.index)
        ax.set_xticks(range(pivot.shape[1]),
                      [d.strftime("%Y-%m") for d in pivot.columns],
                      rotation=45, ha="right")
        ax.set_title(f"Feature CSI (top {len(pivot)} by max) — {model_stem}")
        fig.colorbar(im, ax=ax, label="PSI (clipped at 0.25)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "csi_heatmap.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------

def main(
    snapshotdate,
    modelname,
    performance_prediction_date=None,
    inference_id=None,
    label_horizon_months=6,
):
    """
    Monitor one snapshot against one model artefact.

    Timeline (all read from the artefact; nothing is hard-coded):
        T           = training date (artefact data_dates / filename)
        first cohort = T - horizon, the first origination month the model
                       never saw (labels dated >= T are excluded from training)

    For a snapshot S:
      Drift        : features and scores of loans originated at S vs the
                     training reference.            needs predictions + features at S
      Stability    : scores at S vs scores at S - 1 month (same model).
      Performance  : scores of cohort S - horizon vs labels observed at S;
                     only if that cohort >= first cohort.   needs labels at S

    Drift and performance are independent: if there are no originations at S
    (no features/predictions) drift is skipped and performance still runs.

    Assumptions:
        model_predictions is the positive-class probability.
        label = 1 denotes the bad/event outcome.
    """
    snapshot_ts = pd.to_datetime(snapshotdate, format="%Y-%m-%d", errors="raise")
    snapshot_ym = snapshotdate.replace("-", "_")
    inference_id = inference_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    model_filename = os.path.basename(modelname)
    model_stem = os.path.splitext(model_filename)[0]
    model_path = os.path.join(PROJECT_DIR, "model_bank", model_filename)

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model_monitor")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # -------------------------------------------------------------
        # 1. Load the model artefact and derive its timeline.
        # -------------------------------------------------------------
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artefact not found: {model_path}")

        with open(model_path, "rb") as f:
            art = pickle.load(f)

        if "reference" not in art:
            raise KeyError(
                f"The artefact {model_path} has no 'reference'. "
                "Regenerate the artefact using the updated training "
                "notebook."
            )

        ref = art["reference"]

        required_ref_keys = {"score_edges", "score_pct", "features"}
        missing_ref_keys = required_ref_keys - set(ref)
        if missing_ref_keys:
            raise KeyError(
                "Missing reference keys in model artefact: "
                f"{sorted(missing_ref_keys)}"
            )

        score_edges = np.asarray(ref["score_edges"], dtype=float)
        score_pct = np.asarray(ref["score_pct"], dtype=float)

        if len(score_edges) != len(score_pct) + 1:
            raise ValueError(
                "Reference score_edges must contain exactly one more "
                "element than score_pct."
            )

        train_date = infer_train_date(model_stem, art)
        first_cohort = first_unseen_cohort(train_date, label_horizon_months)

        if snapshot_ts < first_cohort:
            raise ValueError(
                f"Snapshot {snapshotdate} precedes the first unseen cohort "
                f"{first_cohort.date()} (train date {train_date.date()} minus "
                f"{label_horizon_months}m) for {model_filename}."
            )

        perf_ts = resolve_performance_cohort(
            snapshot_ts,
            first_cohort,
            label_horizon_months,
            performance_prediction_date,
        )
        performance_date = (
            perf_ts.strftime("%Y-%m-%d") if perf_ts is not None else None
        )

        # -------------------------------------------------------------
        # 2. Paths at the snapshot.
        # -------------------------------------------------------------
        pred_dir = os.path.join(
            PROJECT_DIR, "datamart", "gold", "model_predictions", model_stem
        )
        drift_preds_path = os.path.join(
            pred_dir, f"{model_stem}_predictions_{snapshot_ym}.parquet"
        )
        features_path = os.path.join(
            PROJECT_DIR, "datamart", "gold", "feature_store",
            f"gold_feature_store_{snapshot_ym}.parquet",
        )
        labels_path = os.path.join(
            PROJECT_DIR, "datamart", "gold", "label_store",
            f"gold_label_store_{snapshot_ym}.parquet",
        )

        metrics = {
            "snapshot_date": snapshotdate,
            "inference_id": inference_id,
            "model": model_filename,
            "model_version": art.get("model_version", model_stem),
            "model_train_date": train_date.strftime("%Y-%m-%d"),
            "model_date": first_cohort.strftime("%Y-%m-%d"),  # first unseen cohort
            "months_since_first_cohort": int(
                (snapshot_ts.year - first_cohort.year) * 12
                + (snapshot_ts.month - first_cohort.month)
            ),
            "drift_status": "ok",
            "n_predictions": 0,
            "n_valid_scores": 0,
            "n_features_matched": 0,
            "n_features_unmatched": 0,
            "score_psi": np.nan,
            "score_psi_vs_prev_month": np.nan,
            "n_features_monitored": 0,
            "max_feature_psi": np.nan,
            "n_features_psi_gt_0.25": 0,
            "n_features_all_missing": 0,
            "max_missing_rate_delta": np.nan,
            "n_features_missing_shift": 0,
            "data_quality_status": "not_evaluated",
        }
        covariates = pd.DataFrame()

        # -------------------------------------------------------------
        # 3. Drift and stability (needs predictions + features at S).
        # -------------------------------------------------------------
        if not os.path.exists(drift_preds_path):
            metrics["drift_status"] = "predictions_missing"
            print(f"Drift skipped: {drift_preds_path} not found.")
        elif not os.path.exists(features_path):
            metrics["drift_status"] = "features_missing"
            print(f"Drift skipped: {features_path} not found.")
        else:
            drift_preds = read_parquet_to_pandas(
                spark, drift_preds_path, "drift predictions"
            )
            features = read_parquet_to_pandas(spark, features_path, "features")

            if not {"loan_id", "model_predictions"}.issubset(drift_preds.columns):
                raise ValueError(
                    "Predictions must contain loan_id and model_predictions."
                )
            if "loan_id" not in features.columns:
                raise ValueError("Feature store must contain loan_id.")

            validate_unique_ids(drift_preds, "Drift predictions")
            validate_unique_ids(features, "Feature store")

            # Keep features aligned with loans actually scored.
            drift_features = drift_preds[["loan_id"]].merge(
                features,
                on="loan_id",
                how="left",
                validate="one_to_one",
                indicator="_feature_match",
            )
            n_feature_matches = int(
                (drift_features["_feature_match"] == "both").sum()
            )
            n_feature_unmatched = int(
                (drift_features["_feature_match"] == "left_only").sum()
            )
            drift_features = drift_features.drop(columns=["_feature_match"])

            # Score drift (vs training) and stability (vs previous month).
            scores = pd.to_numeric(
                drift_preds["model_predictions"], errors="coerce"
            )
            valid_scores = scores.dropna()
            valid_scores = valid_scores[
                np.isfinite(valid_scores) & valid_scores.between(0, 1)
            ]

            metrics.update({
                "n_predictions": int(len(drift_preds)),
                "n_valid_scores": int(len(valid_scores)),
                "n_features_matched": n_feature_matches,
                "n_features_unmatched": n_feature_unmatched,
            })

            if len(valid_scores) and len(score_pct):
                actual_score_pct = binned_pct(valid_scores, score_edges)
                if actual_score_pct is not None:
                    metrics["score_psi"] = psi(score_pct, actual_score_pct)

                prev_scores = load_previous_scores(
                    pred_dir, model_stem, snapshot_ts, first_cohort
                )
                if prev_scores is not None:
                    prev_pct = binned_pct(prev_scores, score_edges)
                    if prev_pct is not None and actual_score_pct is not None:
                        metrics["score_psi_vs_prev_month"] = psi(
                            prev_pct, actual_score_pct
                        )

            # Feature drift against the TRAINING reference.
            covariates = compute_feature_drift(
                ref["features"], drift_features, snapshotdate
            )
            metrics.update(summarise_covariates(covariates))

        # -------------------------------------------------------------
        # 4. Performance (needs labels at S + cohort S - horizon scored).
        # -------------------------------------------------------------
        labels = None
        if perf_ts is not None and os.path.exists(labels_path):
            labels = read_parquet_to_pandas(spark, labels_path, "labels")
            if not {"loan_id", "label"}.issubset(labels.columns):
                raise ValueError("Label store must contain loan_id and label.")
            validate_unique_ids(labels, "Label store")

        perf_metrics, current_roc = compute_performance(
            spark, pred_dir, model_stem, perf_ts, labels, labels_path,
            snapshotdate,
        )

        metrics.update({
            "label_horizon_months": label_horizon_months,
            "performance_prediction_date": performance_date,
            "label_observation_date": snapshotdate,
            **perf_metrics,
            "auc_oot_ref": art.get("results", {}).get("auc_oot", np.nan),
        })

        # -------------------------------------------------------------
        # 5. Save monitoring outputs.
        # -------------------------------------------------------------
        output_dir = os.path.join(
            PROJECT_DIR, "datamart", "gold", "model_monitoring", model_stem
        )
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = os.path.join(
            output_dir, f"metrics_{snapshot_ym}_{inference_id}.parquet"
        )
        covariates_path = os.path.join(
            output_dir, f"covariate_{snapshot_ym}_{inference_id}.parquet"
        )

        pd.DataFrame([metrics]).to_parquet(metrics_path, index=False)
        if not covariates.empty:
            covariates.to_parquet(covariates_path, index=False)

        if current_roc is not None:
            roc_path = os.path.join(
                output_dir, f"roc_{snapshot_ym}_{inference_id}.parquet"
            )
            current_roc.to_parquet(roc_path, index=False)

        save_monitoring_plots(
            output_dir=output_dir,
            model_stem=model_stem,
            reference_roc=ref.get("roc_oot"),
            current_roc=current_roc,
            reference_auc=float(
                art.get("results", {}).get("auc_oot", np.nan)
            ),
        )

        try:
            save_drift_plots(output_dir=output_dir, model_stem=model_stem)
        except Exception as exc:  # metrics are already saved; don't fail the task
            print(f"Drift plots skipped: {exc!r}")

        print("\nMonitoring metrics:")
        print(pd.Series(metrics).to_string())
        print(f"\nMetrics saved to: {metrics_path}")
        if not covariates.empty:
            print(f"Covariates saved to: {covariates_path}")
        print(f"Plots saved under: {output_dir}")

    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Monitor drift, stability and labelled performance of a model "
            "on cohorts it never saw in training."
        )
    )

    parser.add_argument(
        "--snapshotdate",
        required=True,
        help="Monitoring snapshot and label observation date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--modelname",
        default=None,
        help="Model artefact filename, e.g. credit_model_2024_09_01.pkl",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Monitor every artefact in model_bank whose first unseen cohort "
            "is on or before the snapshot, instead of a single --modelname."
        ),
    )
    parser.add_argument(
        "--deployed-only",
        action="store_true",
        help="With --all-models, skip models trained after the snapshot.",
    )
    parser.add_argument(
        "--inference-id",
        default=None,
        help="Unique inference/batch ID. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--performance-prediction-date",
        default=None,
        help=(
            "Optional override for the cohort used in AUC/ROC. Defaults to "
            "snapshotdate minus --label-horizon-months. Must be on or after "
            "the first unseen cohort."
        ),
    )
    parser.add_argument(
        "--label-horizon-months",
        type=int,
        default=6,
        help="Months between origination and label observation (default 6).",
    )

    args = parser.parse_args()

    if bool(args.modelname) == bool(args.all_models):
        parser.error("Provide exactly one of --modelname or --all-models.")

    if not args.all_models:
        main(
            snapshotdate=args.snapshotdate,
            modelname=args.modelname,
            performance_prediction_date=args.performance_prediction_date,
            inference_id=args.inference_id,
            label_horizon_months=args.label_horizon_months,
        )
    else:
        models = eligible_models(
            args.snapshotdate, args.label_horizon_months, args.deployed_only
        )
        if not models:
            raise SystemExit(
                f"No eligible models in {PROJECT_DIR}/model_bank for "
                f"snapshot {args.snapshotdate}."
            )
        print(f"Monitoring {len(models)} model(s): {models}")

        failures = []
        for name in models:
            print(f"\n===== {name} @ {args.snapshotdate} =====")
            try:
                main(
                    snapshotdate=args.snapshotdate,
                    modelname=name,
                    performance_prediction_date=args.performance_prediction_date,
                    inference_id=args.inference_id,
                    label_horizon_months=args.label_horizon_months,
                )
            except Exception as exc:  # one bad model must not hide the others
                failures.append((name, repr(exc)))
                print(f"FAILED {name}: {exc!r}")

        if failures:
            print("\nFailures:")
            for name, err in failures:
                print(f"  {name}: {err[:200]}")
            raise SystemExit(1)