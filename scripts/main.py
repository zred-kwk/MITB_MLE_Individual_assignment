import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------------------------------------------------
# Config: label store and feature store have independent date ranges
# ---------------------------------------------------------------------------

CONFIG = {
    "label":   {"start": "2023-01-01", "end": "2025-12-01"},
    "feature": {"start": "2023-01-01", "end": "2024-12-01"},
}

# Label parameters. A label for snapshot t is observed at t + MOB months.
DPD = 30
MOB = 6

# Labels and features are joined on loan_id, NOT on snapshot_date:
# a label at snapshot t describes the loan originated at t - MOB months,
# while the feature row is stamped with the origination date.
JOIN_KEYS = ["loan_id"]

# Minimum share of label rows that must have features, per snapshot month,
# on the months where features are expected to exist.
MIN_COVERAGE = 0.95


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))

        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_parquet_dir(folder_path):
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    return spark.read.option("header", "true").parquet(*files_list)


label_dates = generate_first_of_month_dates(CONFIG["label"]["start"], CONFIG["label"]["end"])
feature_dates = generate_first_of_month_dates(CONFIG["feature"]["start"], CONFIG["feature"]["end"])

# Loan data feeds both stores, so bronze lms and silver loan cover the union of both ranges.
loan_dates = sorted(set(label_dates) | set(feature_dates))

print("label dates:  ", label_dates[0], "->", label_dates[-1], f"({len(label_dates)})")
print("feature dates:", feature_dates[0], "->", feature_dates[-1], f"({len(feature_dates)})")
print("loan dates:   ", loan_dates[0], "->", loan_dates[-1], f"({len(loan_dates)})")


# ---------------------------------------------------------------------------
# Bronze layer
# ---------------------------------------------------------------------------

# --- lms (loan): shared by both stores -> union range ---
bronze_lms_directory = ensure_dir("datamart/bronze/lms/")

for date_str in loan_dates:
    utils.data_processing_bronze_table.process_bronze_lms_table(date_str, bronze_lms_directory, spark)

# --- clickstream: feature range only ---
bronze_clickstream_directory = ensure_dir("datamart/bronze/feature_clickstream/")

for date_str in feature_dates:
    utils.data_processing_bronze_table.process_bronze_clickstream_table(date_str, bronze_clickstream_directory, spark)

# --- attributes: feature range only ---
bronze_attributes_directory = ensure_dir("datamart/bronze/feature_attributes/")

for date_str in feature_dates:
    utils.data_processing_bronze_table.process_bronze_attributes_table(date_str, bronze_attributes_directory, spark)

# --- financials: feature range only ---
bronze_financials_directory = ensure_dir("datamart/bronze/feature_financials/")

for date_str in feature_dates:
    utils.data_processing_bronze_table.process_bronze_financials_table(date_str, bronze_financials_directory, spark)


# ---------------------------------------------------------------------------
# Silver layer
# ---------------------------------------------------------------------------

# --- loan: shared by both stores -> union range ---
silver_loan_daily_directory = ensure_dir("datamart/silver/loan_daily/")

for date_str in loan_dates:
    utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# --- financials: feature range only ---
silver_financials_directory = ensure_dir("datamart/silver/financials/")

for date_str in feature_dates:
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_financials_directory, silver_financials_directory, spark)

# --- attributes: feature range only ---
silver_attributes_directory = ensure_dir("datamart/silver/attributes/")

for date_str in feature_dates:
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)

# --- clickstream: feature range only ---
silver_clickstream_directory = ensure_dir("datamart/silver/clickstream/")

for date_str in feature_dates:
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)


# ---------------------------------------------------------------------------
# Gold layer: labels (label range only; depends on loan silver, not on features)
# ---------------------------------------------------------------------------

gold_label_store_directory = ensure_dir("datamart/gold/label_store/")

for date_str in label_dates:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=DPD, mob=MOB)

label_df = read_parquet_dir(gold_label_store_directory)
print("label row_count:", label_df.count())
label_df.show()


# ---------------------------------------------------------------------------
# Gold layer: feature store (feature range only; does not read the label store)
# ---------------------------------------------------------------------------

gold_feature_store_directory = ensure_dir("datamart/gold/feature_store/")

for date_str in feature_dates:
    utils.data_processing_gold_table.process_gold_feature_table(
        date_str,
        silver_loan_daily_directory,
        silver_financials_directory,
        silver_attributes_directory,
        silver_clickstream_directory,
        gold_feature_store_directory,
        spark,
    )

feature_df = read_parquet_dir(gold_feature_store_directory)
print("feature row_count:", feature_df.count())
feature_df.show()


# ---------------------------------------------------------------------------
# Coverage report: where labels and features overlap (the only place both stores meet)
# ---------------------------------------------------------------------------

# Bucket by ORIGINATION month = label snapshot - MOB, since features are keyed on it.
labels = label_df.select(
    *JOIN_KEYS,
    F.date_format(F.add_months(F.col("snapshot_date"), -MOB), "yyyy-MM").alias("origination_month"),
).distinct()
feature_keys = feature_df.select(*JOIN_KEYS).distinct().withColumn("has_features", F.lit(1))

coverage = (
    labels
    .join(feature_keys, on=JOIN_KEYS, how="left")
    .groupBy("origination_month")
    .agg(
        F.count("*").alias("n_labels"),
        F.sum(F.coalesce(F.col("has_features"), F.lit(0))).alias("n_with_features"),
    )
    .withColumn("coverage", F.round(F.col("n_with_features") / F.col("n_labels"), 4))
    .orderBy("origination_month")
)
coverage.show(100, truncate=False)

feature_months = sorted({d[:7] for d in feature_dates})

# Origination months inside the feature range must be covered.
bad = coverage.filter(
    F.col("origination_month").isin(feature_months) & (F.col("coverage") < MIN_COVERAGE)
).collect()
if bad:
    raise ValueError(
        "Feature coverage below threshold inside the feature range: "
        + ", ".join(f"{r['origination_month']}={r['coverage']}" for r in bad)
    )

# Origination months outside the feature range cannot be trained on; report them explicitly.
outside = coverage.filter(~F.col("origination_month").isin(feature_months)).collect()
if outside:
    print("NOTE: labels with no feature partitions (excluded from training):",
          ", ".join(r["origination_month"] for r in outside))

last_usable_label_snapshot = (
    datetime.strptime(feature_dates[-1], "%Y-%m-%d") + relativedelta(months=MOB)
).strftime("%Y-%m-%d")
print("latest label snapshot with features:", last_usable_label_snapshot)