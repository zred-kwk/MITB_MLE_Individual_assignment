import argparse
import os
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


# to call this script: python pipeline_stage.py --stage gold_feature --snapshotdate "2024-09-01"
#
# One data-pipeline stage for one snapshot date. These are the same util calls (and directories)
# that the backfill loops in data_processing_main.ipynb make, one date at a time.

BRONZE_LMS_DIR = "datamart/bronze/lms/"
BRONZE_CLICKSTREAM_DIR = "datamart/bronze/feature_clickstream/"
BRONZE_ATTRIBUTES_DIR = "datamart/bronze/feature_attributes/"
BRONZE_FINANCIALS_DIR = "datamart/bronze/feature_financials/"

SILVER_LOAN_DIR = "datamart/silver/loan_daily/"
SILVER_FINANCIALS_DIR = "datamart/silver/financials/"
SILVER_ATTRIBUTES_DIR = "datamart/silver/attributes/"
SILVER_CLICKSTREAM_DIR = "datamart/silver/clickstream/"

GOLD_LABEL_DIR = "datamart/gold/label_store/"
GOLD_FEATURE_DIR = "datamart/gold/feature_store/"

# stage name -> (output directory, function(date_str, spark))
STAGES = {
    "bronze_lms": (
        BRONZE_LMS_DIR,
        lambda d, spark: bronze.process_bronze_lms_table(d, BRONZE_LMS_DIR, spark)),
    "bronze_clickstream": (
        BRONZE_CLICKSTREAM_DIR,
        lambda d, spark: bronze.process_bronze_clickstream_table(d, BRONZE_CLICKSTREAM_DIR, spark)),
    "bronze_attributes": (
        BRONZE_ATTRIBUTES_DIR,
        lambda d, spark: bronze.process_bronze_attributes_table(d, BRONZE_ATTRIBUTES_DIR, spark)),
    "bronze_financials": (
        BRONZE_FINANCIALS_DIR,
        lambda d, spark: bronze.process_bronze_financials_table(d, BRONZE_FINANCIALS_DIR, spark)),

    "silver_loan": (
        SILVER_LOAN_DIR,
        lambda d, spark: silver.process_silver_loan_table(d, BRONZE_LMS_DIR, SILVER_LOAN_DIR, spark)),
    "silver_financials": (
        SILVER_FINANCIALS_DIR,
        lambda d, spark: silver.process_silver_financials_table(d, BRONZE_FINANCIALS_DIR, SILVER_FINANCIALS_DIR, spark)),
    "silver_attributes": (
        SILVER_ATTRIBUTES_DIR,
        lambda d, spark: silver.process_silver_attributes_table(d, BRONZE_ATTRIBUTES_DIR, SILVER_ATTRIBUTES_DIR, spark)),
    "silver_clickstream": (
        SILVER_CLICKSTREAM_DIR,
        lambda d, spark: silver.process_silver_clickstream_table(d, BRONZE_CLICKSTREAM_DIR, SILVER_CLICKSTREAM_DIR, spark)),

    "gold_label": (
        GOLD_LABEL_DIR,
        lambda d, spark: gold.process_labels_gold_table(d, SILVER_LOAN_DIR, GOLD_LABEL_DIR, spark, dpd=30, mob=6)),
    "gold_feature": (
        GOLD_FEATURE_DIR,
        lambda d, spark: gold.process_gold_feature_table(
            d, SILVER_LOAN_DIR, SILVER_FINANCIALS_DIR, SILVER_ATTRIBUTES_DIR, SILVER_CLICKSTREAM_DIR,
            GOLD_FEATURE_DIR, spark)),
}


def main(stage, snapshotdate):
    print("\n\n---starting job---\n\n")
    print("stage:", stage, "| snapshot date:", snapshotdate)

    datetime.strptime(snapshotdate, "%Y-%m-%d")  # fail early on a malformed date

    output_directory, run_stage = STAGES[stage]
    os.makedirs(output_directory, exist_ok=True)

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    try:
        run_stage(snapshotdate, spark)
    finally:
        spark.stop()

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run one data pipeline stage")
    parser.add_argument("--stage", type=str, required=True, choices=sorted(STAGES), help="pipeline stage")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.stage, args.snapshotdate)