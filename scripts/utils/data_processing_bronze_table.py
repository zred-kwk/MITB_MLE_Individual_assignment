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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def _process_bronze_table(snapshot_date_str, csv_file_path, directory, file_prefix, spark):
    """
    Shared bronze ingestion: read the source CSV, keep one snapshot_date, save as a CSV partition.

    Raises if the source has no rows for the snapshot date, so an empty month
    stops the run here instead of becoming an empty partition downstream.
    """
    # connect to source back end - IRL connect to back end source system
    # load data - IRL ingest from back end source system
    # Cast both sides to date: avoids silent zero-row matches when inferSchema
    # reads snapshot_date as a string, or as a timestamp with a time component.
    df = (
        spark.read.csv(csv_file_path, header=True, inferSchema=True)
        .filter(F.to_date(col("snapshot_date")) == F.to_date(F.lit(snapshot_date_str)))
    )

    n = df.count()
    print(snapshot_date_str, csv_file_path, "row count:", n)
    if n == 0:
        raise ValueError(f"No rows in source for {snapshot_date_str}: {csv_file_path}")

    # save bronze table to datamart - IRL connect to database to write
    partition_name = f"{file_prefix}_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return df


def process_bronze_lms_table(snapshot_date_str, bronze_lms_directory, spark):
    return _process_bronze_table(
        snapshot_date_str, "data/lms_loan_daily.csv", bronze_lms_directory, "bronze_loan_daily", spark
    )


def process_bronze_clickstream_table(snapshot_date_str, bronze_clickstream_directory, spark):
    return _process_bronze_table(
        snapshot_date_str, "data/feature_clickstream.csv", bronze_clickstream_directory, "bronze_clickstream", spark
    )


def process_bronze_attributes_table(snapshot_date_str, bronze_attributes_directory, spark):
    return _process_bronze_table(
        snapshot_date_str, "data/features_attributes.csv", bronze_attributes_directory, "bronze_attributes", spark
    )


def process_bronze_financials_table(snapshot_date_str, bronze_financials_directory, spark):
    return _process_bronze_table(
        snapshot_date_str, "data/features_financials.csv", bronze_financials_directory, "bronze_financials", spark
    )