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

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

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

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)


# ---------------------------------------------------------------------------
# Bronze layer
# ---------------------------------------------------------------------------

# --- lms (loan) ---
bronze_lms_directory = "datamart/bronze/lms/"
if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_lms_table(date_str, bronze_lms_directory, spark)

# --- clickstream ---
bronze_clickstream_directory = "datamart/bronze/feature_clickstream/"
if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_clickstream_table(date_str, bronze_clickstream_directory, spark)

# --- attributes ---
bronze_attributes_directory = "datamart/bronze/feature_attributes/"
if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_attributes_table(date_str, bronze_attributes_directory, spark)

# --- financials ---
bronze_financials_directory = "datamart/bronze/feature_financials/"
if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_financials_table(date_str, bronze_financials_directory, spark)


# ---------------------------------------------------------------------------
# Silver layer
# ---------------------------------------------------------------------------

# --- loan ---
silver_loan_daily_directory = "datamart/silver/loan_daily/"
if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# --- financials ---
silver_financials_directory = "datamart/silver/financials/"
if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_financials_directory, silver_financials_directory, spark)

# --- attributes ---
silver_attributes_directory = "datamart/silver/attributes/"
if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)

# --- clickstream ---
silver_clickstream_directory = "datamart/silver/clickstream/"
if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)


# ---------------------------------------------------------------------------
# Gold layer: labels
# ---------------------------------------------------------------------------

gold_label_store_directory = "datamart/gold/label_store/"
if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6)

folder_path = gold_label_store_directory
files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
label_df = spark.read.option("header", "true").parquet(*files_list)
print("label row_count:", label_df.count())
label_df.show()


# ---------------------------------------------------------------------------
# Gold layer: feature store
# ---------------------------------------------------------------------------

gold_feature_store_directory = "datamart/gold/feature_store/"
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_gold_feature_table(
        date_str,
        silver_loan_daily_directory,
        silver_financials_directory,
        silver_attributes_directory,
        silver_clickstream_directory,
        gold_feature_store_directory,
        spark,
    )

folder_path = gold_feature_store_directory
files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
feature_df = spark.read.option("header", "true").parquet(*files_list)
print("feature row_count:", feature_df.count())
feature_df.show()