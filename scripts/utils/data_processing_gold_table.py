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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    n = df.count()
    if n == 0:
        print(f"WARNING: no loans at mob={mob} on {snapshot_date_str} "
              f"(expected only in the first {mob} months of loan data)")

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


import os
import glob
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType



from dateutil.relativedelta import relativedelta

def _load_clickstream_preapplication(directory, prefix, cutoff_date, spark, window_months=6):
    """
    Aggregate clickstream partitions in the trailing window
    (cutoff_date - window_months, cutoff_date] into one row per Customer_ID:
    mean of each fe_ column, plus how many months of history were available.

    The cutoff month's own partition MUST exist and be non-empty; otherwise
    this raises rather than silently reusing older data.
    """
    window_start = cutoff_date - relativedelta(months=window_months)

    parsed = []
    for f in glob.glob(os.path.join(directory, f"{prefix}_*.parquet")):
        date_part = os.path.basename(f).replace(prefix + "_", "").replace(".parquet", "")
        try:
            file_date = datetime.strptime(date_part, "%Y_%m_%d")
        except ValueError:
            continue
        if file_date <= cutoff_date:
            parsed.append((file_date, f))

    if not parsed:
        raise ValueError(f"No clickstream partitions on or before {cutoff_date:%Y-%m-%d}")

    latest_date, latest_file = max(parsed)
    if latest_date < cutoff_date:
        raise ValueError(
            f"Clickstream stale for {cutoff_date:%Y-%m-%d}: latest partition is {latest_date:%Y-%m-%d}"
        )
    if spark.read.parquet(latest_file).limit(1).count() == 0:
        raise ValueError(f"Clickstream partition for {cutoff_date:%Y-%m-%d} is empty")

    valid_files = [f for d, f in parsed if d > window_start]

    # The window must be complete: a missing middle month would otherwise be averaged over silently.
    first_date = min(d for d, _ in parsed)
    delta = relativedelta(cutoff_date, first_date)
    months_since_first = delta.years * 12 + delta.months
    expected = min(window_months, months_since_first + 1)
    if len(valid_files) != expected:
        raise ValueError(
            f"Clickstream window incomplete for {cutoff_date:%Y-%m-%d}: "
            f"{len(valid_files)}/{expected} partitions"
        )

    df = spark.read.parquet(*valid_files)
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    if not fe_cols:
        raise ValueError(f"No fe_ columns in clickstream partitions for {cutoff_date:%Y-%m-%d}")

    agg_exprs = [F.round(F.avg(c), 2).alias(c) for c in fe_cols]

    return df.groupBy("Customer_ID").agg(*agg_exprs)


def process_gold_feature_table(
    origination_snapshot_date_str,
    silver_loan_daily_directory,
    silver_financials_directory,
    silver_attributes_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark,
):
    origination_date = datetime.strptime(origination_snapshot_date_str, "%Y-%m-%d")

    # -----------------------------------------------------------------
    # 1. Base population = loans that ORIGINATED this month, i.e. mob == 0
    #    in this month's loan_daily monitoring partition. 
    # -----------------------------------------------------------------
    loan_partition = "silver_loan_daily_" + origination_snapshot_date_str.replace('-', '_') + '.parquet'
    loan_filepath = silver_loan_daily_directory + loan_partition
    loan_df = spark.read.parquet(loan_filepath)
    print('loaded from:', loan_filepath, 'row count:', loan_df.count())

    base = (
        loan_df.filter(col("mob") == 0)
        .select(
            col("loan_id"),
            col("Customer_ID"),
            col("loan_start_date"),
            col("tenure"),
            col("loan_amt"),
            col("due_amt").alias("Scheduled_Installment_Amt"),
            col("snapshot_date"),  # == origination_date for this filtered set
        )
    )
    print('loans originating this month:', base.count())

    # -----------------------------------------------------------------
    # 2. Financials / attributes
    # -----------------------------------------------------------------
    fin_partition = "silver_financials_" + origination_snapshot_date_str.replace('-', '_') + '.parquet'
    fin_filepath = silver_financials_directory + fin_partition
    if not os.path.exists(fin_filepath):
        raise FileNotFoundError(f"Missing financials partition: {fin_filepath}")
    fin_df = spark.read.parquet(fin_filepath).drop("snapshot_date")

    attr_partition = "silver_attributes_" + origination_snapshot_date_str.replace('-', '_') + '.parquet'
    attr_filepath = silver_attributes_directory + attr_partition
    if not os.path.exists(attr_filepath):
        raise FileNotFoundError(f"Missing attributes partition: {attr_filepath}")
    attr_df = spark.read.parquet(attr_filepath).drop("snapshot_date")

    # -----------------------------------------------------------------
    # 3. Clickstream: pre-application engagement only (<= origination date)
    # -----------------------------------------------------------------
    click_df = _load_clickstream_preapplication(
        silver_clickstream_directory, "silver_clickstream", origination_date, spark
    )

    df = base
    for name, right in [("financials", fin_df), ("attributes", attr_df), ("clickstream", click_df)]:
        pre_count = df.count()
        df = df.join(right, on="Customer_ID", how="left")
        assert df.count() == pre_count, f"fan-out detected in {name} join (Customer_ID not unique)"

    # -----------------------------------------------------------------
    # 4. Derived features -- all sourced from origination-time financials/
    #    attributes/loan-terms (no leakage from the monitoring window)
    # -----------------------------------------------------------------
    df = df.withColumn(
        "EMI_to_Income_Ratio",
        F.round(when(col("Monthly_Inhand_Salary") > 0,
                      col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")), 4)
    )
    df = df.withColumn(
        "Debt_to_Income_Ratio",
        F.round(when(col("Annual_Income") > 0,
                      col("Outstanding_Debt") / col("Annual_Income")), 4)
    )
    df = df.withColumn(
        "Investment_to_Income_Ratio",
        F.round(when(col("Monthly_Inhand_Salary") > 0,
                      col("Amount_invested_monthly") / col("Monthly_Inhand_Salary")), 4)
    )
    df = df.withColumn(
        "Balance_to_Income_Ratio",
        F.round(when(col("Monthly_Inhand_Salary") > 0,
                      col("Monthly_Balance") / col("Monthly_Inhand_Salary")), 4)
    )
    df = df.withColumn(
        "Delayed_Payment_Rate",
        F.round(when(col("Num_of_Loan") > 0,
                      col("Num_of_Delayed_Payment") / col("Num_of_Loan")), 4)
    )
    df = df.withColumn(
        "Post_EMI_Slack",
        F.round(col("Monthly_Inhand_Salary") - col("Total_EMI_per_month") - col("Scheduled_Installment_Amt"), 2)
    )

    df = df.withColumn(
        "Loan_Type_Array",
        when(col("Type_of_Loan") == "No Loan", F.array())
        .otherwise(F.expr(
            "filter(transform(split(regexp_replace(Type_of_Loan, ' and ', ''), ','), x -> trim(x)), x -> x != '')"
        ))
    )
    df = df.withColumn("Loan_Type_Diversity", F.size(F.array_distinct(col("Loan_Type_Array")))).drop("Loan_Type_Array")

    df = df.withColumn(
        "Credit_Utilization_Band",
        when(col("Credit_Utilization_Ratio") < 30, "Low")
        .when(col("Credit_Utilization_Ratio") < 40, "Medium")
        .when(col("Credit_Utilization_Ratio").isNotNull(), "High")
    )
    df = df.withColumn(
        "Credit_History_Band",
        when(col("Credit_History_Age_Months") < 60, "New")
        .when(col("Credit_History_Age_Months") < 180, "Established")
        .when(col("Credit_History_Age_Months").isNotNull(), "Long_Tenure")
    )
    df = df.withColumn(
        "Recent_Credit_Expansion_Flag",
        when(col("Changed_Credit_Limit") > 0, 1).otherwise(0)
    )

    df = df.withColumn(
        "Age_Band",
        when(col("Age") < 25, "Under25")
        .when(col("Age") < 40, "25to39")
        .when(col("Age") < 60, "40to59")
        .when(col("Age").isNotNull(), "60plus")
    )
    df = df.withColumn(
        "Occupation_Group",
        when(col("Occupation").isin("Accountant", "Architect", "Doctor", "Engineer", "Lawyer", "Scientist", "Developer"), "Professional")
        .when(col("Occupation").isin("Journalist", "Musician", "Writer", "Media_Manager"), "Creative")
        .when(col("Occupation").isin("Manager", "Entrepreneur"), "Business")
        .when(col("Occupation") == "Mechanic", "Trade")
        .when(col("Occupation") == "Teacher", "Education")
        .otherwise("Unknown")
    )

    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    if fe_cols:
        df = df.withColumn("Clickstream_Mean", F.round(sum(col(c) for c in fe_cols) / len(fe_cols), 2))
        df = df.withColumn(
            "Clickstream_Vector_Norm",
            F.round(F.sqrt(sum(col(c) * col(c) for c in fe_cols)), 2)
        ) # Euclidean norm of the clickstream feature vector

    df = df.drop("Credit_History_Age", "Credit_History_Years", "Credit_History_Months_part","Name","SSN", "Post_EMI_Slack", "Monthly_Inhand_Salary","Loan_Type_Diversity")

    # -----------------------------------------------------------------
    # 5. Save gold table
    # -----------------------------------------------------------------
    partition_name = "gold_feature_store_" + origination_snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df