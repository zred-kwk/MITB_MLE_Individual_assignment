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



def _load_clickstream_preapplication(directory, prefix, cutoff_date, spark):
    """
    Aggregate every clickstream partition dated ON OR BEFORE cutoff_date
    (the loan's origination date) into one row per Customer_ID: mean of
    each fe_ column, plus how many months of history were available.

    Anchoring to <= origination date (rather than <= the mob==6 monitoring
    date) keeps this a clean
    "pre-application engagement" feature set, and avoids any leakage from the outcome window.
    """
    pattern = os.path.join(directory, f"{prefix}_*.parquet")
    files = glob.glob(pattern)

    valid_files = []
    for f in files:
        date_part = os.path.basename(f).replace(prefix + "_", "").replace(".parquet", "")
        try:
            file_date = datetime.strptime(date_part, "%Y_%m_%d")
        except ValueError:
            continue
        if file_date <= cutoff_date:
            valid_files.append(f)

    if not valid_files:
        return None

    df = spark.read.parquet(*valid_files)
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    if not fe_cols:
        return None

    agg_exprs = [F.round(F.avg(c), 2).alias(c) for c in fe_cols]
    agg_exprs.append(F.count(F.lit(1)).alias("Clickstream_Months_Observed"))

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
    fin_df = spark.read.parquet(fin_filepath).drop("snapshot_date") if os.path.exists(fin_filepath) else None

    attr_partition = "silver_attributes_" + origination_snapshot_date_str.replace('-', '_') + '.parquet'
    attr_filepath = silver_attributes_directory + attr_partition
    attr_df = spark.read.parquet(attr_filepath).drop("snapshot_date") if os.path.exists(attr_filepath) else None

    # -----------------------------------------------------------------
    # 3. Clickstream: pre-application engagement only (<= origination date)
    # -----------------------------------------------------------------
    click_df = _load_clickstream_preapplication(
        silver_clickstream_directory, "silver_clickstream", origination_date, spark
    )

    df = base
    if fin_df is not None:
        pre_count = df.count()
        df = df.join(fin_df, on="Customer_ID", how="left")
        assert df.count() == pre_count, "fan-out detected in financials join (Customer_ID not unique in fin_df)"
    if attr_df is not None:
        pre_count = df.count()
        df = df.join(attr_df, on="Customer_ID", how="left")
        assert df.count() == pre_count, "fan-out detected in attributes join (Customer_ID not unique in attr_df)"
    if click_df is not None:
        pre_count = df.count()
        df = df.join(click_df, on="Customer_ID", how="left")
        assert df.count() == pre_count, "fan-out detected in clickstream join (Customer_ID not unique in click_df)"

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

    df = df.drop("Credit_History_Age", "Credit_History_Years", "Credit_History_Months_part","Name","SSN")

    # -----------------------------------------------------------------
    # 5. Save gold table
    # -----------------------------------------------------------------
    partition_name = "gold_feature_store_" + origination_snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df