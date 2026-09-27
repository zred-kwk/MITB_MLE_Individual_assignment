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

from pyspark.sql.functions import col, when, regexp_replace, regexp_extract, trim, round as spark_round
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, DateType


def process_silver_loan_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type on columns that don't need special handling
    column_type_map = {
        "Customer_ID": StringType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_Credit_Inquiries": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Annual_Income: strip stray underscores, cast to double, null out impossible values
    df = df.withColumn(
        "Annual_Income",
        spark_round(regexp_replace(col("Annual_Income"), "_", "").cast("double"), 2)
    )
    df = df.withColumn(
        "Annual_Income",
        when((col("Annual_Income") < 0) | (col("Annual_Income") > 200000), None)
        .otherwise(col("Annual_Income"))
    )

    # Monthly_Inhand_Salary: round
    df = df.withColumn("Monthly_Inhand_Salary", spark_round(col("Monthly_Inhand_Salary"), 2))

    # Num_Bank_Accounts: null out impossible values
    df = df.withColumn(
        "Num_Bank_Accounts",
        when((col("Num_Bank_Accounts") < 0) | (col("Num_Bank_Accounts") > 10), None)
        .otherwise(col("Num_Bank_Accounts"))
    )

    # Num_Credit_Card: null out impossible values
    df = df.withColumn(
        "Num_Credit_Card",
        when(col("Num_Credit_Card") > 10, None).otherwise(col("Num_Credit_Card"))
    )

    # Interest_Rate: null out impossible values
    df = df.withColumn(
        "Interest_Rate",
        when(col("Interest_Rate") > 34, None).otherwise(col("Interest_Rate"))
    )

    # Num_of_Loan: strip underscores, cast to int, null out impossible values
    df = df.withColumn(
        "Num_of_Loan",
        regexp_replace(col("Num_of_Loan"), "_", "").cast("int")
    )
    df = df.withColumn(
        "Num_of_Loan",
        when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 10), None)
        .otherwise(col("Num_of_Loan"))
    )

    # Type_of_Loan: null means no loan on file
    df = df.withColumn(
        "Type_of_Loan",
        when(col("Type_of_Loan").isNull(), "No Loan").otherwise(col("Type_of_Loan"))
    )

    # Num_of_Delayed_Payment: strip underscores, cast to int, null out impossible values
    df = df.withColumn(
        "Num_of_Delayed_Payment",
        regexp_replace(col("Num_of_Delayed_Payment"), "_", "").cast("int")
    )
    df = df.withColumn(
        "Num_of_Delayed_Payment",
        when((col("Num_of_Delayed_Payment") < 0) | (col("Num_of_Delayed_Payment") > 28), None)
        .otherwise(col("Num_of_Delayed_Payment"))
    )

    # Changed_Credit_Limit: "_" placeholder -> null, else cast and round
    df = df.withColumn(
        "Changed_Credit_Limit",
        when(col("Changed_Credit_Limit") == "_", None)
        .otherwise(spark_round(col("Changed_Credit_Limit").cast("double"), 2))
    )

    # Num_Credit_Inquiries: null out impossible values
    df = df.withColumn(
        "Num_Credit_Inquiries",
        when((col("Num_Credit_Inquiries") < 0) | (col("Num_Credit_Inquiries") > 20), None)
        .otherwise(col("Num_Credit_Inquiries"))
    )

    # Credit_Mix: "_" placeholder -> null
    df = df.withColumn(
        "Credit_Mix",
        when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix"))
    )

    # Outstanding_Debt: strip underscores, cast to double
    df = df.withColumn(
        "Outstanding_Debt",
        regexp_replace(col("Outstanding_Debt"), "_", "").cast("double")
    )

    # Credit_Utilization_Ratio: round
    df = df.withColumn("Credit_Utilization_Ratio", spark_round(col("Credit_Utilization_Ratio"), 2))

    # Credit_History_Age: "X Years and Y Months" -> numeric months
    df = df.withColumn("Credit_History_Years", regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast("int"))
    df = df.withColumn("Credit_History_Months_part", regexp_extract(col("Credit_History_Age"), r"and (\d+) Months", 1).cast("int"))
    df = df.withColumn("Credit_History_Age_Months", col("Credit_History_Years") * 12 + col("Credit_History_Months_part"))

    # Payment_of_Min_Amount: only Yes/No are valid, "NM" -> null
    df = df.withColumn(
        "Payment_of_Min_Amount",
        when(col("Payment_of_Min_Amount").isin("Yes", "No"), col("Payment_of_Min_Amount"))
        .otherwise(None)
    )

    # Total_EMI_per_month: null out anything outside a sane range, round
    df = df.withColumn(
        "Total_EMI_per_month",
        spark_round(when(col("Total_EMI_per_month").between(0, 2000), col("Total_EMI_per_month")).otherwise(None), 2)
    )

    # Amount_invested_monthly: "__10000__" sentinel -> null, else strip underscores, cast, round
    df = df.withColumn(
        "Amount_invested_monthly",
        spark_round(
            when(trim(col("Amount_invested_monthly")) == "__10000__", None)
            .otherwise(regexp_replace(trim(col("Amount_invested_monthly")), r"^_+|_+$", "").cast("double")),
            2
        )
    )

    # Payment_Behaviour: "!@9#%8" placeholder -> null
    df = df.withColumn("Payment_Behaviour", trim(col("Payment_Behaviour")))
    df = df.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour"))
    )

    # Monthly_Balance: sentinel -> null, else strip underscores, cast, round
    df = df.withColumn(
        "Monthly_Balance",
        spark_round(
            when(trim(col("Monthly_Balance")) == "__-333333333333333333333333333__", None)
            .otherwise(regexp_replace(trim(col("Monthly_Balance")), r"^_+|_+$", "").cast("double")),
            2
        )
    )

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": StringType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Age: strip out any non-digit characters (drops minus sign, trailing underscores, etc.),
    # then take the first 2 digits of what remains
    df = df.withColumn(
        "Age",
        regexp_extract(regexp_replace(col("Age"), "[^0-9]", ""), "^(\\d{1,2})", 1).cast("int")
    )
    # null out anything still implausible
    df = df.withColumn(
        "Age",
        when((col("Age") > 100), None).otherwise(col("Age"))
    )

    # Occupation: "_______" placeholder -> null
    df = df.withColumn(
        "Occupation",
        when(col("Occupation") == "_______", None).otherwise(col("Occupation"))
    )

    # SSN: keep only values matching the valid ###-##-#### format, else null
    df = df.withColumn(
        "SSN",
        when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")).otherwise(None)
    )

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # fe_1 .. fe_20 came back with zero nulls and clean numeric values during EDA,
    # so no value-level cleaning is needed here beyond a consistent schema
    column_type_map = {f"fe_{i}": IntegerType() for i in range(1, 21)}
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df