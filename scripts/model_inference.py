import os
import glob
import pandas as pd
import pickle
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # connect to feature store
    folder_path = "datamart/gold/feature_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    features_store_sdf = spark.read.option("header", "true").parquet(*files_list)

    # --- load features for loans originated on the snapshot date (no label yet) ---
    features_sdf = features_store_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    features_pdf = features_sdf.toPandas()
    if features_pdf.empty:
        raise ValueError(f"No feature rows for snapshot {config['snapshot_date_str']}")

    # --- preprocess with the transformer fitted in training ---
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    feature_cols = list(transformer_stdscaler.feature_names_in_)      # exact training columns, in training order

    X_inference = transformer_stdscaler.transform(features_pdf[feature_cols])
    print("X_inference", X_inference.shape)

    # --- predict ---
    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference)[:, 1]

    y_inference_pdf = features_pdf[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference

    gold_directory = f"datamart/gold/model_predictions/{config["model_name"][:-4]}/"
    print(gold_directory)

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # save gold table - IRL connect to database to write
    partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

        # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)