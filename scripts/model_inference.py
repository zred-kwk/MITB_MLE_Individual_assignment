import os
import re
import glob
import pickle
import pprint
import argparse
from datetime import datetime

import pyspark
from pyspark.sql.functions import col


def eligible_models(snapshotdate):
    """
    Artefacts in model_bank trained strictly BEFORE the snapshot.

    A model trained on the snapshot date itself is excluded: training runs in
    parallel with inference, so that artefact may not exist yet or may still
    be mid-write. The DAG scores it separately with --modelname once training
    has finished.
    """
    snapshot = datetime.strptime(snapshotdate, "%Y-%m-%d")
    names = []
    for path in sorted(glob.glob("model_bank/*.pkl")):
        name = os.path.basename(path)
        m = re.search(r"(\d{4})_(\d{2})_(\d{2})", name)
        if m and datetime(*map(int, m.groups())) < snapshot:
            names.append(name)
    return names


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
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)

    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # connect to feature store
    folder_path = "datamart/gold/feature_store/"
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    features_store_sdf = spark.read.option("header", "true").parquet(*files_list)

    # --- load features for loans originated on the snapshot date (no label yet) ---
    features_sdf = features_store_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    features_pdf = features_sdf.toPandas()
    if features_pdf.empty:
        raise ValueError(f"No feature rows for snapshot {config['snapshot_date_str']}")

    # --- preprocess with the transformer fitted in training ---
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    feature_cols = list(transformer_stdscaler.feature_names_in_)  # exact training columns, in training order

    X_inference = transformer_stdscaler.transform(features_pdf[feature_cols])
    print("X_inference", X_inference.shape)

    # --- predict ---
    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference)[:, 1]

    y_inference_pdf = features_pdf[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference

    gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    print(gold_directory)

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # save gold table - IRL connect to database to write
    partition_name = (
        config["model_name"][:-4]
        + "_predictions_"
        + config["snapshot_date_str"].replace('-', '_')
        + '.parquet'
    )
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    # --- end spark session ---
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, default=None, help="model_name")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Score with every model trained before the snapshot.",
    )
    args = parser.parse_args()

    if bool(args.modelname) == bool(args.all_models):
        parser.error("Provide exactly one of --modelname or --all-models.")

    if not args.all_models:
        main(args.snapshotdate, args.modelname)
    else:
        models = eligible_models(args.snapshotdate)
        if not models:
            raise SystemExit(
                f"No models in model_bank trained before {args.snapshotdate}."
            )
        print(f"Scoring with {len(models)} model(s): {models}")

        failures = []
        for name in models:
            print(f"\n===== {name} @ {args.snapshotdate} =====")
            try:
                main(args.snapshotdate, name)
            except Exception as exc:  # one bad model must not block the others
                failures.append((name, repr(exc)))
                print(f"FAILED {name}: {exc!r}")

        if failures:
            print("\nFailures:")
            for name, err in failures:
                print(f"  {name}: {err[:200]}")
            raise SystemExit(1)