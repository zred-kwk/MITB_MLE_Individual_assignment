from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = "/opt/airflow"

# First snapshot date with a complete training window.
FIRST_MODEL_DATE = "2024-09-01"


def script_cmd(script, args=""):
    """
    Execute a pipeline script from the project root.

    Airflow's {{ ds }} is the logical date in YYYY-MM-DD format.
    """
    return (
        f"cd {PROJECT_DIR} && "
        f"python3 scripts/{script} "
        f'--snapshotdate "{{{{ ds }}}}" '
        f"{args}"
    )


def stage_cmd(stage):
    """Execute a pipeline stage."""
    return script_cmd(
        "pipeline_stage.py",
        f"--stage {stage}",
    )


def is_model_date(ds, **context):
    """Only train models from the first eligible snapshot date."""
    return ds >= FIRST_MODEL_DATE


# ============================================================
# DEFAULT TASK SETTINGS
# ============================================================

default_args = {
    "owner": "airflow",
    "depends_on_past": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ============================================================
# DAG DEFINITION
# ============================================================

with DAG(
    dag_id="credit_risk_pipeline",
    description="Monthly credit-risk data and ML pipeline",
    schedule="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    default_args=default_args,
    tags=["credit-risk", "ml-pipeline"],
) as dag:

    # ========================================================
    # 1. LABEL STORE
    # ========================================================

    bronze_lms = BashOperator(
        task_id="bronze_lms",
        bash_command=script_cmd("bronze_label_store.py"),
    )

    silver_loan = BashOperator(
        task_id="silver_loan",
        bash_command=stage_cmd("silver_loan"),
    )

    gold_label = BashOperator(
        task_id="gold_label",
        bash_command=stage_cmd("gold_label"),
    )

    label_store_completed = EmptyOperator(
        task_id="label_store_completed",
    )

    bronze_lms >> silver_loan >> gold_label >> label_store_completed


    # ========================================================
    # 2. FEATURE STORE
    # ========================================================

    # Bronze layer: run independent ingestion tasks in parallel.

    bronze_clickstream = BashOperator(
        task_id="bronze_clickstream",
        bash_command=stage_cmd("bronze_clickstream"),
    )

    bronze_attributes = BashOperator(
        task_id="bronze_attributes",
        bash_command=stage_cmd("bronze_attributes"),
    )

    bronze_financials = BashOperator(
        task_id="bronze_financials",
        bash_command=stage_cmd("bronze_financials"),
    )

    # Silver layer: transform each source independently.

    silver_clickstream = BashOperator(
        task_id="silver_clickstream",
        bash_command=stage_cmd("silver_clickstream"),
    )

    silver_attributes = BashOperator(
        task_id="silver_attributes",
        bash_command=stage_cmd("silver_attributes"),
    )

    silver_financials = BashOperator(
        task_id="silver_financials",
        bash_command=stage_cmd("silver_financials"),
    )

    # Gold layer: combine the loan population and silver features.

    gold_feature = BashOperator(
        task_id="gold_feature",
        bash_command=stage_cmd("gold_feature"),
    )

    feature_store_completed = EmptyOperator(
        task_id="feature_store_completed",
    )

    # Bronze -> Silver dependencies.

    bronze_clickstream >> silver_clickstream
    bronze_attributes >> silver_attributes
    bronze_financials >> silver_financials

    # Gold features require all silver inputs, including silver loans.

    [
        silver_loan,
        silver_clickstream,
        silver_attributes,
        silver_financials,
    ] >> gold_feature

    gold_feature >> feature_store_completed


    # ========================================================
    # 3. MODEL TRAINING
    # ========================================================

    # IMPORTANT:
    # These tasks must not depend on the previous month's
    # model-task state. Earlier dates intentionally skip training.

    model_train_gate = ShortCircuitOperator(
        task_id="model_train_gate",
        python_callable=is_model_date,
        depends_on_past=False,
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command=script_cmd("model_train.py"),
        depends_on_past=False,
    )

    # Both stores must be complete before checking the date
    # and starting model training.

    [
        label_store_completed,
        feature_store_completed,
    ] >> model_train_gate >> model_train


    # ========================================================
    # 4. MODEL INFERENCE
    # ========================================================

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=script_cmd(
            "model_inference.py",
            '--modelname "credit_model_{{ ds | replace("-", "_") }}.pkl"',
        ),
        depends_on_past=False,
    )

    model_train >> model_inference


    # ========================================================
    # 5. MODEL MONITORING
    # ========================================================

    # Placeholder until a monitoring script is implemented.

    model_monitor = EmptyOperator(
        task_id="model_monitor",
        depends_on_past=False,
    )

    model_inference >> model_monitor
