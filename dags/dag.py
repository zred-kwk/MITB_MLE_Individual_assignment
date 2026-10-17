from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta

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

# Last month every feature source covers (clickstream etc. stop here).
FEATURE_END_DATE = "2024-12-01"

# Months on book used for the label window.
MOB = 6


def _plus_months(ds, n):
    return (datetime.strptime(ds, "%Y-%m-%d") + relativedelta(months=n)).strftime(
        "%Y-%m-%d"
    )


# Training at T reads features through T-7 months, so
# T <= FEATURE_END_DATE + MOB + 1 months  ->  2025-07-01.
LAST_MODEL_DATE = _plus_months(FEATURE_END_DATE, MOB + 1)


def script_cmd(script, args=""):
    """
    Execute a pipeline script from the project root.

    Airflow's {{ ds }} is the logical date in YYYY-MM-DD format.
    PROJECT_DIR is exported because model_monitor.py resolves its paths
    from that variable, while the other scripts use paths relative to the
    working directory.
    """
    return (
        f"cd {PROJECT_DIR} && "
        f"PROJECT_DIR={PROJECT_DIR} python3 scripts/{script} "
        f'--snapshotdate "{{{{ ds }}}}" '
        f"{args}"
    )


def stage_cmd(stage):
    """Execute a pipeline stage."""
    return script_cmd(
        "pipeline_stage.py",
        f"--stage {stage}",
    )


def is_feature_date(ds, **context):
    """Feature sources only exist up to FEATURE_END_DATE."""
    return ds <= FEATURE_END_DATE


def is_model_date(ds, **context):
    """Only train within the window where both labels and features exist."""
    return FIRST_MODEL_DATE <= ds <= LAST_MODEL_DATE


def is_scoring_date(ds, **context):
    """
    Only score and monitor once a model trained at an EARLIER snapshot
    exists. Inference and monitoring use models with train date < snapshot,
    so the model trained in this same run is never picked up.
    """
    return ds > FIRST_MODEL_DATE


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
    end_date=datetime(2025, 11, 1),  # label range; features stop earlier via feature_gate
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

    # Past FEATURE_END_DATE this gate skips the whole feature chain (and
    # everything downstream of it: inference, own-month scoring, monitoring)
    # instead of letting bronze_clickstream fail. ShortCircuitOperator skips
    # all downstream tasks regardless of trigger rules by default, and
    # depends_on_past accepts a skipped previous instance, so the chain
    # resumes cleanly.

    feature_gate = ShortCircuitOperator(
        task_id="feature_gate",
        python_callable=is_feature_date,
        ignore_downstream_trigger_rules=False,
    )

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

    feature_gate >> [bronze_clickstream, bronze_attributes, bronze_financials]

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
    # 3. MODEL TRAINING (parallel stream A)
    # ========================================================

    # IMPORTANT:
    # These tasks must not depend on the previous month's
    # model-task state. Dates outside the model window intentionally
    # skip training.

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

    # Training at T reads labels through T-1 month and features through
    # T-7 months. It never reads T's own features, so it waits only for the
    # label store, not feature_store_completed. (model_train.py reads
    # explicit partitions, so it is safe to run alongside gold_feature.)

    label_store_completed >> model_train_gate >> model_train

    # The model trained in this run also scores its own snapshot (model
    # credit_model_2024_09_01 -> predictions for 2024_09_01). This needs
    # both the trained model and this month's features. Past FEATURE_END_DATE
    # feature_store_completed is skipped, so this task is skipped too.

    model_inference_own_month = BashOperator(
        task_id="model_inference_own_month",
        bash_command=script_cmd(
            "model_inference.py",
            "--modelname credit_model_{{ ds.replace('-', '_') }}.pkl",
        ),
        depends_on_past=False,
    )

    [model_train, feature_store_completed] >> model_inference_own_month


    # ========================================================
    # 4. MODEL INFERENCE (parallel stream B)
    # ========================================================

    # Inference only needs the gold features for this snapshot. It scores
    # with models trained at EARLIER snapshots (train date < snapshot), so
    # it never touches the artefact that model_train is writing in this
    # run, and it can run in parallel with training.
    #
    # On the first model date no earlier model exists, and model_inference.py
    # exits non-zero with "no eligible models". The gate therefore skips only
    # this task (ignore_downstream_trigger_rules=False), so monitoring still
    # runs on that date on the strength of model_inference_own_month. Dates
    # before training starts are skipped by model_train_gate, which skips
    # everything downstream of model_train, including monitoring.

    model_inference_gate = ShortCircuitOperator(
        task_id="model_inference_gate",
        python_callable=is_scoring_date,
        ignore_downstream_trigger_rules=False,
        depends_on_past=False,
    )

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=script_cmd("model_inference.py", "--all-models"),
        depends_on_past=False,
    )

    feature_store_completed >> model_inference_gate >> model_inference


    # ========================================================
    # 5. MODEL MONITORING (joins both streams)
    # ========================================================

    # Monitoring runs last, once every prediction for this snapshot exists:
    # model_inference (models trained earlier) and model_inference_own_month
    # (the model trained in this run), plus the labels. On the first model
    # date model_inference is skipped by its gate, so the join accepts a
    # skipped parent as long as one parent succeeded and none failed.
    # Dates before training starts are skipped by model_train_gate, and dates
    # past FEATURE_END_DATE by feature_gate; both skip everything downstream,
    # including this join.
    #
    # The fixed --inference-id makes a retry overwrite its own metrics
    # files instead of adding duplicate rows for the same snapshot.

    model_monitor_start = EmptyOperator(
        task_id="model_monitor_start",
        trigger_rule="none_failed_min_one_success",
    )

    model_monitor = BashOperator(
        task_id="model_monitor",
        bash_command=script_cmd(
            "model_monitor.py",
            '--all-models --deployed-only --inference-id "{{ ts_nodash }}"',
        ),
        depends_on_past=False,
    )

    model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")

    [
        model_inference,
        model_inference_own_month,
        label_store_completed,
    ] >> model_monitor_start >> model_monitor >> model_monitor_completed