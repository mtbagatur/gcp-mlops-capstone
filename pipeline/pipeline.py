"""Training pipeline: extract → validate (+ drift check) → train → evaluate → (gate) register + deploy."""
from kfp import dsl

from pipeline.components import (
    check_drift,
    evaluate_model,
    extract_data,
    register_and_deploy,
    train_model,
    validate_data,
)


@dsl.pipeline(name="income-training", description="Nightly retraining with a quality gate")
def income_pipeline(
    project: str,
    region: str = "us-central1",
    table: str = "mtb-mlops-capstone.mlops.income_training",
    bucket: str = "mtb-mlops-capstone-mlops",
    service: str = "income-api",
    min_rows: int = 20000,
    min_auc: float = 0.90,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    drift_days: int = 7,
    drift_min_requests: int = 50,
    drift_threshold: float = 0.2,
    drift_include_synthetic: bool = False,
):
    data = extract_data(project=project, table=table)
    checked = validate_data(data=data.outputs["data"], min_rows=min_rows)
    # Runs alongside training; a WARNING log line triggers the email alert policy.
    check_drift(
        project=project,
        data=data.outputs["data"],
        days=drift_days,
        min_requests=drift_min_requests,
        threshold=drift_threshold,
        include_synthetic=drift_include_synthetic,
    )

    trained = train_model(data=data.outputs["data"], learning_rate=learning_rate, max_iter=max_iter)
    trained.after(checked)

    evaluated = evaluate_model(data=data.outputs["data"], model=trained.outputs["model"])

    with dsl.If(evaluated.outputs["Output"] >= min_auc, name="quality-gate"):
        register_and_deploy(
            project=project,
            region=region,
            bucket=bucket,
            service=service,
            run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            roc_auc=evaluated.outputs["Output"],
            model=trained.outputs["model"],
        )
