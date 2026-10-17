"""Vertex AI pipeline components. All run on the trainer image, so they import the
same training code that the unit tests cover."""
import os

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

PROJECT = os.getenv("PROJECT_ID", "mtb-mlops-capstone")
REGION = os.getenv("REGION", "us-central1")
TRAINER_IMAGE = os.getenv(
    "TRAINER_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/mlops/income-trainer:latest"
)

component = dsl.component(base_image=TRAINER_IMAGE, install_kfp_package=False)


@component
def extract_data(project: str, table: str, data: Output[Dataset]):
    from google.cloud import bigquery

    df = bigquery.Client(project=project).query(f"SELECT * FROM `{table}`").to_dataframe()
    df.to_parquet(data.path, index=False)
    data.metadata.update({"source": f"bq://{table}", "rows": len(df)})


@component
def validate_data(data: Input[Dataset], min_rows: int, stats: Output[Metrics]):
    import pandas as pd

    from training.validate import validate

    for k, v in validate(pd.read_parquet(data.path), min_rows=min_rows).items():
        stats.log_metric(k, v)


@component
def train_model(data: Input[Dataset], learning_rate: float, max_iter: int, model: Output[Model]):
    import joblib
    import pandas as pd

    from training.features import TARGET, prepare
    from training.train import build_model

    tr = pd.read_parquet(data.path).query("split == 'TRAIN'")
    m = build_model(learning_rate, max_iter).fit(prepare(tr), tr[TARGET].astype(int))
    joblib.dump(m, model.path)
    model.metadata.update({"framework": "scikit-learn", "learning_rate": learning_rate, "max_iter": max_iter})


@component
def evaluate_model(data: Input[Dataset], model: Input[Model], metrics: Output[Metrics]) -> float:
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from training.features import TARGET, prepare

    te = pd.read_parquet(data.path).query("split == 'TEST'")
    proba = joblib.load(model.path).predict_proba(prepare(te))[:, 1]
    auc = round(float(roc_auc_score(te[TARGET].astype(int), proba)), 4)
    metrics.log_metric("roc_auc", auc)
    metrics.log_metric("n_test", len(te))
    return auc


@component
def register_and_deploy(
    project: str,
    region: str,
    bucket: str,
    service: str,
    run_name: str,
    roc_auc: float,
    model: Input[Model],
) -> str:
    """Runs only after the quality gate passes: version the model, register it and
    roll Cloud Run to a new revision that loads it."""
    import re

    from google.cloud import aiplatform, run_v2, storage

    version = re.sub(r"[^a-z0-9-]", "-", run_name.lower())[-40:].strip("-")
    gcs = storage.Client(project=project).bucket(bucket)
    for path in (f"models/{version}/model.joblib", "models/current/model.joblib"):
        gcs.blob(path).upload_from_filename(model.path)

    # Model Registry: one parent model, a new version per approved run (lineage, G9).
    aiplatform.init(project=project, location=region)
    existing = aiplatform.Model.list(filter='display_name="income-model"')
    aiplatform.Model.upload(
        display_name="income-model",
        parent_model=existing[0].resource_name if existing else None,
        artifact_uri=f"gs://{bucket}/models/{version}",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest",
        labels={"roc_auc": str(roc_auc).replace(".", "_"), "pipeline_run": version[:63]},
        is_default_version=True,
    )

    # New Cloud Run revision: changing MODEL_VERSION forces a fresh model load.
    runc = run_v2.ServicesClient()
    svc = runc.get_service(name=f"projects/{project}/locations/{region}/services/{service}")
    env = [e for e in svc.template.containers[0].env if e.name != "MODEL_VERSION"]
    env.append(run_v2.EnvVar(name="MODEL_VERSION", value=version))
    del svc.template.containers[0].env[:]
    svc.template.containers[0].env.extend(env)
    runc.update_service(service=svc).result()
    return version
