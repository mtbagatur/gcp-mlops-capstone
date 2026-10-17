"""Prediction API, deployed to Cloud Run (scales to zero, requirement G5)."""
import json
import os
import pathlib
import tempfile
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app import narrative
from app.explain import Explainer
from training.features import prepare

MODEL_URI = os.getenv("MODEL_URI", "artifacts/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")  # set by the training pipeline
APP_VERSION = os.getenv("APP_VERSION", "local")  # set by CI (commit SHA)

app = FastAPI(title="income-mlops", version="0.1.0")


class Person(BaseModel):
    age: int = Field(ge=16, le=100)
    workclass: str
    education: str
    education_num: int = Field(ge=1, le=16)
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(ge=0)
    capital_loss: int = Field(ge=0)
    hours_per_week: int = Field(ge=1, le=99)
    native_country: str


class Prediction(BaseModel):
    high_income_probability: float
    high_income: bool
    model_version: str


class Factor(BaseModel):
    feature: str
    value: str | int | float
    contribution: float
    direction: str


class ExplainedPrediction(Prediction):
    confidence: str
    base_log_odds: float
    top_factors: list[Factor]
    contributions: dict[str, float]
    narrative: dict | None
    narrative_error: str | None = None


def _download(uri: str) -> str:
    from google.cloud import storage

    bucket, _, blob = uri.removeprefix("gs://").partition("/")
    dst = pathlib.Path(tempfile.gettempdir()) / "model.joblib"
    storage.Client().bucket(bucket).blob(blob).download_to_filename(dst)
    return str(dst)


@lru_cache(maxsize=1)
def get_model():
    path = _download(MODEL_URI) if MODEL_URI.startswith("gs://") else MODEL_URI
    return joblib.load(path)


# Not /healthz: Cloud Run reserves paths ending in "z" and answers them itself.
def log_prediction(person: "Person", p: float, endpoint: str, source: str = "live") -> None:
    """One structured line per prediction. Cloud Run turns JSON stdout into jsonPayload,
    and a log sink copies these rows to BigQuery for the drift check (G8)."""
    record = {"event": "prediction", "severity": "INFO", "endpoint": endpoint, "source": source,
              "model_version": MODEL_VERSION, "probability": round(p, 4), **person.model_dump()}
    print(json.dumps(record), flush=True)


@lru_cache(maxsize=1)
def get_explainer():
    return Explainer(get_model())


def confidence(p: float) -> str:
    """Deterministic, not left to the LLM."""
    distance = abs(p - 0.5)
    return "high" if distance >= 0.35 else "medium" if distance >= 0.15 else "low"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "app_version": APP_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(person: Person, x_traffic_source: str = Header("live")):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(503, "model not available")
    row = prepare(pd.DataFrame([person.model_dump()]))
    p = float(model.predict_proba(row)[0, 1])
    log_prediction(person, p, "predict", x_traffic_source)
    return Prediction(high_income_probability=round(p, 4), high_income=p >= 0.5, model_version=MODEL_VERSION)


@app.post("/predict-explained", response_model=ExplainedPrediction)
def predict_explained(
    person: Person,
    lang: str = Query("en", pattern="^(en|tr)$"),
    x_traffic_source: str = Header("live"),
):
    try:
        model = get_model()
        explainer = get_explainer()
    except FileNotFoundError:
        raise HTTPException(503, "model not available")
    except ValueError as e:  # model trained before explanations existed
        raise HTTPException(503, str(e))
    row = prepare(pd.DataFrame([person.model_dump()]))
    p = float(model.predict_proba(row)[0, 1])
    log_prediction(person, p, "predict-explained", x_traffic_source)
    exp = explainer.explain(row)

    # The numbers are the source of truth; the LLM summary is best-effort.
    story, error = None, None
    try:
        story = narrative.narrate(p, exp["top_factors"], lang)
    except Exception as e:  # noqa: BLE001 - any LLM failure degrades to numbers only
        error = f"{type(e).__name__}: {str(e)[:200]}"

    return ExplainedPrediction(
        high_income_probability=round(p, 4),
        high_income=p >= 0.5,
        model_version=MODEL_VERSION,
        confidence=confidence(p),
        narrative=story,
        narrative_error=error,
        **exp,
    )
