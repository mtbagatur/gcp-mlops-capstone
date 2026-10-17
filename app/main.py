"""Prediction API, deployed to Cloud Run (scales to zero, requirement G5)."""
import os
import pathlib
import tempfile
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.features import prepare

MODEL_URI = os.getenv("MODEL_URI", "artifacts/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")

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
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(person: Person):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(503, "model not available")
    row = prepare(pd.DataFrame([person.model_dump()]))
    p = float(model.predict_proba(row)[0, 1])
    return Prediction(high_income_probability=round(p, 4), high_income=p >= 0.5, model_version=MODEL_VERSION)
