"""Train the income model, evaluate it and enforce the quality gate.

Data comes from BigQuery (default) or a local CSV/Parquet with the same columns
as mlops.income_training. Writes model.joblib and metrics.json to --out.
Exits non-zero if ROC AUC on the TEST split is below --min-auc (requirement G3).
"""
import argparse
import json
import pathlib
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from training.features import CATEGORICAL, NUMERIC, TARGET, prepare
except ImportError:  # run as a script from training/
    from features import CATEGORICAL, NUMERIC, TARGET, prepare

SEED = 42


def load(source: str) -> pd.DataFrame:
    if source.startswith("bq://"):
        from google.cloud import bigquery

        table = source.removeprefix("bq://")
        return bigquery.Client(project=table.split(".")[0]).query(f"SELECT * FROM `{table}`").to_dataframe()
    if source.endswith(".parquet"):
        return pd.read_parquet(source)
    return pd.read_csv(source)


def build_model(learning_rate: float = 0.1, max_iter: int = 200) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        categorical_features=list(range(len(CATEGORICAL))),
        random_state=SEED,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def fit_model(tr: pd.DataFrame, learning_rate: float, max_iter: int, n_background: int = 100) -> Pipeline:
    """Fit on the TRAIN rows and attach a fixed background sample for Shapley explanations."""
    model = build_model(learning_rate, max_iter).fit(prepare(tr), tr[TARGET].astype(int))
    model.explain_background_ = prepare(tr.sample(min(n_background, len(tr)), random_state=SEED))
    return model


def train(df: pd.DataFrame, learning_rate: float, max_iter: int) -> tuple[Pipeline, dict]:
    tr = df[df["split"] == "TRAIN"]
    te = df[df["split"] == "TEST"]
    model = fit_model(tr, learning_rate, max_iter)
    auc = roc_auc_score(te[TARGET].astype(int), model.predict_proba(prepare(te))[:, 1])
    metrics = {
        "roc_auc": round(float(auc), 4),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "learning_rate": learning_rate,
        "max_iter": max_iter,
    }
    return model, metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="bq://mlops.income_training")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--min-auc", type=float, default=0.90)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=200)
    a = p.parse_args(argv)

    model, metrics = train(load(a.source), a.learning_rate, a.max_iter)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))

    if metrics["roc_auc"] < a.min_auc:
        print(f"QUALITY GATE FAILED: roc_auc {metrics['roc_auc']} < {a.min_auc}; model not saved", file=sys.stderr)
        return 1
    joblib.dump(model, out / "model.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
