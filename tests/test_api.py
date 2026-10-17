import joblib
import pytest
from fastapi.testclient import TestClient

from app import main
from training import train


@pytest.fixture
def client(synthetic_df, tmp_path, monkeypatch):
    model, _ = train.train(synthetic_df, 0.1, 100)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(main, "MODEL_URI", str(path))
    main.get_model.cache_clear()
    yield TestClient(main.app)
    main.get_model.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probability(client, person):
    r = client.post("/predict", json=person)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["high_income_probability"] <= 1.0
    assert body["high_income"] == (body["high_income_probability"] >= 0.5)


def test_unseen_category_does_not_crash(client, person):
    r = client.post("/predict", json={**person, "native_country": "Atlantis"})
    assert r.status_code == 200


def test_invalid_input_rejected(client, person):
    assert client.post("/predict", json={**person, "age": 5}).status_code == 422


def test_missing_model_returns_503(monkeypatch, person):
    monkeypatch.setattr(main, "MODEL_URI", "/nonexistent/model.joblib")
    main.get_model.cache_clear()
    assert TestClient(main.app).post("/predict", json=person).status_code == 503
    main.get_model.cache_clear()


def test_model_trained_on_bigquery_dtypes_serves(synthetic_df, tmp_path, monkeypatch, person):
    """Regression: BigQuery returns nullable Int64 columns; serving must still work."""
    from training.features import NUMERIC, TARGET

    bq_like = synthetic_df.astype({c: "Int64" for c in NUMERIC + [TARGET]})
    model, _ = train.train(bq_like, 0.1, 100)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(main, "MODEL_URI", str(path))
    main.get_model.cache_clear()
    assert TestClient(main.app).post("/predict", json=person).status_code == 200
    main.get_model.cache_clear()
