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
    main.get_explainer.cache_clear()
    yield TestClient(main.app)
    main.get_model.cache_clear()
    main.get_explainer.cache_clear()


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


def test_explained_prediction_with_narrative(client, person, monkeypatch):
    calls = {}

    def fake_narrate(p, factors, lang):
        calls.update(p=p, factors=factors, lang=lang)
        return {"summary": "The model weighed capital gain heavily.", "key_factors": ["a", "b", "c"]}

    monkeypatch.setattr(main.narrative, "narrate", fake_narrate)
    r = client.post("/predict-explained?lang=tr", json=person)
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_factors"]) == 3
    assert body["narrative"]["summary"].startswith("The model")
    assert body["narrative_error"] is None
    assert body["confidence"] in {"low", "medium", "high"}
    assert calls["lang"] == "tr" and len(calls["factors"]) == 3


def test_llm_failure_degrades_to_numbers(client, person, monkeypatch):
    def broken(*a, **k):
        raise TimeoutError("vertex timeout")

    monkeypatch.setattr(main.narrative, "narrate", broken)
    r = client.post("/predict-explained", json=person)
    assert r.status_code == 200
    assert r.json()["narrative"] is None
    assert "TimeoutError" in r.json()["narrative_error"]
    assert len(r.json()["top_factors"]) == 3


def test_unknown_language_rejected(client, person):
    assert client.post("/predict-explained?lang=de", json=person).status_code == 422


@pytest.mark.parametrize("p,expected", [(0.97, "high"), (0.02, "high"), (0.75, "medium"), (0.55, "low")])
def test_confidence_bands(p, expected):
    assert main.confidence(p) == expected


def test_model_without_background_returns_503(synthetic_df, tmp_path, monkeypatch, person):
    model, _ = train.train(synthetic_df, 0.1, 100)
    del model.explain_background_
    path = tmp_path / "old.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(main, "MODEL_URI", str(path))
    main.get_model.cache_clear()
    main.get_explainer.cache_clear()
    c = TestClient(main.app)
    assert c.post("/predict", json=person).status_code == 200
    assert c.post("/predict-explained", json=person).status_code == 503
    main.get_model.cache_clear()
    main.get_explainer.cache_clear()


def test_prediction_is_logged_as_structured_json(client, person, capsys):
    import json

    client.post("/predict", json=person)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if '"event": "prediction"' in ln]
    rec = json.loads(lines[-1])
    assert rec["endpoint"] == "predict" and rec["age"] == person["age"]
    assert 0 <= rec["probability"] <= 1


def test_synthetic_traffic_is_tagged(client, person, capsys):
    import json

    client.post("/predict", json=person, headers={"X-Traffic-Source": "synthetic"})
    lines = [ln for ln in capsys.readouterr().out.splitlines() if '"event": "prediction"' in ln]
    assert json.loads(lines[-1])["source"] == "synthetic"
