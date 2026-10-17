import json

from training import train


def test_model_learns_signal(synthetic_df):
    _, metrics = train.train(synthetic_df, learning_rate=0.1, max_iter=100)
    assert metrics["roc_auc"] > 0.85


def test_training_is_reproducible(synthetic_df):
    """G2: same data and parameters give the same metric."""
    _, a = train.train(synthetic_df, 0.1, 100)
    _, b = train.train(synthetic_df, 0.1, 100)
    assert a == b


def test_quality_gate_blocks_weak_model(synthetic_df, tmp_path):
    """G3: a model below the threshold is never written, and the run fails."""
    src = tmp_path / "data.csv"
    synthetic_df.to_csv(src, index=False)
    rc = train.main(["--source", str(src), "--out", str(tmp_path / "out"), "--min-auc", "0.999"])
    assert rc == 1
    assert not (tmp_path / "out" / "model.joblib").exists()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["roc_auc"] < 0.999


def test_passing_model_is_saved(synthetic_df, tmp_path):
    src = tmp_path / "data.csv"
    synthetic_df.to_csv(src, index=False)
    rc = train.main(["--source", str(src), "--out", str(tmp_path / "out"), "--min-auc", "0.5"])
    assert rc == 0
    assert (tmp_path / "out" / "model.joblib").exists()
