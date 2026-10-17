import pytest

from training.validate import validate


def test_valid_data_passes(synthetic_df):
    stats = validate(synthetic_df, min_rows=1000)
    assert stats["rows"] == len(synthetic_df)


def test_too_few_rows_fails(synthetic_df):
    with pytest.raises(ValueError, match="rows"):
        validate(synthetic_df.head(100), min_rows=1000)


def test_missing_column_fails(synthetic_df):
    with pytest.raises(ValueError, match="missing columns"):
        validate(synthetic_df.drop(columns=["occupation"]), min_rows=1000)


def test_skewed_split_fails(synthetic_df):
    df = synthetic_df.copy()
    df["split"] = "TRAIN"
    with pytest.raises(ValueError, match="share"):
        validate(df, min_rows=1000)


def test_label_drift_fails(synthetic_df):
    df = synthetic_df.copy()
    df["high_income"] = 1
    with pytest.raises(ValueError, match="positive rate"):
        validate(df, min_rows=1000)
