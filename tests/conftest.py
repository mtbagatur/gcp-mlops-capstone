import numpy as np
import pandas as pd
import pytest

from training.features import TARGET

CATS = {
    "workclass": ["Private", "Self-emp-not-inc", "Local-gov"],
    "education": ["Bachelors", "HS-grad", "Masters"],
    "marital_status": ["Married-civ-spouse", "Never-married", "Divorced"],
    "occupation": ["Exec-managerial", "Craft-repair", "Sales"],
    "relationship": ["Husband", "Not-in-family", "Own-child"],
    "race": ["White", "Black", "Asian-Pac-Islander"],
    "sex": ["Male", "Female"],
    "native_country": ["United-States", "Mexico", "India"],
}


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Small dataset with the production schema and a learnable signal."""
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({k: rng.choice(v, n) for k, v in CATS.items()})
    df["age"] = rng.integers(18, 70, n)
    df["education_num"] = rng.integers(1, 17, n)
    df["capital_gain"] = rng.choice([0, 0, 0, 5000, 15000], n)
    df["capital_loss"] = rng.choice([0, 0, 0, 1900], n)
    df["hours_per_week"] = rng.integers(10, 70, n)
    score = (df["education_num"] - 9) * 0.5 + (df["age"] - 40) * 0.05 + (df["capital_gain"] > 0) * 3
    # Threshold chosen so ~25% are positive, like the real census data.
    df[TARGET] = (score + rng.normal(0, 1, n) > 3.0).astype(int)
    df["split"] = rng.permutation(["TRAIN"] * 1600 + ["VALIDATE"] * 200 + ["TEST"] * 200)
    return df


@pytest.fixture
def person() -> dict:
    return {
        "age": 45, "workclass": "Private", "education": "Masters", "education_num": 14,
        "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital_gain": 15000, "capital_loss": 0, "hours_per_week": 50,
        "native_country": "United-States",
    }
