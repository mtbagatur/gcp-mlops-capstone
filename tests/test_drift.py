import numpy as np
import pandas as pd

from training.drift import drift_report, psi_categorical, psi_numeric


def test_same_distribution_is_stable(synthetic_df):
    tr = synthetic_df[synthetic_df.split == "TRAIN"]
    recent = tr.sample(500, random_state=1)
    report = drift_report(tr, recent)
    assert report["drifted"] == []
    assert max(report["psi"].values()) < 0.1


def test_shifted_numeric_feature_is_flagged(synthetic_df):
    tr = synthetic_df[synthetic_df.split == "TRAIN"]
    recent = tr.sample(500, random_state=1).copy()
    recent["age"] = recent["age"] + 25
    report = drift_report(tr, recent)
    assert report["drifted"][0] == "age"
    assert report["psi"]["age"] > 0.2


def test_new_categories_count_as_drift(synthetic_df):
    tr = synthetic_df[synthetic_df.split == "TRAIN"]
    recent = tr.sample(300, random_state=1).copy()
    recent["native_country"] = "Atlantis"
    assert psi_categorical(tr["native_country"], recent["native_country"]) > 0.2


def test_psi_is_zero_for_identical_samples():
    x = np.arange(1000)
    assert abs(psi_numeric(pd.Series(x), pd.Series(x))) < 1e-9
