"""Feature schema shared by training and the API."""

NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "high_income"


def prepare(df):
    """Normalise dtypes so training and serving see identical inputs.

    BigQuery returns nullable Int64 columns, which sklearn turns into object
    arrays; the API builds plain int/str frames. Cast both to the same types.
    """
    out = df[FEATURES].copy()
    out[NUMERIC] = out[NUMERIC].astype("float64")
    out[CATEGORICAL] = out[CATEGORICAL].astype(str)
    return out
