"""Data checks run before training. Any failure stops the pipeline."""
import pandas as pd

try:
    from training.features import FEATURES, TARGET
except ImportError:
    from features import FEATURES, TARGET

EXPECTED_SPLIT = {"TRAIN": 0.80, "VALIDATE": 0.10, "TEST": 0.10}


def validate(df: pd.DataFrame, min_rows: int = 20_000) -> dict:
    """Return summary stats; raise ValueError listing every failed check."""
    errors = []
    missing = [c for c in FEATURES + [TARGET, "split"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    if len(df) < min_rows:
        errors.append(f"only {len(df)} rows (< {min_rows})")
    nulls = df[FEATURES + [TARGET]].isna().sum()
    if nulls.any():
        errors.append(f"nulls: {nulls[nulls > 0].to_dict()}")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        errors.append("target is not binary 0/1")

    positive_rate = float(df[TARGET].mean())
    if not 0.15 <= positive_rate <= 0.35:
        errors.append(f"positive rate {positive_rate:.3f} outside [0.15, 0.35]")

    shares = df["split"].value_counts(normalize=True).to_dict()
    for split, expected in EXPECTED_SPLIT.items():
        if abs(shares.get(split, 0.0) - expected) > 0.02:
            errors.append(f"{split} share {shares.get(split, 0.0):.3f} != {expected}")

    if (df["age"] < 16).any() or (df["hours_per_week"] <= 0).any():
        errors.append("out-of-range age or hours_per_week")

    if errors:
        raise ValueError("; ".join(errors))
    return {"rows": int(len(df)), "positive_rate": round(positive_rate, 4)}
