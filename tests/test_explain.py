import numpy as np
import pytest

from app.explain import Explainer
from app import narrative
from app.narrative import build_prompt, is_clean
from training import train
from training.features import prepare


@pytest.fixture(scope="module")
def model(synthetic_df):
    m, _ = train.train(synthetic_df, 0.1, 100)
    return m


def test_shapley_values_add_up_to_model_output(model, synthetic_df):
    """Efficiency property: base + sum(contributions) == model log-odds."""
    ex = Explainer(model)
    row = prepare(synthetic_df.iloc[[3]])
    out = ex.explain(row)
    raw = model.decision_function(row)[0]
    assert np.isclose(out["base_log_odds"] + sum(out["contributions"].values()), raw, atol=1e-3)


def test_top_factors_sorted_by_magnitude(model, synthetic_df):
    out = Explainer(model).explain(prepare(synthetic_df.iloc[[5]]))
    mags = [abs(f["contribution"]) for f in out["top_factors"]]
    assert mags == sorted(mags, reverse=True) and len(mags) == 3
    for f in out["top_factors"]:
        assert f["direction"] == ("raises" if f["contribution"] > 0 else "lowers")


def test_model_without_background_is_rejected(model):
    import copy

    m = copy.deepcopy(model)
    del m.explain_background_
    with pytest.raises(ValueError, match="explain_background_"):
        Explainer(m)


def test_prompt_contains_causality_guardrail():
    prompt = build_prompt(0.8, [{"feature": "age", "contribution": 0.5}], "tr")
    assert "NOT causes" in prompt and "Turkish" in prompt


def test_same_input_same_explanation(model, synthetic_df):
    ex = Explainer(model)
    row = prepare(synthetic_df.iloc[[9]])
    assert ex.explain(row) == ex.explain(row)


def test_clean_text_check():
    assert is_clean({"summary": "Model, yönetici mesleğini ağırlıklı buldu (%89,9).", "key_factors": ["45 yaş"]})
    assert not is_clean({"summary": "Model mesleğĿ", "key_factors": []})


def test_garbled_answer_is_retried_with_thinking(monkeypatch):
    budgets = []

    def fake_generate(prompt, thinking_budget):
        budgets.append(thinking_budget)
        return {"summary": "bozuk Ŀ" if thinking_budget == 0 else "temiz", "key_factors": []}

    monkeypatch.setattr(narrative, "_generate", fake_generate)
    assert narrative.narrate(0.9, [], "tr")["summary"] == "temiz"
    assert budgets == [0, 256]
