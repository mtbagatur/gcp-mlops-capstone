"""Per-prediction Shapley attributions (requirement G6).

Uses a permutation explainer on the encoded features: the model's native
categorical splits are not handled exactly by TreeSHAP, while permutation
Shapley values add up exactly to the model output (log-odds) by construction.
"""
import threading

import numpy as np
import shap


class Explainer:
    def __init__(self, model):
        if not hasattr(model, "explain_background_"):
            raise ValueError("model has no explain_background_; retrain with training.train.fit_model")
        self.prep = model.named_steps["prep"]
        self.clf = model.named_steps["clf"]
        self.names = [n.split("__", 1)[1] for n in self.prep.get_feature_names_out()]
        bg = self.prep.transform(model.explain_background_)
        self.max_evals = 2 * len(self.names) + 1 + 200
        self._ex = shap.PermutationExplainer(
            self.clf.decision_function, shap.maskers.Independent(bg, max_samples=len(bg)), seed=0
        )
        self._lock = threading.Lock()
        self._ex(bg[:1], max_evals=self.max_evals, silent=True)  # JIT warm-up

    def explain(self, row, top_k: int = 3) -> dict:
        """row: one prepared input row (DataFrame). Contributions are in log-odds."""
        x = self.prep.transform(row)
        # shap samples permutations from numpy's global RNG. Re-seed per call so the same
        # input always gets the same explanation; the lock keeps threads from interleaving.
        with self._lock:
            np.random.seed(0)
            e = self._ex(x, max_evals=self.max_evals, silent=True)
        contrib = dict(zip(self.names, e.values[0].astype(float)))
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        values = row.iloc[0].to_dict()
        return {
            "base_log_odds": round(float(np.ravel(e.base_values)[0]), 4),
            "contributions": {k: round(v, 4) for k, v in contrib.items()},
            "top_factors": [
                {
                    "feature": k,
                    "value": values[k] if not isinstance(values[k], float) else round(values[k], 2),
                    "contribution": round(v, 4),
                    "direction": "raises" if v > 0 else "lowers",
                }
                for k, v in top
            ],
        }
