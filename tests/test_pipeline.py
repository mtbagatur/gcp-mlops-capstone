import yaml
from kfp import compiler

from pipeline.pipeline import income_pipeline


def test_pipeline_compiles_with_quality_gate(tmp_path):
    out = tmp_path / "p.yaml"
    compiler.Compiler().compile(income_pipeline, str(out))
    spec = yaml.safe_load(out.read_text())
    tasks = spec["root"]["dag"]["tasks"]
    assert {"extract-data", "validate-data", "check-drift", "train-model", "evaluate-model"} <= set(tasks)
    # register/deploy must live only inside the gate's condition branch
    gate = tasks["condition-1"]
    assert "min_auc" in gate["triggerPolicy"]["condition"]
    assert "register-and-deploy" not in tasks
