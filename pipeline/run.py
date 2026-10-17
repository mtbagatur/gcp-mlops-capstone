"""Compile the pipeline and optionally submit a run.

  python -m pipeline.run --compile-only                 # writes income_pipeline.yaml
  python -m pipeline.run                                # submit with defaults
  python -m pipeline.run --min-auc 0.99                 # gate test: deploy must be skipped
"""
import argparse

from kfp import compiler

from pipeline.pipeline import income_pipeline

PIPELINE_FILE = "income_pipeline.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="mtb-mlops-capstone")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--min-auc", type=float, default=0.90)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--wait", action="store_true")
    a = p.parse_args()

    compiler.Compiler().compile(income_pipeline, PIPELINE_FILE)
    print(f"compiled {PIPELINE_FILE}")
    if a.compile_only:
        return

    from google.cloud import aiplatform

    aiplatform.init(project=a.project, location=a.region)
    job = aiplatform.PipelineJob(
        display_name="income-training",
        template_path=PIPELINE_FILE,
        pipeline_root=f"gs://{a.project}-mlops/pipeline-root",
        parameter_values={"project": a.project, "region": a.region, "min_auc": a.min_auc},
        enable_caching=False,
    )
    job.submit(service_account=f"vertex-pipelines@{a.project}.iam.gserviceaccount.com")
    print(job._dashboard_uri())
    if a.wait:
        job.wait()


if __name__ == "__main__":
    main()
