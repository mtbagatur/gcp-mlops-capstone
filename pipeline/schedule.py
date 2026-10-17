"""Create (or replace) the nightly retraining schedule (requirement G4).

  python -m pipeline.schedule            # 03:00 Europe/Istanbul every night
  python -m pipeline.schedule --pause    # stop nightly runs (e.g. between demos)
  python -m pipeline.schedule --resume

The schedule stores the compiled pipeline spec, so re-run this after changing the
DAG. Component code changes need no re-run: steps use the trainer image's
:latest tag, which CI rebuilds on every push.
"""
import argparse

from google.cloud import aiplatform
from kfp import compiler

from pipeline.pipeline import income_pipeline

NAME = "income-nightly"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="mtb-mlops-capstone")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--cron", default="TZ=Europe/Istanbul 0 3 * * *")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pause", action="store_true")
    g.add_argument("--resume", action="store_true")
    a = p.parse_args()

    aiplatform.init(project=a.project, location=a.region)
    existing = aiplatform.PipelineJobSchedule.list(filter=f'display_name="{NAME}"')

    if a.pause or a.resume:
        for s in existing:
            s.pause() if a.pause else s.resume()
            print(f"{'paused' if a.pause else 'resumed'} {s.resource_name}")
        return

    for s in existing:
        s.delete()
        print(f"deleted old schedule {s.resource_name}")

    compiler.Compiler().compile(income_pipeline, "income_pipeline.yaml")
    job = aiplatform.PipelineJob(
        display_name="income-training",
        template_path="income_pipeline.yaml",
        pipeline_root=f"gs://{a.project}-mlops/pipeline-root",
        parameter_values={"project": a.project, "region": a.region},
        enable_caching=False,
    )
    sched = job.create_schedule(
        display_name=NAME,
        cron=a.cron,
        max_concurrent_run_count=1,
        service_account=f"vertex-pipelines@{a.project}.iam.gserviceaccount.com",
    )
    print(f"created {sched.resource_name} ({a.cron})")


if __name__ == "__main__":
    main()
