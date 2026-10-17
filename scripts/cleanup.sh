#!/usr/bin/env bash
# Tear down everything this project creates in GCP, so it costs nothing between demos.
# Dry run by default: prints what would be deleted. Pass --yes to actually delete.
#   scripts/cleanup.sh            # show the plan
#   scripts/cleanup.sh --yes      # delete
# The whole system can be rebuilt from the README in about half an hour.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-mtb-mlops-capstone}"
REGION="${REGION:-us-central1}"
DO="${1:-}"

run() {
  if [[ "$DO" == "--yes" ]]; then
    echo "+ $*"
    "$@"
  else
    echo "[dry-run] $*"
  fi
}

cd "$(dirname "$0")/.."

echo "== 1. Stop nightly retraining"
run .venv/bin/python -m pipeline.schedule --project "$PROJECT_ID" --region "$REGION" --delete

echo "== 2. Delete the Cloud Run service (deployed by Cloud Build, not Terraform)"
if gcloud run services describe income-api --region "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  run gcloud run services delete income-api --region "$REGION" --project "$PROJECT_ID" --quiet
fi

echo "== 3. Delete Model Registry entries"
for m in $(gcloud ai models list --region "$REGION" --project "$PROJECT_ID" \
             --filter='displayName=income-model' --format='value(name)' 2>/dev/null); do
  run gcloud ai models delete "$m" --region "$REGION" --project "$PROJECT_ID" --quiet
done

echo "== 4. Destroy Terraform-managed infrastructure (bucket, datasets, registry, IAM, alerts, budget)"
if [[ "$DO" == "--yes" ]]; then
  (cd infra/terraform && terraform destroy -auto-approve)
else
  (cd infra/terraform && terraform plan -destroy -no-color | grep -E '^Plan:|will be destroyed' || true)
fi

echo "== Left in place (free): Cloud Build trigger and GitHub connection, the <project>_cloudbuild bucket."
[[ "$DO" == "--yes" ]] || echo "Nothing was deleted. Re-run with --yes to apply."
