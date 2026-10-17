"""Stage 1 check: the split ratios are close to 80/10/10 and the label rate is similar."""
import os
import sys

from google.cloud import bigquery

SQL = """
SELECT split, COUNT(*) AS n, ROUND(AVG(high_income), 3) AS positive_rate
FROM `{project}.mlops.income_training`
GROUP BY split ORDER BY split
"""

project = os.environ["PROJECT_ID"]
rows = list(bigquery.Client(project=project).query(SQL.format(project=project)).result())
total = sum(r.n for r in rows)
expected = {"TRAIN": 0.80, "VALIDATE": 0.10, "TEST": 0.10}
ok = True
for r in rows:
    share = r.n / total
    flag = "ok" if abs(share - expected[r.split]) < 0.02 else "OFF"
    ok &= flag == "ok"
    print(f"{r.split:<9} n={r.n:<6} share={share:.3f} positive_rate={r.positive_rate} {flag}")
sys.exit(0 if ok else 1)
