"""Summarize Qwen3-VL view-ablation results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/exp09_qwen3vl_view_ablation"
CONFIGS = ("front", "front_side", "front_side_back", "four_view")
CRITERIA = (
    "components", "silhouette", "sewing_pattern", "material_drape",
    "motif_texture", "regional_period", "wearing_context",
)


rows = []
schema_attempts = []
for path in sorted(EXP.glob("*_front.json")):
    asset = path.name.removesuffix("_front.json")
    for config in CONFIGS:
        record = json.loads((EXP / f"{asset}_{config}.json").read_text(encoding="utf-8"))
        schema_attempts.append(record.get("schema_attempts", 1))
        for criterion in CRITERIA:
            rows.append(
                {
                    "asset_id": asset,
                    "view_config": config,
                    "criterion": criterion,
                    "score": record["assessment"][criterion]["score"],
                }
            )

with (EXP / "criterion_records.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by_config = {}
for config in CONFIGS:
    subset = [row for row in rows if row["view_config"] == config]
    numeric = [int(row["score"]) for row in subset if row["score"] != "NA"]
    by_config[config] = {
        "criterion_records": len(subset),
        "numeric_records": len(numeric),
        "NA_rate": round(sum(row["score"] == "NA" for row in subset) / len(subset), 4),
        "mean_numeric_score": round(sum(numeric) / len(numeric), 4) if numeric else None,
    }

groups = defaultdict(list)
for row in rows:
    groups[(row["asset_id"], row["criterion"])].append(str(row["score"]))
view_stable = sum(len(set(values)) == 1 for values in groups.values())
cultural = [
    values for (asset, criterion), values in groups.items()
    if criterion in ("regional_period", "wearing_context")
]
summary = {
    "assets": 7,
    "view_configurations": 4,
    "assessment_count": 28,
    "criterion_records": len(rows),
    "exact_view_stability_rate": round(view_stable / len(groups), 4),
    "cultural_NA_preservation_rate": round(
        sum(all(value == "NA" for value in values) for values in cultural)
        / len(cultural),
        4,
    ),
    "first_pass_schema_validity_rate": round(
        sum(attempt == 1 for attempt in schema_attempts) / len(schema_attempts),
        4,
    ),
    "by_view_config": by_config,
    "interpretation": "View sensitivity of one VLM; not human cultural validation.",
}
(EXP / "analysis_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
