"""Summarize prompt stability and agreement with author technical assessments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/exp08_qwen3vl_ai_assisted_audit"
AUTHOR = ROOT / "experiments/exp07_named_tool_protocol_application/criterion_assessments.csv"
CRITERION_MAP = {
    "components": "garment_components",
    "silhouette": "silhouette_and_proportion",
    "sewing_pattern": "sewing_pattern_plausibility",
    "material_drape": "material_and_drape",
    "motif_texture": "motif_and_texture",
    "regional_period": "regional_period_consistency",
    "wearing_context": "wearing_context",
}


def numeric(value):
    return None if value == "NA" else int(value)


author = {}
with AUTHOR.open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        author[(row["asset_id"], row["criterion"])] = numeric(
            row["score_0_to_3_or_NA"]
        )

records = []
for path in sorted(EXP.glob("*_direct.json")):
    asset = path.name.removesuffix("_direct.json")
    for variant in ("direct", "evidence_first", "counterclaim"):
        record = json.loads((EXP / f"{asset}_{variant}.json").read_text(encoding="utf-8"))
        for vlm_name, author_name in CRITERION_MAP.items():
            score = numeric(record["assessment"][vlm_name]["score"])
            records.append(
                {
                    "asset_id": asset,
                    "variant": variant,
                    "criterion": author_name,
                    "vlm_score": "NA" if score is None else score,
                    "author_score": (
                        "NA" if author[(asset, author_name)] is None
                        else author[(asset, author_name)]
                    ),
                }
            )

with (EXP / "criterion_records.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

groups = defaultdict(list)
for row in records:
    groups[(row["asset_id"], row["criterion"])].append(row["vlm_score"])

stable = 0
comparable = 0
absolute_errors = []
na_matches = 0
na_total = 0
summaries = []
for (asset, criterion), values in sorted(groups.items()):
    all_same = len(set(values)) == 1
    stable += all_same
    author_score = author[(asset, criterion)]
    numeric_values = [v for v in values if v != "NA"]
    median = float(np.median(numeric_values)) if numeric_values else None
    if author_score is None:
        na_total += 1
        na_matches += all(v == "NA" for v in values)
    elif median is not None:
        comparable += 1
        absolute_errors.append(abs(median - author_score))
    summaries.append(
        {
            "asset_id": asset,
            "criterion": criterion,
            "direct": values[0],
            "evidence_first": values[1],
            "counterclaim": values[2],
            "prompt_stable": all_same,
            "vlm_median": "NA" if median is None else median,
            "author_score": "NA" if author_score is None else author_score,
        }
    )

with (EXP / "stability_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)

summary = {
    "assets": len({row["asset_id"] for row in records}),
    "prompt_variants": 3,
    "criterion_records": len(records),
    "asset_criterion_groups": len(groups),
    "exact_prompt_stability_rate": round(stable / len(groups), 4),
    "cultural_NA_preservation_rate": round(na_matches / na_total, 4),
    "comparable_visual_groups": comparable,
    "median_absolute_error_vs_author_technical_scores": (
        round(float(np.median(absolute_errors)), 4) if absolute_errors else None
    ),
    "mean_absolute_error_vs_author_technical_scores": (
        round(float(np.mean(absolute_errors)), 4) if absolute_errors else None
    ),
    "interpretation": (
        "Agreement is with one author technical assessment, not human cultural "
        "validation. The VLM remains an automated evidence gate."
    ),
}
(EXP / "analysis_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
