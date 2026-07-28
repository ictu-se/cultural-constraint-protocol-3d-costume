"""Create transparent researcher technical assessments for named-tool assets.

Regional/period consistency and wearing context remain NA because no specialist
or community validation was collected. The scores are protocol applications,
not cultural-authenticity judgements.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "exp07_named_tool_protocol_application"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "manuscript" / "figures" / "named_tool_asset_examples.png"

CRITERIA = [
    "garment_components",
    "silhouette_and_proportion",
    "sewing_pattern_plausibility",
    "material_and_drape",
    "motif_and_texture",
    "regional_period_consistency",
    "wearing_context",
]
WEIGHTS = {
    "garment_components": 0.16,
    "silhouette_and_proportion": 0.15,
    "sewing_pattern_plausibility": 0.16,
    "material_and_drape": 0.12,
    "motif_and_texture": 0.15,
    "regional_period_consistency": 0.16,
    "wearing_context": 0.10,
}

ASSETS = {
    "BL_AODAI_001": {
        "tool": "Blender 5.1.2",
        "scores": [2, 2, 1, 1, 1, None, None],
        "notes": [
            "tunic panels, collar, sleeves, side openings, and trousers are inspectable",
            "recognizable proxy silhouette; proportions are researcher-authored",
            "separate component meshes exist but no validated sewing pattern or seam graph",
            "procedural material parameters and renders only; no textile calibration",
            "generic procedural ornament without object-level motif source",
            "NA: documentary fields do not establish object-level region/period agreement",
            "NA: no specialist assessment of a declared wearing context",
        ],
    },
    "BL_AODAI_002": {
        "tool": "Blender 5.1.2",
        "scores": [2, 2, 1, 1, 1, None, None],
        "notes": [
            "same inspectable component set as baseline",
            "long-panel variation remains recognizable but lacks sourced proportion ranges",
            "no validated sewing pattern or seam graph",
            "procedural material only",
            "generic procedural ornament without object-level source",
            "NA: no object-level regional-period source",
            "NA: no specialist context assessment",
        ],
    },
    "BL_AODAI_003": {
        "tool": "Blender 5.1.2",
        "scores": [2, 1, 1, 1, 1, None, None],
        "notes": [
            "declared proxy components remain present",
            "strong flared-hem perturbation conflicts with the baseline proportion declaration",
            "no validated sewing pattern or seam graph",
            "procedural material only",
            "generic procedural ornament without object-level source",
            "NA: no object-level regional-period source",
            "NA: no specialist context assessment",
        ],
    },
    "BL_AODAI_004": {
        "tool": "Blender 5.1.2",
        "scores": [1, 2, 1, 1, 1, None, None],
        "notes": [
            "short sleeves deliberately conflict with the baseline long-sleeve component declaration",
            "remaining silhouette is recognizable as the procedural proxy",
            "no validated sewing pattern or seam graph",
            "procedural material only",
            "generic procedural ornament without object-level source",
            "NA: no object-level regional-period source",
            "NA: no specialist context assessment",
        ],
    },
    "BL_AODAI_005": {
        "tool": "Blender 5.1.2",
        "scores": [2, 1, 1, 1, 1, None, None],
        "notes": [
            "declared proxy components remain present",
            "slim/high-slit perturbation conflicts with baseline proportions",
            "no validated sewing pattern or seam graph",
            "procedural material only",
            "generic procedural ornament without object-level source",
            "NA: no object-level regional-period source",
            "NA: no specialist context assessment",
        ],
    },
    "TSR_AODAI_001": {
        "tool": "TripoSR commit 107cefd",
        "scores": [1, 2, 0, 1, 1, None, None],
        "notes": [
            "single fused person-garment mesh; garment layers and closures are not separable",
            "front-view garment length and outline are recovered; unseen surfaces are inferred",
            "no panels, seams, closures, or sewability evidence in reconstructed mesh",
            "vertex appearance derives from one image and does not encode physical material",
            "front pattern is partially visible but identity and placement cannot be verified in 3D",
            "NA: image metadata is insufficient for regional-period validation",
            "NA: no specialist assessment of the photographed context",
        ],
    },
    "TSR_AODAI_002": {
        "tool": "TripoSR commit 107cefd",
        "scores": [1, 2, 0, 1, 1, None, None],
        "notes": [
            "single fused person-garment reconstruction; proxy layers remain non-separable",
            "controlled full-body input preserves the gross long-panel silhouette; unseen surfaces remain inferred",
            "no panels, seams, closures, or sewability evidence",
            "single-image appearance is not physical material evidence",
            "rendered edge accents smear across fused geometry and are not verifiable motifs",
            "NA: the procedural proxy has no expert-supported regional-period claim",
            "NA: no specialist context assessment",
        ],
    },
}

rows = []
summaries = []
for asset_id, asset in ASSETS.items():
    assessable = []
    for criterion, score, note in zip(CRITERIA, asset["scores"], asset["notes"]):
        rows.append(
            {
                "asset_id": asset_id,
                "tool": asset["tool"],
                "profile": "modern_ao_dai_documentary_test_slot",
                "criterion": criterion,
                "score_0_to_3_or_NA": "NA" if score is None else score,
                "assessor": "author technical assessment; no cultural authority",
                "evidence": note,
                "claim_status": "protocol demonstration, not expert validation",
            }
        )
        if score is not None:
            assessable.append((criterion, score))
    weight_total = sum(WEIGHTS[c] for c, _ in assessable)
    audit_score = 100 * sum(WEIGHTS[c] * score / 3 for c, score in assessable) / weight_total
    summaries.append(
        {
            "asset_id": asset_id,
            "tool": asset["tool"],
            "assessable_criteria": len(assessable),
            "NA_criteria": len(CRITERIA) - len(assessable),
            "audit_score_assessable_only": round(audit_score, 2),
            "hard_flags_score_below_2": sum(score < 2 for _, score in assessable),
            "public_cultural_approval": "not permitted",
        }
    )

with (OUT / "criterion_assessments.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT / "asset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
(OUT / "summary.json").write_text(
    json.dumps(
        {
            "asset_count": len(ASSETS),
            "pipelines": ["Blender 5.1.2 procedural generation", "TripoSR single-image reconstruction"],
            "assessment_type": "author technical assessment",
            "missing_by_design": ["regional_period_consistency", "wearing_context"],
            "claim_boundary": "No expert or community cultural validation.",
        },
        indent=2,
    ),
    encoding="utf-8",
)

# Four-column evidence figure: input/output diversity without implying approval.
panels = [
    (
        ROOT
        / "experiments"
        / "exp05_named_tool_aodai_assets"
        / "renders"
        / "BL_AODAI_001_front.png",
        "Blender baseline",
    ),
    (
        ROOT
        / "experiments"
        / "exp05_named_tool_aodai_assets"
        / "renders"
        / "BL_AODAI_004_front.png",
        "Blender short-sleeve test",
    ),
    (
        ROOT
        / "experiments"
        / "exp06_triposr_reconstruction_ultra"
        / "renders"
        / "TSR_AODAI_001_front.png",
        "TripoSR reconstruction 1",
    ),
    (
        ROOT
        / "experiments"
        / "exp06_triposr_reconstruction_ultra"
        / "renders"
        / "TSR_AODAI_002_front.png",
        "TripoSR controlled-input reconstruction",
    ),
]
tile_w, tile_h = 520, 690
canvas = Image.new("RGB", (tile_w * len(panels), tile_h + 90), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=24)
for index, (path, label) in enumerate(panels):
    image = Image.open(path).convert("RGB")
    image.thumbnail((tile_w - 20, tile_h - 20))
    x = index * tile_w + (tile_w - image.width) // 2
    y = (tile_h - image.height) // 2
    canvas.paste(image, (x, y))
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(
        (index * tile_w + (tile_w - (box[2] - box[0])) // 2, tile_h + 25),
        label,
        fill="black",
        font=font,
    )
canvas.save(FIG, quality=95)
