"""Atomic pairwise Qwen3-VL judge with reference anchors and position swaps."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_qwen3vl_audit import CLI, MMPROJ, MODEL, ROOT, extract_json


OUT = ROOT / "experiments/exp11_qwen3vl_atomic_pairwise"
RENDERS = ROOT / "experiments/exp05_named_tool_aodai_assets/renders"
CASES = {
    "BL_AODAI_002": ("longer front/back panels", "silhouette"),
    "BL_AODAI_003": ("flared hem", "silhouette"),
    "BL_AODAI_004": ("short sleeves", "components"),
    "BL_AODAI_005": ("slimmer body and higher side slit", "silhouette"),
}
CRITERIA = {
    "components": (
        "presence, absence, or type of visible garment parts such as sleeves, "
        "collar, panels, openings, and trousers"
    ),
    "silhouette": (
        "outer shape and visible proportions such as length, width, flare, "
        "fittedness, or slit height"
    ),
    "sewing_pattern": (
        "visible evidence of panel boundaries, seams, closures, joins, or "
        "sewable construction; shading alone is not evidence"
    ),
    "material_drape": (
        "visible material response, thickness, folds, stiffness, or drape; "
        "lighting and color alone are not evidence"
    ),
    "motif_texture": (
        "visible ornament, motif placement, texture identity, or surface pattern"
    ),
}

PROMPT = """You are an automated visual comparison tool, not a cultural expert.
The image contains REFERENCE front/side views and CANDIDATE front/side views.
Evaluate exactly one atomic criterion:

CRITERION: {criterion}
DEFINITION: {definition}
ATOMIC PROPOSITION TO VERIFY: {proposition}

Decide whether the atomic proposition is visibly supported. Ignore differences
belonging only to other criteria. Do not infer unseen geometry. Do not treat
camera, lighting, shading, or color as structural evidence. Cultural
authenticity, region, period, and context are out of scope.

Return exactly one JSON object:
{{
  "criterion": "{criterion}",
  "criterion_changed": true|false,
  "visible_evidence": [],
  "alternative_artifact_explanation": [],
  "confidence": "low"|"medium"|"high",
  "claim_boundary": "atomic technical comparison; not cultural validation"
}}"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sheet(case_id: str, order: str) -> tuple[Path, list[Path]]:
    baseline = [
        RENDERS / "BL_AODAI_001_front.png",
        RENDERS / "BL_AODAI_001_side.png",
    ]
    variant = [
        RENDERS / f"{case_id}_front.png",
        RENDERS / f"{case_id}_side.png",
    ]
    reference, candidate = (
        (baseline, variant) if order == "baseline_left" else (variant, baseline)
    )
    groups = (("REFERENCE", reference), ("CANDIDATE", candidate))
    tile_w, tile_h, header = 700, 900, 70
    canvas = Image.new("RGB", (tile_w * 2, (tile_h + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=30)
    files = []
    for col, (group, images) in enumerate(groups):
        for row, (view, path) in enumerate(zip(("FRONT", "SIDE"), images)):
            files.append(path)
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w - 30, tile_h - 30))
            x0, y0 = col * tile_w, row * (tile_h + header)
            draw.rectangle((x0, y0, x0 + tile_w, y0 + header), fill="#20242b")
            draw.text(
                (x0 + 18, y0 + 16), f"{group} {view}", fill="white", font=font
            )
            x = x0 + (tile_w - image.width) // 2
            y = y0 + header + (tile_h - image.height) // 2
            canvas.paste(image, (x, y))
    output = OUT / f"{case_id}_{order}.png"
    canvas.save(output, quality=95)
    return output, files


def proposition(case_id: str, criterion: str, intervention: str, target: str) -> str:
    if criterion == target:
        return f"The candidate visibly has this controlled change: {intervention}."
    alternatives = {
        "components": (
            "The candidate visibly changes sleeve length, collar type, component "
            "inventory, or the presence of an opening."
        ),
        "silhouette": (
            "The candidate visibly changes tunic length, body width, hem flare, "
            "fittedness, or side-slit height."
        ),
        "sewing_pattern": (
            "The candidate visibly changes seams, closures, panel boundaries, "
            "joins, or sewable construction."
        ),
        "material_drape": (
            "The candidate visibly changes material thickness, stiffness, folds, "
            "or drape independently of lighting."
        ),
        "motif_texture": (
            "The candidate visibly changes ornament, motif placement, texture "
            "identity, or surface pattern."
        ),
    }
    return alternatives[criterion]


def judge(
    sheet: Path, criterion: str, atomic_proposition: str
) -> tuple[dict, str, str, int, float]:
    prompt = PROMPT.format(
        criterion=criterion,
        definition=CRITERIA[criterion],
        proposition=atomic_proposition,
    )
    started = time.time()
    last_error = None
    for attempt in range(2):
        attempt_prompt = prompt + (
            "" if attempt == 0 else
            "\nSchema repair: include every required key and no text outside JSON."
        )
        command = [
            str(CLI), "-m", str(MODEL), "--mmproj", str(MMPROJ),
            "-ngl", "99", "--ctx-size", "8192",
            "--image-min-tokens", "1024", "--image-max-tokens", "1024",
            "--temp", "0", "-n", "500", "--image", str(sheet),
            "-p", attempt_prompt,
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        try:
            result = extract_json(proc.stdout)
            if result.get("criterion") != criterion:
                raise ValueError("criterion echo mismatch")
            if not isinstance(result.get("criterion_changed"), bool):
                raise ValueError("criterion_changed is not boolean")
            return (
                result, proc.stdout, proc.stderr, attempt + 1,
                round(time.time() - started, 3),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Atomic judge schema failed: {last_error}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for case_id, (intervention, target) in CASES.items():
        for order in ("baseline_left", "variant_left"):
            sheet, source_images = make_sheet(case_id, order)
            for criterion in CRITERIA:
                atomic_proposition = proposition(
                    case_id, criterion, intervention, target
                )
                result, stdout, stderr, attempts, elapsed = judge(
                    sheet, criterion, atomic_proposition
                )
                expected = criterion == target
                predicted = result["criterion_changed"]
                record = {
                    "case_id": case_id,
                    "intervention": intervention,
                    "target_criterion": target,
                    "criterion": criterion,
                    "atomic_proposition": atomic_proposition,
                    "order": order,
                    "expected_changed": expected,
                    "predicted_changed": predicted,
                    "correct": expected == predicted,
                    "schema_attempts": attempts,
                    "elapsed_seconds": elapsed,
                    "sheet_sha256": sha256(sheet),
                    "source_sha256": {
                        path.name: sha256(path) for path in source_images
                    },
                    "assessment": result,
                    "raw_stdout": stdout,
                    "runtime_stderr": stderr,
                }
                records.append(record)
                (OUT / f"{case_id}_{criterion}_{order}.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    tp = sum(r["expected_changed"] and r["predicted_changed"] for r in records)
    fp = sum(not r["expected_changed"] and r["predicted_changed"] for r in records)
    tn = sum(not r["expected_changed"] and not r["predicted_changed"] for r in records)
    fn = sum(r["expected_changed"] and not r["predicted_changed"] for r in records)
    pair_groups = {}
    for record in records:
        pair_groups.setdefault((record["case_id"], record["criterion"]), []).append(
            record["predicted_changed"]
        )
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    summary = {
        "cases": len(CASES),
        "criteria_per_case": len(CRITERIA),
        "position_orders": 2,
        "judgement_count": len(records),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round((tp + tn) / len(records), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0,
        "position_consistency_rate": round(
            sum(len(set(values)) == 1 for values in pair_groups.values())
            / len(pair_groups),
            4,
        ),
        "first_pass_schema_validity_rate": round(
            sum(r["schema_attempts"] == 1 for r in records) / len(records), 4
        ),
        "model_sha256": sha256(MODEL),
        "mmproj_sha256": sha256(MMPROJ),
        "claim_boundary": "Atomic technical comparison; not cultural validation.",
    }
    with (OUT / "judgements.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "case_id", "intervention", "target_criterion", "criterion", "order",
            "expected_changed", "predicted_changed", "correct", "schema_attempts",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
