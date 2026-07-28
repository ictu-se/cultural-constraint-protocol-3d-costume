"""Test whether Qwen3-VL localizes known Blender proxy changes pairwise."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_qwen3vl_audit import CLI, MMPROJ, MODEL, ROOT, extract_json


OUT = ROOT / "experiments/exp10_qwen3vl_pairwise_defects"
RENDERS = ROOT / "experiments/exp05_named_tool_aodai_assets/renders"
CASES = {
    "BL_AODAI_002": {
        "intervention": "longer front/back panels",
        "expected_criterion": "silhouette",
    },
    "BL_AODAI_003": {
        "intervention": "flared hem",
        "expected_criterion": "silhouette",
    },
    "BL_AODAI_004": {
        "intervention": "short sleeves",
        "expected_criterion": "components",
    },
    "BL_AODAI_005": {
        "intervention": "slimmer body and higher side slit",
        "expected_criterion": "silhouette",
    },
}
VARIANTS = {
    "direct": "Identify only visible structural differences.",
    "evidence_first": "List visible differences internally before classifying them.",
    "skeptical": "Reject differences that could be explained only by lighting or camera.",
}

PROMPT = """You are comparing a baseline 3D garment proxy (images 1 and 2)
with a controlled variant (images 3 and 4). Images are front and side views.
You are not a cultural expert. Ignore cultural authenticity, region, period,
and context. Do not call shading or color a structural change.

{variant}

Return exactly one JSON object:
{
  "structural_change_detected": true|false,
  "changed_criteria": ["components"|"silhouette"|"sewing_pattern"|"material_drape"|"motif_texture"],
  "visible_differences": [],
  "possible_render_artifacts": [],
  "confidence": "low"|"medium"|"high",
  "claim_boundary": "pairwise technical change detection; not cultural validation"
}"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_contact_sheet(variant_asset: str, images: list[Path]) -> Path:
    labels = (
        "BASELINE FRONT", "BASELINE SIDE",
        "VARIANT FRONT", "VARIANT SIDE",
    )
    tile_w, tile_h, header = 700, 900, 60
    canvas = Image.new("RGB", (tile_w * 2, (tile_h + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=30)
    for index, (path, label) in enumerate(zip(images, labels)):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 30, tile_h - 30))
        col, row = index % 2, index // 2
        x0, y0 = col * tile_w, row * (tile_h + header)
        draw.rectangle((x0, y0, x0 + tile_w, y0 + header), fill="#20242b")
        draw.text((x0 + 18, y0 + 14), label, fill="white", font=font)
        x = x0 + (tile_w - image.width) // 2
        y = y0 + header + (tile_h - image.height) // 2
        canvas.paste(image, (x, y))
    output = OUT / f"{variant_asset}_comparison_sheet.png"
    canvas.save(output, quality=95)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for variant_asset, case in CASES.items():
        images = [
            RENDERS / "BL_AODAI_001_front.png",
            RENDERS / "BL_AODAI_001_side.png",
            RENDERS / f"{variant_asset}_front.png",
            RENDERS / f"{variant_asset}_side.png",
        ]
        contact_sheet = make_contact_sheet(variant_asset, images)
        for prompt_variant, instruction in VARIANTS.items():
            prompt = PROMPT.replace("{variant}", instruction)
            command = [
                str(CLI), "-m", str(MODEL), "--mmproj", str(MMPROJ),
                "-ngl", "99", "--ctx-size", "12288",
                "--image-min-tokens", "1024", "--image-max-tokens", "1024",
                "--temp", "0", "-n", "700",
                "--image", str(contact_sheet), "-p", prompt,
            ]
            started = time.time()
            proc = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            assessment = extract_json(proc.stdout)
            detected = bool(assessment["structural_change_detected"])
            changed = assessment["changed_criteria"]
            record = {
                "variant_asset": variant_asset,
                "prompt_variant": prompt_variant,
                **case,
                "elapsed_seconds": round(time.time() - started, 3),
                "detected": detected,
                "localized_expected_criterion": (
                    detected and case["expected_criterion"] in changed
                ),
                "image_sha256": {path.name: sha256(path) for path in images},
                "contact_sheet": str(contact_sheet.relative_to(ROOT)).replace("\\", "/"),
                "contact_sheet_sha256": sha256(contact_sheet),
                "assessment": assessment,
                "raw_stdout": proc.stdout,
                "runtime_stderr": proc.stderr,
            }
            results.append(record)
            (OUT / f"{variant_asset}_{prompt_variant}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    summary = {
        "case_count": len(CASES),
        "prompt_variants": len(VARIANTS),
        "comparison_count": len(results),
        "change_detection_rate": round(
            sum(r["detected"] for r in results) / len(results), 4
        ),
        "expected_criterion_localization_rate": round(
            sum(r["localized_expected_criterion"] for r in results) / len(results),
            4,
        ),
        "detection_prompt_consistency_rate": round(
            sum(
                len({
                    r["detected"] for r in results
                    if r["variant_asset"] == asset
                }) == 1
                for asset in CASES
            )
            / len(CASES),
            4,
        ),
        "model_sha256": sha256(MODEL),
        "mmproj_sha256": sha256(MMPROJ),
        "claim_boundary": "Pairwise technical change detection; not cultural validation.",
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
