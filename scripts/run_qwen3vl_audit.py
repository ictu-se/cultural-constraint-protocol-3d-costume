"""Run a bounded, reproducible Qwen3-VL visual audit over multi-view renders.

The model is used as an AI-assisted technical evaluator, not as a Vietnamese
cultural authority. Documentary criteria must remain NA without cited sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MODEL_DIR = WORKSPACE / "tools" / "Qwen3-VL-8B-Instruct-GGUF"
MODEL = MODEL_DIR / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ = MODEL_DIR / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
CLI = (
    Path.home()
    / "AppData/Local/Microsoft/WinGet/Packages"
    / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "llama-mtmd-cli.exe"
)
OUT = ROOT / "experiments" / "exp08_qwen3vl_ai_assisted_audit"

ASSETS = {
    **{
        f"BL_AODAI_{index:03d}": ROOT
        / "experiments/exp05_named_tool_aodai_assets/renders"
        for index in range(1, 6)
    },
    "TSR_AODAI_001": ROOT
    / "experiments/exp06_triposr_reconstruction_ultra/renders",
    "TSR_AODAI_002": ROOT
    / "experiments/exp06_triposr_reconstruction_ultra/renders",
}

PROMPT_CORE = """You are an AI-assisted visual inspector, not a cultural expert.
Inspect the supplied front, side, and back renders of one 3D asset.

Rules:
1. Score only visible image evidence. Never infer hidden construction.
2. A shaded line, color boundary, fold, or texture is not proof of a seam,
   sewing panel, closure, material physics, motif identity, region, period, or
   wearing context.
3. For the first five visually inspectable criteria, use 1 when the object is
   recognizable but required construction, physical, or sourcing evidence is
   substantially missing. Use NA only when the image cannot support any
   observation of that criterion.
4. Regional-period consistency and wearing context MUST be NA because no cited
   documentary profile or qualified human assessment is supplied.
5. Scores: 0=visible contradiction/severe failure; 1=major visible defects or
   missing evidence; 2=broad visible agreement with localized uncertainty;
   3=clear visible agreement. Do not use 3 when evidence is only a render.
6. This is a technical audit and cannot certify cultural authenticity.

{variant_instruction}

Return exactly one JSON object:
{
  "components": {"score": 0|1|2|"NA", "visible_evidence": [], "limitations": []},
  "silhouette": {"score": 0|1|2|"NA", "visible_evidence": [], "limitations": []},
  "sewing_pattern": {"score": 0|1|2|"NA", "visible_evidence": [], "limitations": []},
  "material_drape": {"score": 0|1|2|"NA", "visible_evidence": [], "limitations": []},
  "motif_texture": {"score": 0|1|2|"NA", "visible_evidence": [], "limitations": []},
  "regional_period": {"score": "NA", "visible_evidence": [], "limitations": []},
  "wearing_context": {"score": "NA", "visible_evidence": [], "limitations": []},
  "overall_uncertainty": "text",
  "claim_boundary": "AI-assisted technical inspection; not cultural validation"
}"""

PROMPT_VARIANTS = {
    "direct": (
        "Evaluate each criterion independently. Prefer conservative scores and "
        "name the exact visible feature supporting every non-NA score."
    ),
    "evidence_first": (
        "First identify visible evidence and missing evidence internally; then "
        "assign the score. Do not reward general visual attractiveness."
    ),
    "counterclaim": (
        "Actively test whether each apparent feature could instead be a render "
        "artifact or fused geometry. Lower the score when that alternative "
        "cannot be ruled out from the supplied views."
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("First JSON value is not an object")
    return value


def validate(record: dict) -> None:
    required = [
        "components", "silhouette", "sewing_pattern", "material_drape",
        "motif_texture", "regional_period", "wearing_context",
    ]
    for key in required:
        if key not in record:
            raise ValueError(f"Missing criterion: {key}")
    for key in ("regional_period", "wearing_context"):
        if record[key]["score"] != "NA":
            raise ValueError(f"{key} must remain NA")


def run_asset(asset_id: str, render_dir: Path, variant: str) -> dict:
    images = [
        render_dir / f"{asset_id}_front.png",
        render_dir / f"{asset_id}_side.png",
        render_dir / f"{asset_id}_back.png",
    ]
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    command = [
        str(CLI), "-m", str(MODEL), "--mmproj", str(MMPROJ),
        "-ngl", "99", "--ctx-size", "12288", "--image-min-tokens", "1024",
        "--image-max-tokens", "1024", "--temp", "0", "-n", "1000",
    ]
    if variant == "evidence_first":
        images = [images[2], images[0], images[1]]
    elif variant == "counterclaim":
        images = [images[1], images[2], images[0]]
    command.extend(["--image", ",".join(str(image) for image in images)])
    prompt = PROMPT_CORE.replace(
        "{variant_instruction}", PROMPT_VARIANTS[variant]
    )
    command.extend(["-p", prompt])
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    elapsed = time.time() - started
    parsed = extract_json(proc.stdout)
    validate(parsed)
    return {
        "asset_id": asset_id,
        "model": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "quantization": "Q4_K_M",
        "vision_projector": "Q8_0",
        "runtime": "llama.cpp b10103, Vulkan GPU offload requested",
        "temperature": 0,
        "prompt_variant": variant,
        "image_views": [str(path.relative_to(ROOT)).replace("\\", "/") for path in images],
        "image_sha256": {path.name: sha256(path) for path in images},
        "elapsed_seconds": round(elapsed, 3),
        "assessment": parsed,
        "raw_stdout": proc.stdout,
        "runtime_stderr": proc.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=[*ASSETS, "all"], default="all")
    parser.add_argument(
        "--variant", choices=[*PROMPT_VARIANTS, "all"], default="all"
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = ASSETS if args.asset == "all" else {args.asset: ASSETS[args.asset]}
    variants = (
        PROMPT_VARIANTS if args.variant == "all"
        else {args.variant: PROMPT_VARIANTS[args.variant]}
    )
    results = []
    for asset_id, render_dir in selected.items():
        for variant in variants:
            result = run_asset(asset_id, render_dir, variant)
            results.append(result)
            (OUT / f"{asset_id}_{variant}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    manifest = {
        "model_sha256": sha256(MODEL),
        "mmproj_sha256": sha256(MMPROJ),
        "asset_count": len(selected),
        "assessment_count": len(results),
        "prompt_variants": list(variants),
        "claim_boundary": (
            "AI-assisted visual audit only; no expert substitution, community "
            "endorsement, or cultural-authenticity certification."
        ),
        "results": [
            {k: r[k] for k in ("asset_id", "prompt_variant", "elapsed_seconds")}
            for r in results
        ],
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
