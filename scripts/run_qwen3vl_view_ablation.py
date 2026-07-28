"""Measure Qwen3-VL audit sensitivity to the number of rendered views."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from run_qwen3vl_audit import (
    ASSETS, CLI, MMPROJ, MODEL, OUT as BASE_OUT, PROMPT_CORE,
    PROMPT_VARIANTS, ROOT, extract_json, validate,
)


OUT = ROOT / "experiments/exp09_qwen3vl_view_ablation"
VIEW_CONFIGS = {
    "front": ("front",),
    "front_side": ("front", "side"),
    "front_side_back": ("front", "side", "back"),
    "four_view": ("front", "three_quarter", "side", "back"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(asset_id: str, render_dir: Path, config: str, views: tuple[str, ...]) -> dict:
    images = [render_dir / f"{asset_id}_{view}.png" for view in views]
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    prompt = PROMPT_CORE.replace(
        "{variant_instruction}",
        PROMPT_VARIANTS["direct"]
        + " The number of supplied views varies by experimental condition; "
        "do not assume that unseen views agree with visible views.",
    )
    started = time.time()
    last_error = None
    for attempt in range(3):
        attempt_prompt = prompt + (
            "" if attempt == 0 else
            "\nYour prior response failed schema validation. Include every "
            "required key exactly once and output no text outside the JSON."
        )
        command = [
            str(CLI), "-m", str(MODEL), "--mmproj", str(MMPROJ),
            "-ngl", "99", "--ctx-size", "12288", "--image-min-tokens", "1024",
            "--image-max-tokens", "1024", "--temp", "0", "-n", "1400",
            "--image", ",".join(str(path) for path in images), "-p", attempt_prompt,
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        try:
            assessment = extract_json(proc.stdout)
            validate(assessment)
            break
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
    else:
        raise ValueError(f"Schema validation failed after retries: {last_error}")
    return {
        "asset_id": asset_id,
        "view_config": config,
        "views": list(views),
        "model": "Qwen/Qwen3-VL-8B-Instruct-GGUF Q4_K_M",
        "temperature": 0,
        "schema_attempts": attempt + 1,
        "elapsed_seconds": round(time.time() - started, 3),
        "image_sha256": {path.name: sha256(path) for path in images},
        "assessment": assessment,
        "raw_stdout": proc.stdout,
        "runtime_stderr": proc.stderr,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for asset_id, render_dir in ASSETS.items():
        for config, views in VIEW_CONFIGS.items():
            output_path = OUT / f"{asset_id}_{config}.json"
            if output_path.exists():
                try:
                    record = json.loads(output_path.read_text(encoding="utf-8"))
                    validate(record["assessment"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    record = run(asset_id, render_dir, config, views)
            else:
                record = run(asset_id, render_dir, config, views)
            results.append(record)
            output_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    (OUT / "run_manifest.json").write_text(
        json.dumps(
            {
                "asset_count": len(ASSETS),
                "view_configurations": VIEW_CONFIGS,
                "assessment_count": len(results),
                "model_sha256": sha256(MODEL),
                "mmproj_sha256": sha256(MMPROJ),
                "claim_boundary": "View-sensitivity test; not cultural validation.",
                "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results), 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
