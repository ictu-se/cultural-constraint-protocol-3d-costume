"""Write a provenance manifest for the completed TripoSR reconstruction run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import trimesh


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EXP = ROOT / "experiments" / "exp06_triposr_reconstruction_ultra"
LOCAL_METADATA = EXP / "source_metadata.csv"
SOURCE_METADATA = LOCAL_METADATA if LOCAL_METADATA.exists() else (
    WORKSPACE
    / "01_aodai3d_dataset"
    / "raw"
    / "collected_images"
    / "wikimedia_commons_aodai"
    / "metadata.csv"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


metadata = list(csv.DictReader(SOURCE_METADATA.open(encoding="utf-8-sig")))
by_page = {row["page_id"]: row for row in metadata}
items = [
    {"asset_id": "TSR_AODAI_001", "folder": "0",
     "label": "modern boys ao-dai documentary input",
     "source": by_page["125183573"]},
    {"asset_id": "TSR_AODAI_002", "folder": "1",
     "label": "controlled Blender ao-dai proxy input",
     "source": {
         "title": "Study-generated Blender ao-dai proxy hero render",
         "artist": "Authors",
         "url": "../01_aodai3d_dataset/manuscript/figures/blender_v2_raw/hero.png",
         "license_short_name": "study-generated research output",
         "usage_terms": "Reproducibility material; not cultural ground truth",
         "attribution_required": "No"}},
]
rows = []
for item in items:
    asset_id, folder, label, source = (
        item["asset_id"], item["folder"], item["label"], item["source"]
    )
    mesh_path = EXP / folder / "mesh.obj"
    input_path = EXP / folder / "input.png"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    rows.append(
        {
            "asset_id": asset_id,
            "label": label,
            "tool": "TripoSR",
            "tool_commit": "107cefdc244c39106fa830359024f6a2f1c78871",
            "model_id": "stabilityai/TripoSR",
            "pipeline": "single-image 3D reconstruction",
            "mc_resolution": 512,
            "texture_resolution": 4096,
            "render_resolution": "3200x4000",
            "render_engine": "Blender 5.1.2 Cycles, 128 samples, denoising, OptiX requested",
            "source_title": source["title"],
            "source_creator": source["artist"],
            "source_url": source["url"],
            "source_license": source["license_short_name"],
            "source_usage_terms": source["usage_terms"],
            "attribution_required": source["attribution_required"],
            "input_relpath": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "archive_input_relpath": f"results/exp06/{folder}/input.png",
            "input_sha256": digest(input_path),
            "mesh_relpath": str(mesh_path.relative_to(ROOT)).replace("\\", "/"),
            "archive_mesh_relpath": f"results/exp06/{folder}/mesh.obj",
            "archive_texture_relpath": f"results/exp06/{folder}/texture.png",
            "mesh_sha256": digest(mesh_path),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "generation_command": (
                "python run.py <documentary image> <controlled Blender render> "
                "--output-dir experiments/exp06_triposr_reconstruction_ultra "
                "--model-save-format obj --mc-resolution 512 "
                "--foreground-ratio 0.9 --bake-texture "
                "--texture-resolution 4096 --chunk-size 2048"
            ),
            "implementation_note": (
                "Official TripoSR commit with documented scikit-image CPU "
                "marching-cubes fallback because torchmcubes could not compile "
                "without Windows C++ build tools; texture-query tensors were "
                "placed explicitly on the model device for CUDA baking."
            ),
            "claim_status": (
                "actual named-tool reconstruction; output includes the visible "
                "person/scene silhouette and is not a garment-only cultural ground truth"
            ),
        }
    )

with (EXP / "asset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(EXP / "run_summary.json").write_text(
    json.dumps(
        {
            "asset_count": len(rows),
            "tool": "TripoSR",
            "tool_commit": rows[0]["tool_commit"],
            "model_id": rows[0]["model_id"],
            "source_licenses": sorted({row["source_license"] for row in rows}),
            "claim_boundary": (
                "Single-image reconstructions test whether the audit exposes "
                "missing back-view, pattern, construction, and contextual evidence. "
                "They are not expert-validated Vietnamese costume reconstructions."
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)
