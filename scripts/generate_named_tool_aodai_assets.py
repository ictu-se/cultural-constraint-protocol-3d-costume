"""Generate inspectable ao-dai proxy assets with a named, versioned tool.

Run with Blender, not the system Python:
  blender --background --python scripts/generate_named_tool_aodai_assets.py

The generator is adapted from the code-native visual proxy in the sibling
AoDai3D project. Assets are procedural research proxies, not expert-validated
reconstructions and not cultural ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LOCAL_SOURCE = Path(__file__).resolve().parent / "aodai_proxy_generator_source.py"
SOURCE_SCRIPT = LOCAL_SOURCE if LOCAL_SOURCE.exists() else (
    WORKSPACE
    / "01_aodai3d_dataset"
    / "scripts"
    / "render_blender_aodai_assets_v2.py"
)
OUT = ROOT / "experiments" / "exp05_named_tool_aodai_assets"
ASSETS = OUT / "assets"
RENDERS = OUT / "renders"
ASSETS.mkdir(parents=True, exist_ok=True)
RENDERS.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("aodai_proxy_generator", SOURCE_SCRIPT)
generator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generator)

VARIANTS = [
    {
        "asset_id": "BL_AODAI_001",
        "label": "baseline_teal_long_sleeve",
        "color": "teal",
        "length": 2.30,
        "hem_scale": 1.00,
        "waist_scale": 1.00,
        "long_sleeve": True,
        "slit": 1.10,
        "intended_test": "baseline procedural proxy",
    },
    {
        "asset_id": "BL_AODAI_002",
        "label": "crimson_long_panel",
        "color": "crimson",
        "length": 2.42,
        "hem_scale": 1.05,
        "waist_scale": 0.95,
        "long_sleeve": True,
        "slit": 1.15,
        "intended_test": "panel-length and colour variation",
    },
    {
        "asset_id": "BL_AODAI_003",
        "label": "indigo_flared_hem",
        "color": "indigo",
        "length": 2.35,
        "hem_scale": 1.35,
        "waist_scale": 0.92,
        "long_sleeve": True,
        "slit": 1.00,
        "intended_test": "silhouette perturbation",
    },
    {
        "asset_id": "BL_AODAI_004",
        "label": "ivory_short_sleeve",
        "color": "ivory",
        "length": 2.22,
        "hem_scale": 1.00,
        "waist_scale": 1.12,
        "long_sleeve": False,
        "slit": 1.05,
        "intended_test": "component and sleeve-length perturbation",
    },
    {
        "asset_id": "BL_AODAI_005",
        "label": "teal_slim_high_slit",
        "color": "teal",
        "length": 2.28,
        "hem_scale": 0.78,
        "waist_scale": 0.72,
        "long_sleeve": True,
        "slit": 1.22,
        "intended_test": "proportion and slit-height perturbation",
    },
]

EXCLUDED_PREFIXES = (
    "mannequin",
    "smooth mannequin",
    "studio floor",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def garment_mesh_objects():
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and not obj.name.lower().startswith(EXCLUDED_PREFIXES)
    ]


def select_only(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def render_views(asset_id: str):
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    for camera in [obj for obj in list(scene.objects) if obj.type == "CAMERA"]:
        bpy.data.objects.remove(camera, do_unlink=True)
    for view, azimuth in [
        ("front", 0),
        ("three_quarter", -42),
        ("side", -90),
        ("back", 180),
    ]:
        for camera in [obj for obj in list(scene.objects) if obj.type == "CAMERA"]:
            bpy.data.objects.remove(camera, do_unlink=True)
        generator.add_camera(azim=azimuth, elev=8, ortho=3.2)
        scene.render.filepath = str(RENDERS / f"{asset_id}_{view}.png")
        bpy.ops.render.render(write_still=True)


def mesh_counts(objects):
    vertices = sum(len(obj.data.vertices) for obj in objects)
    polygons = sum(len(obj.data.polygons) for obj in objects)
    return vertices, polygons


def main():
    rows = []
    for variant in VARIANTS:
        generator.create_garment(
            variant["color"],
            variant["length"],
            variant["hem_scale"],
            variant["waist_scale"],
            variant["long_sleeve"],
            variant["slit"],
        )
        objects = garment_mesh_objects()
        vertices, polygons = mesh_counts(objects)
        render_views(variant["asset_id"])
        select_only(objects)

        glb_path = ASSETS / f"{variant['asset_id']}.glb"
        obj_path = ASSETS / f"{variant['asset_id']}.obj"
        blend_path = ASSETS / f"{variant['asset_id']}.blend"
        bpy.ops.export_scene.gltf(
            filepath=str(glb_path),
            export_format="GLB",
            use_selection=True,
            export_apply=True,
        )
        bpy.ops.wm.obj_export(
            filepath=str(obj_path),
            export_selected_objects=True,
            apply_modifiers=True,
            export_materials=True,
        )
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

        row = dict(variant)
        row.update(
            {
                "tool": "Blender",
                "tool_version": bpy.app.version_string,
                "pipeline": "procedural Blender Python generation",
                "source_script": (
                    str(SOURCE_SCRIPT.relative_to(ROOT)).replace("\\", "/")
                    if ROOT in SOURCE_SCRIPT.parents
                    else str(SOURCE_SCRIPT.relative_to(WORKSPACE)).replace("\\", "/")
                ),
                "generation_command": "blender --background --python scripts/generate_named_tool_aodai_assets.py",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "vertices_pre_modifier": vertices,
                "polygons_pre_modifier": polygons,
                "glb_relpath": str(glb_path.relative_to(ROOT)).replace("\\", "/"),
                "glb_sha256": sha256(glb_path),
                "obj_relpath": str(obj_path.relative_to(ROOT)).replace("\\", "/"),
                "obj_sha256": sha256(obj_path),
                "blend_relpath": str(blend_path.relative_to(ROOT)).replace("\\", "/"),
                "blend_sha256": sha256(blend_path),
                "license": "study-generated code output; source script retained",
                "claim_status": "procedural proxy; not expert-validated cultural ground truth",
            }
        )
        rows.append(row)

    with (OUT / "asset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "run_summary.json").write_text(
        json.dumps(
            {
                "asset_count": len(rows),
                "tool": "Blender",
                "tool_version": bpy.app.version_string,
                "source_script": str(SOURCE_SCRIPT),
                "claim_boundary": (
                    "Actual inspectable 3D assets generated by a named tool. "
                    "They are procedural research proxies, not outputs of a learned "
                    "model and not expert-validated Vietnamese cultural ground truth."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
