"""Independent mesh-geometry measurements for controlled áo dài proxy assets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "experiments/exp05_named_tool_aodai_assets/assets"
OUT = ROOT / "experiments/exp13_mesh_geometry_detector"
BASELINE = "BL_AODAI_001"
CASES = {
    "BL_AODAI_002": ("longer front/back panels", "silhouette"),
    "BL_AODAI_003": ("flared hem", "silhouette"),
    "BL_AODAI_004": ("short sleeves", "components"),
    "BL_AODAI_005": ("slimmer body and higher side slit", "silhouette"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = [
        geometry for geometry in loaded.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    ]
    if not meshes:
        raise ValueError(f"No triangle geometry in {path}")
    return trimesh.util.concatenate(meshes)


def measurements(asset_id: str) -> dict:
    path = ASSETS / f"{asset_id}.glb"
    mesh = load_mesh(path)
    components = mesh.split(only_watertight=False)
    extents = np.asarray(mesh.extents, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    return {
        "asset_id": asset_id,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "connected_components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "extent_x": float(extents[0]),
        "extent_y": float(extents[1]),
        "extent_z": float(extents[2]),
        "min_x": float(bounds[0, 0]),
        "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]),
        "max_x": float(bounds[1, 0]),
        "max_y": float(bounds[1, 1]),
        "max_z": float(bounds[1, 2]),
        "surface_area": float(mesh.area),
        "signed_volume": float(mesh.volume),
    }


def relative_delta(value: float, baseline: float) -> float:
    return (value - baseline) / abs(baseline) if baseline else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [measurements(BASELINE)] + [
        measurements(asset_id) for asset_id in CASES
    ]
    baseline = rows[0]
    numeric = [
        "vertices", "faces", "connected_components", "euler_number",
        "extent_x", "extent_y", "extent_z", "min_x", "min_y", "min_z",
        "max_x", "max_y", "max_z", "surface_area", "signed_volume",
    ]
    deltas = []
    for row in rows[1:]:
        intervention, target = CASES[row["asset_id"]]
        item = {
            "asset_id": row["asset_id"],
            "baseline_id": BASELINE,
            "intervention": intervention,
            "programmatic_target_criterion": target,
        }
        for field in numeric:
            item[f"{field}_delta"] = row[field] - baseline[field]
            item[f"{field}_relative_delta"] = relative_delta(
                row[field], baseline[field]
            )
        extents_changed = any(
            abs(item[f"extent_{axis}_relative_delta"]) > 0.005
            for axis in "xyz"
        )
        topology_changed = any(
            item[f"{field}_delta"] != 0
            for field in (
                "vertices", "faces", "connected_components", "euler_number"
            )
        )
        item["detected_geometry_change"] = (
            extents_changed
            or topology_changed
            or abs(item["surface_area_relative_delta"]) > 0.005
            or abs(item["signed_volume_relative_delta"]) > 0.005
        )
        deltas.append(item)

    with (OUT / "mesh_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "baseline_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=deltas[0].keys())
        writer.writeheader()
        writer.writerows(deltas)
    summary = {
        "experiment": "independent exported-mesh geometry detector",
        "baseline": BASELINE,
        "case_count": len(deltas),
        "detected_case_count": sum(
            row["detected_geometry_change"] for row in deltas
        ),
        "all_controlled_changes_detected": all(
            row["detected_geometry_change"] for row in deltas
        ),
        "measurements": rows,
        "deltas": deltas,
        "claim_boundary": (
            "Geometry and topology instrumentation only; criterion attribution "
            "comes from programmatic intervention metadata, not cultural review."
        ),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "case_count": len(deltas),
        "detected_case_count": summary["detected_case_count"],
        "all_controlled_changes_detected":
            summary["all_controlled_changes_detected"],
    }, indent=2))


if __name__ == "__main__":
    main()
