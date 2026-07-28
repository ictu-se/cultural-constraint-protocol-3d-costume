"""Render standardized views of TripoSR OBJ outputs with Blender 5.1."""

from pathlib import Path
import math

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "exp06_triposr_reconstruction_ultra"
RENDERS = EXP / "renders"
RENDERS.mkdir(parents=True, exist_ok=True)
ASSETS = [("TSR_AODAI_001", EXP / "0" / "mesh.obj"), ("TSR_AODAI_002", EXP / "1" / "mesh.obj")]


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def material(texture_path=None):
    mat = bpy.data.materials.new("TripoSR baked reconstruction material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.52
    if texture_path and texture_path.exists():
        image = bpy.data.images.load(str(texture_path))
        node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.62, 0.68, 0.76, 1.0)
    return mat


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_scene():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.device = "GPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type in {"OPTIX", "CUDA"}
    except Exception as exc:
        print(f"Cycles GPU setup fallback: {exc}")
    scene.render.resolution_x = 3200
    scene.render.resolution_y = 4000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.94, 0.94, 0.94)
    bpy.ops.object.light_add(type="AREA", location=(3.5, -4.5, 5.0))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 4.0
    bpy.ops.object.light_add(type="AREA", location=(-3.0, 1.5, 3.0))
    bpy.context.object.data.energy = 450
    bpy.context.object.data.size = 3.0


def normalize_mesh(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    extent = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
    obj.scale *= 2.8 / max(extent, 1e-8)
    bpy.context.view_layer.update()
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(points, Vector()) / 8
    obj.location -= center
    obj.location.z += 1.45


def render_asset(asset_id, path):
    clear()
    setup_scene()
    bpy.ops.wm.obj_import(filepath=str(path))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {path}")
    obj = meshes[0]
    obj.name = asset_id
    obj.rotation_euler.y = math.radians(-90)
    bpy.context.view_layer.update()
    normalize_mesh(obj)
    obj.data.materials.clear()
    obj.data.materials.append(material(path.parent / "texture.png"))
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.object
    floor.data.materials.append(material())
    for name, azimuth in [("front", 0), ("three_quarter", -42), ("side", -90), ("back", 180)]:
        for camera in [x for x in list(bpy.context.scene.objects) if x.type == "CAMERA"]:
            bpy.data.objects.remove(camera, do_unlink=True)
        a = math.radians(azimuth)
        bpy.ops.object.camera_add(location=(5 * math.sin(a), -5 * math.cos(a), 2.0))
        cam = bpy.context.object
        look_at(cam, (0, 0, 1.45))
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = 3.4
        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = str(RENDERS / f"{asset_id}_{name}.png")
        bpy.ops.render.render(write_still=True)


for asset_id, path in ASSETS:
    render_asset(asset_id, path)
