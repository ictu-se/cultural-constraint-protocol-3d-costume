import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "figures" / "blender_v2_raw"
OUT.mkdir(parents=True, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def mat_principled(name, color, rough=0.35, metallic=0.0, alpha=1.0, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Alpha"].default_value = alpha
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = sheen
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.18
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = 0.22
    m.diffuse_color = color
    if alpha < 1.0:
        m.blend_method = "BLEND"
        m.use_screen_refraction = True
    return m


def setup_scene(res=(1500, 1900)):
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("World")
    scene.world.color = (1.0, 1.0, 1.0)

    # Studio floor/backdrop.
    floor_mat = mat_principled("matte warm grey studio", (0.82, 0.84, 0.83, 1), rough=0.78)
    bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0, 0, -0.02))
    floor = bpy.context.object
    floor.name = "studio floor"
    floor.data.materials.append(floor_mat)

    bpy.ops.object.light_add(type="AREA", location=(0, -4.0, 4.6))
    key = bpy.context.object
    key.name = "large rectangular softbox"
    key.data.energy = 650
    key.data.size = 5.2
    bpy.ops.object.light_add(type="AREA", location=(-3.2, -1.0, 2.8))
    fill = bpy.context.object
    fill.name = "left fill"
    fill.data.energy = 130
    fill.data.size = 4.0
    bpy.ops.object.light_add(type="AREA", location=(2.7, 2.1, 3.2))
    rim = bpy.context.object
    rim.name = "rim softbox"
    rim.data.energy = 210
    rim.data.size = 3.2


MAT_SKIN = None
MAT_SILK = None
MAT_SILK_DARK = None
MAT_TROUSER = None
MAT_GOLD = None
MAT_SEAM = None
MAT_PATTERN = None


def init_materials(color_variant="teal"):
    global MAT_SKIN, MAT_SILK, MAT_SILK_DARK, MAT_TROUSER, MAT_GOLD, MAT_SEAM, MAT_PATTERN
    palettes = {
        "teal": ((0.02, 0.42, 0.46, 1), (0.00, 0.20, 0.22, 1)),
        "crimson": ((0.62, 0.05, 0.12, 1), (0.24, 0.015, 0.04, 1)),
        "indigo": ((0.08, 0.15, 0.48, 1), (0.03, 0.05, 0.18, 1)),
        "ivory": ((0.85, 0.78, 0.58, 1), (0.42, 0.32, 0.18, 1)),
    }
    main, dark = palettes[color_variant]
    MAT_SKIN = mat_principled("warm mannequin", (0.78, 0.67, 0.58, 1), rough=0.62)
    MAT_SILK = mat_principled("procedural silk " + color_variant, main, rough=0.24, sheen=0.85)
    MAT_SILK_DARK = mat_principled("deep silk " + color_variant, dark, rough=0.30, sheen=0.72)
    MAT_TROUSER = mat_principled("silk ivory trouser", (0.90, 0.91, 0.86, 1), rough=0.44, sheen=0.35)
    MAT_GOLD = mat_principled("embroidered gold", (1.0, 0.68, 0.22, 1), rough=0.32, metallic=0.25)
    MAT_SEAM = mat_principled("slit piping", (0.95, 0.14, 0.23, 1), rough=0.42)
    MAT_PATTERN = mat_principled("pattern craft paper", (0.96, 0.80, 0.35, 1), rough=0.72)


def shade(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)


def add_camera(azim=-35, elev=9, dist=6.0, ortho=3.45):
    theta = math.radians(azim)
    phi = math.radians(elev)
    target = Vector((0, 0, 1.35))
    loc = Vector((dist * math.sin(theta) * math.cos(phi), -dist * math.cos(theta) * math.cos(phi), dist * math.sin(phi) + 1.45))
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    direction = target - loc
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho
    cam.data.lens = 80
    bpy.context.scene.camera = cam
    return cam


def add_mannequin():
    # Better mannequin proportions: body is mostly hidden by garment.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.15, location=(0, -0.005, 2.67))
    head = bpy.context.object
    head.name = "smooth mannequin head"
    head.scale = (0.86, 0.78, 1.05)
    head.data.materials.append(MAT_SKIN)
    shade(head)

    for name, loc, scale in [
        ("torso", (0, 0, 1.74), (0.27, 0.18, 0.70)),
        ("hip", (0, 0, 1.05), (0.32, 0.18, 0.34)),
    ]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=20, radius=1, location=loc)
        obj = bpy.context.object
        obj.name = "mannequin " + name
        obj.scale = scale
        obj.data.materials.append(MAT_SKIN)
        shade(obj)


def add_tube_between(name, p0, p1, radius, mat, bevel=12):
    mid = (Vector(p0) + Vector(p1)) / 2
    direction = Vector(p1) - Vector(p0)
    length = direction.length
    bpy.ops.mesh.primitive_cylinder_add(vertices=bevel, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    shade(obj)
    return obj


def body_radius(z, hem_scale=1.0, waist_scale=1.0):
    # z: 0.08 hem to 2.35 shoulder
    v = max(0, min(1, (2.35 - z) / 2.27))
    shoulder = 0.27 * math.exp(-((z - 2.18) ** 2) / 0.08)
    waist = 0.20 + 0.07 * waist_scale * math.exp(-((z - 1.42) ** 2) / 0.20)
    hem = 0.28 * hem_scale * (v ** 1.55)
    return waist + shoulder + hem


def make_panel(name, side="front", length=2.30, hem_scale=1.0, waist_scale=1.0, slit=1.10):
    rows, cols = 64, 34
    verts = []
    faces = []
    is_front = side == "front"
    center_angle = -math.pi / 2 if is_front else math.pi / 2
    width_angle = math.radians(128 if is_front else 116)
    z_top = 2.35
    z_bottom = max(0.06, z_top - length)
    for i in range(rows):
        t = i / (rows - 1)
        z = z_top * (1 - t) + z_bottom * t
        r = body_radius(z, hem_scale, waist_scale)
        # Soft wind/drape folds.
        fold = 0.028 * math.sin(18 * t) + 0.016 * math.sin(7 * t + (0 if is_front else 1.2))
        for j in range(cols):
            u = (j / (cols - 1) - 0.5) * 2
            angle = center_angle + u * width_angle / 2
            # Side opening: below slit, panels separate more visibly near edges.
            edge_open = 0.10 * max(0, (slit - z) / max(slit, 0.01)) * (abs(u) ** 2.8)
            rr = r + fold + edge_open
            x = rr * math.cos(angle)
            y = rr * math.sin(angle)
            # Slight asymmetric natural cloth waviness.
            x += 0.018 * math.sin(10 * t + 2 * u)
            y += 0.012 * math.cos(8 * t + 3 * u)
            verts.append((x, y, z))
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = i * cols + j
            faces.append((a, a + 1, a + cols + 1, a + cols))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(MAT_SILK if is_front else MAT_SILK_DARK)
    obj.modifiers.new("cloth subdivision", "SUBSURF").levels = 1
    solid = obj.modifiers.new("thin fabric thickness", "SOLIDIFY")
    solid.thickness = 0.009
    solid.offset = 0
    shade(obj)
    return obj


def add_collar():
    # High collar as a short open cylinder.
    rows, cols = 8, 48
    verts, faces = [], []
    for i in range(rows):
        z = 2.30 + i / (rows - 1) * 0.20
        for j in range(cols):
            angle = 2 * math.pi * j / cols
            r_x, r_y = 0.22, 0.16
            verts.append((r_x * math.cos(angle), r_y * math.sin(angle), z))
    for i in range(rows - 1):
        for j in range(cols):
            faces.append((i * cols + j, i * cols + (j + 1) % cols, (i + 1) * cols + (j + 1) % cols, (i + 1) * cols + j))
    mesh = bpy.data.meshes.new("collarMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("structured high collar", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(MAT_SILK)
    obj.modifiers.new("collar thickness", "SOLIDIFY").thickness = 0.012
    shade(obj)


def add_sleeves(long=True):
    z_end = 1.12 if long else 1.68
    for side in (-1, 1):
        p0 = (side * 0.27, -0.01, 2.20)
        p1 = (side * 0.65, -0.08, z_end)
        obj = add_tube_between("silk sleeve", p0, p1, 0.073, MAT_SILK)
        obj.scale.x *= 0.75


def add_trousers():
    for side in (-1, 1):
        add_tube_between("ivory trouser leg", (side * 0.11, 0.0, 1.05), (side * 0.13, 0.0, 0.05), 0.075, MAT_TROUSER, bevel=18)


def add_piping_and_embroidery(slit=1.10):
    # Side slit piping.
    for side in (-1, 1):
        for angle in (-math.radians(25), math.radians(25)):
            x = side * (0.30 + 0.12 * abs(math.sin(angle)))
            add_tube_between("side slit piping", (x, -0.23, slit), (x + side * 0.08, -0.24, 0.22), 0.010, MAT_SEAM, bevel=10)
    # Gold embroidered floral chain on front panel.
    for k, z in enumerate([2.05, 1.72, 1.38, 1.03]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.035, location=(0.0, -0.37, z))
        c = bpy.context.object
        c.name = "gold blossom"
        c.scale = (1.0, 0.20, 1.0)
        c.data.materials.append(MAT_GOLD)
        shade(c)
        for side in (-1, 1):
            add_tube_between("gold leaf stitch", (0.02 * side, -0.37, z), (0.12 * side, -0.36, z + 0.06), 0.006, MAT_GOLD, bevel=8)


def create_garment(color="teal", length=2.30, hem_scale=1.0, waist_scale=1.0, long_sleeve=True, slit=1.10):
    setup_scene()
    init_materials(color)
    add_mannequin()
    make_panel("front ao dai panel", "front", length, hem_scale, waist_scale, slit)
    make_panel("back ao dai panel", "back", length * 0.98, hem_scale * 0.95, waist_scale, slit)
    add_collar()
    add_sleeves(long_sleeve)
    add_trousers()
    add_piping_and_embroidery(slit)


def render(path, azim=-35, elev=9, ortho=3.25):
    add_camera(azim=azim, elev=elev, ortho=ortho)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_all():
    create_garment("teal")
    render(OUT / "hero.png", -35, 9, 3.1)
    for name, azim in [("front", 0), ("three_quarter", -42), ("side", -90), ("back", 180)]:
        create_garment("teal")
        render(OUT / f"view_{name}.png", azim, 8, 3.2)
    variants = [
        ("crimson_long", "crimson", 2.42, 1.05, 0.95, True, 1.15),
        ("indigo_flare", "indigo", 2.35, 1.35, 0.92, True, 1.00),
        ("ivory_short_sleeve", "ivory", 2.22, 1.00, 1.12, False, 1.05),
        ("teal_slim", "teal", 2.28, 0.78, 0.72, True, 1.22),
    ]
    for name, color, length, hem, waist, sleeve, slit in variants:
        create_garment(color, length, hem, waist, sleeve, slit)
        render(OUT / f"variant_{name}.png", -38, 8, 3.18)
    # Landmark overlay.
    create_garment("teal")
    lm = mat_principled("bright landmark", (1.0, 0.84, 0.08, 1), rough=0.25)
    for loc in [(0, -0.38, 2.34), (-0.27, -0.30, 2.18), (-0.64, -0.18, 1.12), (-0.27, -0.35, 1.46), (-0.40, -0.35, 1.10), (-0.50, -0.33, 0.12)]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.045, location=loc)
        obj = bpy.context.object
        obj.data.materials.append(lm)
        shade(obj)
    render(OUT / "landmarks.png", -35, 8, 3.15)
    # Pattern layout as separate top-down scene.
    render_pattern_scene()


def render_pattern_scene():
    setup_scene(res=(1500, 1200))
    init_materials("teal")
    panels = [
        ("front panel", [(-1.0, -0.04, 0.0), (-0.25, -0.04, 0.0), (-0.10, -0.04, 2.35), (-1.14, -0.04, 2.35)], MAT_PATTERN),
        ("back panel", [(0.02, -0.04, 0.0), (0.77, -0.04, 0.0), (0.92, -0.04, 2.35), (-0.12, -0.04, 2.35)], MAT_PATTERN),
        ("sleeve", [(-0.90, -0.04, 2.62), (0.35, -0.04, 2.62), (0.50, -0.04, 2.95), (-1.05, -0.04, 2.95)], MAT_SILK),
        ("collar", [(0.58, -0.04, 2.65), (1.12, -0.04, 2.65), (1.12, -0.04, 2.84), (0.58, -0.04, 2.84)], MAT_GOLD),
    ]
    for _, corners, mat in panels:
        mesh = bpy.data.meshes.new("panel")
        mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
        mesh.update()
        obj = bpy.data.objects.new("pattern panel", mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)
        obj.modifiers.new("paper solidify", "SOLIDIFY").thickness = 0.012
    for x in (-0.08, 0.18, 0.48):
        add_tube_between("stitch relation", (x, -0.07, 2.42), (x + 0.16, -0.07, 2.58), 0.010, MAT_SEAM, bevel=8)
    add_camera(azim=0, elev=72, dist=4.5, ortho=3.6)
    bpy.context.scene.render.filepath = str(OUT / "pattern_layout.png")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    render_all()
