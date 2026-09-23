"""Build the three pieces of hardware the simulation shows, in Blender, and export them as glTF:

    cap.glb    a head wearing a 32 channel EEG cap, electrodes and cabling bundled into a harness
    pcb.glb    the decoder board: substrate, gold edge connector, the readout chip, passives, ports
    scope.glb  a bench oscilloscope; the mesh named "Screen" is swapped for a live canvas in the page

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b -P fly-eeg/blender_hardware.py

Everything is modelled to a 1.0 = 1 metre scale and re-scaled in the viewer. Materials are
Principled BSDF so the glTF carries metallic / roughness straight into three.js.
"""
import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

OUT = Path(__file__).resolve().parent / "sim" / "models"


# ---------------------------------------------------------------- helpers ----
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, colour, metallic=0.0, rough=0.5, emit=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if emit and "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = 1.0
    return m


def put(obj, material, shade_smooth=False, name=None):
    obj.data.materials.append(material)
    if shade_smooth:
        for p in obj.data.polygons:
            p.use_smooth = True
    if name:
        obj.name = name
    return obj


def cube(_size, loc=(0, 0, 0), scale=(1, 1, 1), bevel=0.0, segments=2):
    bpy.ops.mesh.primitive_cube_add(size=2, location=loc)   # size 2 => `scale` is the half-extent
    o = bpy.context.object
    o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        m = o.modifiers.new("b", "BEVEL")
        m.width, m.segments, m.limit_method = bevel, segments, "ANGLE"
        bpy.ops.object.modifier_apply(modifier=m.name)
    return o


def cyl(r, d, loc=(0, 0, 0), rot=(0, 0, 0), verts=48, bevel=0.0):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d, location=loc, rotation=rot, vertices=verts)
    o = bpy.context.object
    if bevel:
        m = o.modifiers.new("b", "BEVEL")
        m.width, m.segments, m.limit_method = bevel, 2, "ANGLE"
        bpy.ops.object.modifier_apply(modifier=m.name)
    return o


def join(objs, name):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    return o


def tube(points, radius, res=12):
    """A smooth cable through `points`, as real geometry."""
    cu = bpy.data.curves.new("cable", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = res
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(points) - 1)
    for bp, p in zip(sp.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    o = bpy.data.objects.new("cable", cu)
    bpy.context.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


def export(name):
    bpy.ops.object.select_all(action="SELECT")
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(OUT / f"{name}.glb"), export_format="GLB",
                              use_selection=True, export_apply=True, export_yup=True)
    print("wrote", OUT / f"{name}.glb", flush=True)


# ------------------------------------------------------------------- cap ----
def head_shape():
    """A mannequin head: a sphere pushed into a cranium, jaw and neck."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.098, segments=64, ring_count=48)
    h = bpy.context.object
    bm = bmesh.new()
    bm.from_mesh(h.data)
    for v in bm.verts:
        x, y, z = v.co
        v.co.x = x * 0.80                                   # narrower than deep
        v.co.y = y * 1.02
        if z < 0:                                           # jaw tapers forward and in
            t = -z / 0.098
            v.co.x *= 1 - 0.34 * t * t
            v.co.y *= 1 - 0.16 * t * t
            v.co.y += 0.016 * t * t
            v.co.z = z * 1.22
        else:                                               # cranium: flatten the back
            if y < 0:
                v.co.y = y * (1 - 0.16 * (z / 0.098))
    bm.to_mesh(h.data)
    bm.free()
    sub = h.modifiers.new("s", "SUBSURF")
    sub.levels = sub.render_levels = 1
    bpy.ops.object.modifier_apply(modifier=sub.name)
    neck = cyl(0.040, 0.10, loc=(0, 0.004, -0.128), verts=40)
    return join([h, neck], "Head")


def eeg_cap():
    reset()
    skin = mat("skin", (0.78, 0.63, 0.55), 0, 0.62)
    fabric = mat("fabric", (0.055, 0.075, 0.12), 0, 0.92)
    strap = mat("strap", (0.10, 0.12, 0.17), 0, 0.85)
    ring = mat("electrode", (0.88, 0.89, 0.91), 1.0, 0.22)
    gel = mat("gel", (0.55, 0.78, 0.95), 0, 0.18)
    wire = mat("wire", (0.12, 0.14, 0.18), 0, 0.42)
    sheath = mat("sheath", (0.20, 0.23, 0.28), 0, 0.55)

    head = put(head_shape(), skin, True)

    # the cap itself: the top of a slightly larger sphere, thickened into fabric
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1035, segments=64, ring_count=48)
    shell = bpy.context.object
    bm = bmesh.new()
    bm.from_mesh(shell.data)
    for v in bm.verts:
        v.co.x *= 0.80
        v.co.y *= 1.02
        if v.co.y < 0 and v.co.z > 0:
            v.co.y *= 1 - 0.16 * (v.co.z / 0.1035)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.026], context="VERTS")
    bm.to_mesh(shell.data)
    bm.free()
    sol = shell.modifiers.new("t", "SOLIDIFY")
    sol.thickness, sol.offset = 0.0035, 1
    bpy.ops.object.modifier_apply(modifier=sol.name)
    sub = shell.modifiers.new("s", "SUBSURF")
    sub.levels = sub.render_levels = 1
    bpy.ops.object.modifier_apply(modifier=sub.name)
    put(shell, fabric, True, "Cap")

    # chin strap
    band = tube([(-0.070, 0.012, -0.030), (-0.052, 0.030, -0.090), (0, 0.040, -0.112),
                 (0.052, 0.030, -0.090), (0.070, 0.012, -0.030)], 0.0045)
    put(band, strap, True)

    # 32 electrodes on a 10-20-like grid, each a ring with gel, each with its own lead
    parts, cables = [], []
    rows = [(0.30, 4), (0.58, 8), (0.90, 9), (1.22, 7), (1.55, 4)]      # polar angle, count
    hub = Vector((0, -0.105, 0.052))
    for theta, n in rows:
        for i in range(n):
            phi = -math.pi / 2 + math.pi * (i + 0.5) / n if n > 1 else 0
            d = Vector((math.sin(theta) * math.sin(phi) * 0.80,
                        -math.sin(theta) * math.cos(phi) * 1.02,
                        math.cos(theta)))
            p = Vector((d.x, d.y, d.z)) * 0.1085
            rot = d.to_track_quat("Z", "Y").to_euler()
            body = cyl(0.0068, 0.0052, loc=p, rot=rot, verts=28, bevel=0.0009)
            put(body, ring, True)
            hole = cyl(0.0030, 0.0062, loc=p + d * 0.0006, rot=rot, verts=20)
            put(hole, gel, True)
            parts += [body, hole]
            mid = p + d * 0.010 + Vector((0, -0.016, 0.004))
            cables.append(tube([p + d * 0.002, mid, hub + Vector((0, 0.014, 0.006)), hub], 0.0010))
    for cbl in cables:
        put(cbl, wire, True)
    # the leads gather into one sheathed harness leaving the back of the head
    harness = tube([hub, hub + Vector((0, -0.05, -0.03)), hub + Vector((0.01, -0.10, -0.12)),
                    hub + Vector((0.02, -0.13, -0.24))], 0.0085)
    put(harness, sheath, True)
    collar = cyl(0.011, 0.016, loc=hub + Vector((0, -0.012, -0.008)), rot=(math.radians(64), 0, 0), verts=28, bevel=0.001)
    put(collar, sheath, True)

    join([head, shell, band, harness, collar] + parts + cables, "EEGCap")
    export("cap")


# ------------------------------------------------------------------- pcb ----
def pcb():
    reset()
    board_m = mat("solder_mask", (0.035, 0.115, 0.075), 0, 0.42)
    gold = mat("gold", (0.86, 0.68, 0.30), 1.0, 0.24)
    silver = mat("tin", (0.80, 0.82, 0.85), 1.0, 0.30)
    black = mat("epoxy", (0.045, 0.048, 0.055), 0, 0.42)
    grey = mat("passive", (0.18, 0.18, 0.20), 0, 0.55)
    tan = mat("cap_body", (0.16, 0.14, 0.30), 0, 0.45)
    white = mat("connector", (0.88, 0.89, 0.90), 0, 0.55)

    W, D, T = 0.105, 0.075, 0.0016
    board = cube(1, scale=(W / 2, D / 2, T / 2), bevel=0.0006)
    put(board, board_m, False, "Board")
    parts = []

    # gold edge connector along the front edge
    for i in range(20):
        x = -W / 2 + 0.010 + i * (W - 0.020) / 19
        p = cube(1, loc=(x, -D / 2 + 0.006, T / 2 + 0.00012), scale=(0.0016, 0.0055, 0.00012))
        put(p, gold)
        parts.append(p)

    # the readout chip, with real gull-wing pins
    body = cube(1, loc=(0.004, 0.010, T / 2 + 0.0016), scale=(0.0135, 0.0135, 0.0016), bevel=0.0004)
    put(body, black, False, "Chip")
    parts.append(body)
    for side in range(4):
        for i in range(14):
            t = -0.0118 + i * 0.00182
            if side == 0:   loc, sc = (0.004 + t, 0.010 + 0.0148, T / 2 + 0.0008), (0.0005, 0.0018, 0.0002)
            elif side == 1: loc, sc = (0.004 + t, 0.010 - 0.0148, T / 2 + 0.0008), (0.0005, 0.0018, 0.0002)
            elif side == 2: loc, sc = (0.004 + 0.0148, 0.010 + t, T / 2 + 0.0008), (0.0018, 0.0005, 0.0002)
            else:           loc, sc = (0.004 - 0.0148, 0.010 + t, T / 2 + 0.0008), (0.0018, 0.0005, 0.0002)
            p = cube(1, loc=loc, scale=sc)
            put(p, silver)
            parts.append(p)

    # passives scattered on the board, an electrolytic can, a crystal, a header and a port
    spots = [(-0.040, 0.026), (-0.033, 0.026), (-0.026, 0.026), (-0.040, 0.016), (-0.033, 0.016),
             (0.030, 0.028), (0.037, 0.028), (0.030, 0.019), (0.037, 0.019), (0.044, 0.028),
             (-0.030, -0.014), (-0.022, -0.014), (0.026, -0.020), (0.034, -0.020), (0.042, -0.020)]
    for i, (x, y) in enumerate(spots):
        p = cube(1, loc=(x, y, T / 2 + 0.00035), scale=(0.0020, 0.0011, 0.00035))
        put(p, grey if i % 3 else tan)
        parts.append(p)
    for x, y, r, h in [(-0.043, -0.006, 0.0055, 0.011), (-0.030, 0.004, 0.0042, 0.009)]:
        can = cyl(r, h, loc=(x, y, T / 2 + h / 2), verts=40, bevel=0.0004)
        put(can, tan)
        lid = cyl(r * 0.92, 0.0004, loc=(x, y, T / 2 + h), verts=40)
        put(lid, silver)
        parts += [can, lid]
    xtal = cube(1, loc=(0.030, 0.006, T / 2 + 0.0012), scale=(0.0055, 0.0030, 0.0012), bevel=0.0005)
    put(xtal, silver)
    parts.append(xtal)
    for i in range(10):                                     # pin header
        p = cube(1, loc=(-0.046 + i * 0.00254, 0.034, T / 2 + 0.0022), scale=(0.00035, 0.00035, 0.0022))
        put(p, gold)
        parts.append(p)
    hb = cube(1, loc=(-0.046 + 4.5 * 0.00254, 0.034, T / 2 + 0.0008), scale=(0.0125, 0.0013, 0.0008))
    put(hb, black)
    parts.append(hb)
    port = cube(1, loc=(0.040, -0.030, T / 2 + 0.0026), scale=(0.0065, 0.0055, 0.0026), bevel=0.0005)
    put(port, silver)
    shellw = cube(1, loc=(0.040, -0.036, T / 2 + 0.0026), scale=(0.0055, 0.0012, 0.0018))
    put(shellw, white)
    parts += [port, shellw]
    for sx in (-1, 1):                                      # mounting holes, drilled
        for sy in (-1, 1):
            h = cyl(0.0018, 0.004, loc=(sx * (W / 2 - 0.005), sy * (D / 2 - 0.005), 0), verts=20)
            b = board.modifiers.new("h", "BOOLEAN")
            b.operation, b.object = "DIFFERENCE", h
            bpy.context.view_layer.objects.active = board
            bpy.ops.object.modifier_apply(modifier=b.name)
            bpy.data.objects.remove(h, do_unlink=True)
            ringm = cyl(0.0028, 0.00022, loc=(sx * (W / 2 - 0.005), sy * (D / 2 - 0.005), T / 2 + 0.0001), verts=28)
            put(ringm, gold)
            parts.append(ringm)

    join([board] + parts, "DecoderBoard")
    export("pcb")


# ----------------------------------------------------------------- scope ----
def oscilloscope():
    reset()
    case = mat("case", (0.40, 0.43, 0.48), 0, 0.44)
    dark = mat("bezel", (0.09, 0.10, 0.12), 0, 0.42)
    knob = mat("knob", (0.14, 0.15, 0.17), 0, 0.38)
    metal = mat("metal", (0.72, 0.74, 0.77), 1.0, 0.28)
    screen_m = mat("screen", (0.02, 0.03, 0.04), 0, 0.18, emit=(0.02, 0.03, 0.04))
    accent = mat("accent", (0.85, 0.42, 0.12), 0, 0.45)

    W, H, D = 0.34, 0.20, 0.13
    body = cube(1, scale=(W / 2, D / 2, H / 2), bevel=0.006, segments=3)
    put(body, case, False, "Body")
    parts = []

    # recessed screen area on the front face (front = -Y)
    bez = cube(1, loc=(-0.055, -D / 2 - 0.0015, 0.012), scale=(0.098, 0.004, 0.068), bevel=0.003, segments=2)
    put(bez, dark)
    scr = cube(1, loc=(-0.055, -D / 2 - 0.0088, 0.012), scale=(0.089, 0.0011, 0.059))
    put(scr, screen_m, False, "Screen")   # sits proud of the bezel, or the bezel hides it
    parts += [bez, scr]

    # soft function buttons down the right of the screen and along the bottom
    for i in range(5):
        b = cube(1, loc=(0.052, -D / 2 - 0.003, 0.058 - i * 0.026), scale=(0.010, 0.002, 0.007), bevel=0.001)
        put(b, dark, True)
        parts.append(b)
    for i in range(6):
        b = cube(1, loc=(-0.128 + i * 0.026, -D / 2 - 0.003, -0.070), scale=(0.010, 0.002, 0.006), bevel=0.001)
        put(b, dark, True)
        parts.append(b)

    # the control cluster on the right: two big knobs, four small, with skirts and pointers
    def make_knob(x, z, r, h, tone):
        k = cyl(r, h, loc=(x, -D / 2 - h / 2, z), rot=(math.radians(90), 0, 0), verts=48, bevel=0.0008)
        put(k, tone)
        skirt = cyl(r * 1.22, 0.004, loc=(x, -D / 2 - 0.002, z), rot=(math.radians(90), 0, 0), verts=48)
        put(skirt, dark)
        ptr = cube(1, loc=(x, -D / 2 - h - 0.0005, z + r * 0.55), scale=(0.0012, 0.0012, r * 0.30))
        put(ptr, accent)
        return [k, skirt, ptr]
    parts += make_knob(0.112, 0.052, 0.020, 0.018, knob)
    parts += make_knob(0.112, -0.006, 0.020, 0.018, knob)
    for i, (x, z) in enumerate([(0.086, 0.052), (0.086, -0.006), (0.138, 0.052), (0.138, -0.006)]):
        parts += make_knob(x, z, 0.0085, 0.011, knob)

    # BNC inputs along the bottom right, with collars and centre pins
    for i in range(3):
        x = 0.078 + i * 0.030
        col = cyl(0.0105, 0.012, loc=(x, -D / 2 - 0.006, -0.062), rot=(math.radians(90), 0, 0), verts=40, bevel=0.0006)
        put(col, metal)
        pin = cyl(0.0016, 0.014, loc=(x, -D / 2 - 0.007, -0.062), rot=(math.radians(90), 0, 0), verts=20)
        put(pin, metal)
        ringm = cyl(0.0135, 0.002, loc=(x, -D / 2 - 0.001, -0.062), rot=(math.radians(90), 0, 0), verts=40)
        put(ringm, dark)
        parts += [col, pin, ringm]

    # carry handle, vents and feet
    handle = tube([(-0.10, 0, H / 2 + 0.002), (-0.09, -0.02, H / 2 + 0.040),
                   (0.04, -0.02, H / 2 + 0.040), (0.05, 0, H / 2 + 0.002)], 0.0055)
    put(handle, dark, True)
    parts.append(handle)
    for i in range(14):
        v = cube(1, loc=(-W / 2 + 0.03 + i * 0.020, D / 2 + 0.0008, 0.02), scale=(0.0035, 0.0012, 0.050))
        put(v, dark)
        parts.append(v)
    for sx in (-1, 1):
        for sy in (-1, 1):
            f = cyl(0.010, 0.008, loc=(sx * (W / 2 - 0.022), sy * (D / 2 - 0.022), -H / 2 - 0.003), verts=32, bevel=0.0015)
            put(f, dark)
            parts.append(f)

    join([body] + parts, "Oscilloscope")
    export("scope")


eeg_cap()
pcb()
oscilloscope()
print("hardware done", flush=True)
