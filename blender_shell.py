"""Build the brain hull the viewer draws, in Blender: the MaleCNS neuropil meshes are merged,
voxel remeshed into one closed surface, smoothed, decimated, and ambient occlusion is baked into a
per-vertex byte so the shell reads as a solid object in the browser without any lighting cost.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b -P fly-eeg/blender_shell.py

Writes sim/data/hull.bin: uint32 nv, nf; float32 (nv, 3) vertices; float32 (nv, 3) normals;
uint8 (nv,) ambient occlusion; uint32 (nf, 3) faces. Micrometres, same frame as the skeletons.
"""
import base64, json, struct, sys
from pathlib import Path

import bmesh, bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
NEUROPILS = HERE.parent / "nfly" / "data" / "anatomy" / "neuropils.json"
OUT = HERE / "sim" / "data" / "hull.bin"
OUTSIDE = {"ME", "LO", "LOP", "LA", "AME", "CV-anterior", "CRN"}    # optic lobes and neck: not simulated
VOXEL = 3.0          # remesh cell size in micrometres: small enough to keep the lobes, large enough to close gaps
TARGET_FACES = 40000
AO_RAYS, AO_DIST = 24, 60.0


def load_regions():
    rois = json.loads(NEUROPILS.read_text())["rois"]
    verts, faces = [], []
    for r in rois:
        if r.get("region") == "vnc" or r["name"].split("(")[0] in OUTSIDE:
            continue
        v = memoryview(base64.b64decode(r["vertices"])).cast("f")
        f = memoryview(base64.b64decode(r["faces"])).cast("I")
        off = len(verts)
        verts += [(v[3 * i], v[3 * i + 1], v[3 * i + 2]) for i in range(len(v) // 3)]
        faces += [(f[3 * i] + off, f[3 * i + 1] + off, f[3 * i + 2] + off) for i in range(len(f) // 3)]
    return verts, faces


def build_hull(verts, faces):
    mesh = bpy.data.meshes.new("hull")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("hull", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    m = obj.modifiers.new("remesh", "REMESH")           # one closed surface out of 90 overlapping shells
    m.mode, m.voxel_size, m.use_smooth_shade = "VOXEL", VOXEL, True
    bpy.ops.object.modifier_apply(modifier=m.name)
    m = obj.modifiers.new("smooth", "SMOOTH")
    m.factor, m.iterations = 1.0, 8
    bpy.ops.object.modifier_apply(modifier=m.name)
    if len(obj.data.polygons) > TARGET_FACES:
        m = obj.modifiers.new("decimate", "DECIMATE")
        m.ratio = TARGET_FACES / len(obj.data.polygons)
        bpy.ops.object.modifier_apply(modifier=m.name)
    return obj


def triangulate(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.calc_normals_split() if hasattr(obj.data, "calc_normals_split") else None


def ambient_occlusion(mesh):
    """Fraction of a hemisphere around each vertex normal that is not blocked, by ray casting."""
    import random
    random.seed(0)
    tree = BVHTree.FromPolygons([v.co.copy() for v in mesh.vertices],
                                [tuple(p.vertices) for p in mesh.polygons], all_triangles=True)
    dirs = []
    for i in range(AO_RAYS):                                  # cosine-ish hemisphere sample set, reused per vertex
        z = (i + 0.5) / AO_RAYS
        r, phi = (1 - z * z) ** 0.5, 2.399963 * i
        dirs.append(Vector((r * __import__("math").cos(phi), r * __import__("math").sin(phi), z)))
    ao = bytearray(len(mesh.vertices))
    for vi, v in enumerate(mesh.vertices):
        n = v.normal.normalized()
        up = Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))
        t = n.cross(up).normalized()
        b = n.cross(t)
        origin = v.co + n * 0.35
        open_rays = 0
        for d in dirs:
            w = t * d.x + b * d.y + n * d.z
            hit = tree.ray_cast(origin, w, AO_DIST)
            if hit[0] is None:
                open_rays += 1
        ao[vi] = int(255 * (open_rays / AO_RAYS) ** 0.8)
    return ao


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    verts, faces = load_regions()
    print(f"neuropils: {len(verts):,} vertices, {len(faces):,} faces", flush=True)
    obj = build_hull(verts, faces)
    triangulate(obj)
    mesh = obj.data
    mesh.calc_normals() if hasattr(mesh, "calc_normals") else None
    nv, nf = len(mesh.vertices), len(mesh.polygons)
    print(f"hull: {nv:,} vertices, {nf:,} faces", flush=True)
    ao = ambient_occlusion(mesh)
    print(f"ambient occlusion: mean {sum(ao) / len(ao):.0f}/255", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(struct.pack("<II", nv, nf))
        fh.write(b"".join(struct.pack("<3f", *v.co) for v in mesh.vertices))
        fh.write(b"".join(struct.pack("<3f", *v.normal) for v in mesh.vertices))
        fh.write(bytes(ao))
        fh.write(b"".join(struct.pack("<3I", *p.vertices) for p in mesh.polygons))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


main()
